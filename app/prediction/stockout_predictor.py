"""Read-only product-level stockout projection using the existing demand predictor."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import os
from typing import Any, Iterable

from app.data.extractor import connect_read_only, get_active_product_reference
from app.prediction.demand_predictor import (
    DeploymentModelBundle,
    PredictionValidationError,
    predict_demand,
)


MAX_FORECAST_DAYS = 7


class StockUnavailableError(PredictionValidationError):
    """Raised when a stock snapshot cannot be reconstructed safely."""


@dataclass(frozen=True)
class InventoryLotSnapshot:
    inventory_id: int
    available_quantity: float
    created_at: datetime
    expired_at: datetime | None


@dataclass(frozen=True)
class ProductStockSnapshot:
    product_id: int
    product_reference: str
    as_of_date: date
    available_stock: float
    usable_lots: tuple[InventoryLotSnapshot, ...]


def get_available_stock(
    product_id: int,
    as_of_date: date | str,
    database_url: str | None = None,
) -> ProductStockSnapshot:
    """Reconstruct sellable product stock at the end of an historical day.

    Inventory is the lot table. For each lot existing by ``as_of_date``, the
    balance is reconstructed from the persisted movement quantities, rather
    than from today's ``remainingQuantity``. A lot is usable when its balance
    is positive and it is not expired at this date-level end-of-day cutoff.
    """
    if not isinstance(product_id, int) or isinstance(product_id, bool) or product_id <= 0:
        raise StockUnavailableError("product_id doit être un entier strictement positif.")
    snapshot_date = _coerce_date(as_of_date)
    database_url = database_url or os.getenv("DATABASE_URL")
    if not database_url:
        raise StockUnavailableError("DATABASE_URL est obligatoire.")
    reference = get_active_product_reference(product_id, database_url)
    if reference is None:
        from app.prediction.demand_predictor import ProductNotFoundError

        raise ProductNotFoundError(f"Le produit actif {product_id} est introuvable.")

    end_exclusive = snapshot_date + timedelta(days=1)
    query = '''
        SELECT
            inventory."ID",
            inventory."createdAt",
            inventory."expiredAt",
            COALESCE(
                SUM(movement."incomingQuantity" - movement."outgoingQuantity"),
                0
            ) AS reconstructed_quantity
        FROM "Inventories" inventory
        JOIN "Products" product ON product."ID" = inventory."productId"
        LEFT JOIN "InventoryMovement" movement
            ON movement."inventoryId" = inventory."ID"
           AND movement."createdAt" < %s
        WHERE inventory."productId" = %s
          AND product."deletedAt" IS NULL
          AND inventory."createdAt" < %s
        GROUP BY inventory."ID", inventory."createdAt", inventory."expiredAt"
        ORDER BY inventory."createdAt", inventory."ID"
    '''
    cutoff = datetime.combine(end_exclusive, time.min)
    usable_lots: list[InventoryLotSnapshot] = []
    with connect_read_only(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (cutoff, product_id, cutoff))
            for inventory_id, created_at, expired_at, reconstructed_quantity in cursor.fetchall():
                quantity = float(reconstructed_quantity)
                if quantity < 0:
                    raise StockUnavailableError(
                        f"Le lot {inventory_id} du produit {product_id} a un stock historique négatif.",
                    )
                # The Nest sales service accepts a lot while expiredAt >= now.
                # At daily granularity, the snapshot is taken at the end of the
                # requested date, represented by the next midnight cutoff.
                is_usable = quantity > 0 and (expired_at is None or expired_at >= cutoff)
                if is_usable:
                    usable_lots.append(
                        InventoryLotSnapshot(int(inventory_id), quantity, created_at, expired_at),
                    )
    return ProductStockSnapshot(
        product_id=product_id,
        product_reference=reference,
        as_of_date=snapshot_date,
        available_stock=float(sum(lot.available_quantity for lot in usable_lots)),
        usable_lots=tuple(usable_lots),
    )


def calculate_stock_projection(
    current_stock: float,
    demand_predictions: Iterable[dict[str, object]],
) -> dict[str, object]:
    """Project stock day by day using unrounded predictedDemand values only."""
    if current_stock < 0:
        raise ValueError("current_stock ne peut pas être négatif.")
    opening_stock = float(current_stock)
    already_out_of_stock = opening_stock <= 0
    first_stockout_date: str | None = None
    first_stockout_day: int | None = None
    total_shortage = 0.0
    daily_projection: list[dict[str, object]] = []
    total_predicted_demand = 0.0

    for day_index, prediction in enumerate(demand_predictions, start=1):
        forecast_date = str(prediction["date"])
        predicted_demand = float(prediction["predictedDemand"])
        if predicted_demand < 0:
            raise ValueError("predictedDemand ne peut pas être négatif.")
        planned_incoming = 0.0
        projected_raw = opening_stock + planned_incoming - predicted_demand
        projected_ending = max(0.0, projected_raw)
        shortage = max(0.0, -projected_raw)
        stockout = projected_raw <= 0.0
        if stockout and first_stockout_date is None:
            first_stockout_date = forecast_date
            first_stockout_day = day_index
        daily_projection.append(
            {
                "date": forecast_date,
                "openingStock": opening_stock,
                "predictedDemand": predicted_demand,
                "plannedIncomingQuantity": planned_incoming,
                "projectedEndingStockRaw": projected_raw,
                "projectedEndingStock": projected_ending,
                "shortageQuantity": shortage,
                "stockout": stockout,
            },
        )
        total_predicted_demand += predicted_demand
        total_shortage += shortage
        opening_stock = projected_ending

    return {
        "totalPredictedDemand": total_predicted_demand,
        "alreadyOutOfStock": already_out_of_stock,
        "stockoutExpected": already_out_of_stock or first_stockout_date is not None,
        "predictedStockoutDate": first_stockout_date,
        "daysUntilStockout": first_stockout_day,
        "remainingStockAfterHorizon": opening_stock,
        "totalShortageAfterHorizon": total_shortage,
        "dailyProjection": daily_projection,
    }


def calculate_lot_stock_projection(
    lots: Iterable[InventoryLotSnapshot],
    demand_predictions: Iterable[dict[str, object]],
) -> dict[str, object]:
    """Project stock at lot level, excluding lots that expire during the horizon.

    The sales service first applies an optional promotion-lot priority and then
    uses ``createdAt`` / inventory id order. Demand forecasts are product-level
    totals and do not identify an offer or a cart line, so that optional
    promotion-specific allocation cannot be reconstructed faithfully here. The
    deterministic base order shared by the service is therefore FIFO by
    ``createdAt`` then inventory id.

    A daily forecast has no transaction time. We consequently use the existing
    end-of-day snapshot convention: a lot must still be sellable at the next
    midnight to cover the complete forecast day. This matches the backend rule
    ``expiredAt >= now`` and avoids counting residual stock after its expiry.
    """
    projected_lots = [
        {
            "inventory_id": lot.inventory_id,
            "remaining": float(lot.available_quantity),
            "created_at": lot.created_at,
            "expired_at": lot.expired_at,
        }
        for lot in lots
    ]
    if any(lot["remaining"] < 0 for lot in projected_lots):
        raise ValueError("La quantité d'un lot ne peut pas être négative.")
    projected_lots.sort(key=lambda lot: (lot["created_at"], lot["inventory_id"]))

    initial_stock = float(sum(lot["remaining"] for lot in projected_lots))
    already_out_of_stock = initial_stock <= 0
    first_stockout_date: str | None = None
    first_stockout_day: int | None = None
    total_shortage = 0.0
    total_predicted_demand = 0.0
    daily_projection: list[dict[str, object]] = []

    for day_index, prediction in enumerate(demand_predictions, start=1):
        forecast_date = _coerce_date(str(prediction["date"]))
        predicted_demand = float(prediction["predictedDemand"])
        if predicted_demand < 0:
            raise ValueError("predictedDemand ne peut pas être négatif.")

        end_of_day = datetime.combine(forecast_date + timedelta(days=1), time.min)
        expired_quantity = 0.0
        usable_lots: list[dict[str, object]] = []
        for lot in projected_lots:
            if lot["remaining"] <= 0:
                continue
            expired_at = lot["expired_at"]
            if expired_at is not None and expired_at < end_of_day:
                expired_quantity += float(lot["remaining"])
                lot["remaining"] = 0.0
            else:
                usable_lots.append(lot)

        opening_stock = float(sum(float(lot["remaining"]) for lot in usable_lots))
        planned_incoming = 0.0
        projected_raw = opening_stock + planned_incoming - predicted_demand
        allocated_demand = min(opening_stock, predicted_demand)
        to_allocate = allocated_demand
        for lot in usable_lots:
            if to_allocate <= 0:
                break
            consumed = min(float(lot["remaining"]), to_allocate)
            lot["remaining"] = float(lot["remaining"]) - consumed
            to_allocate -= consumed

        projected_ending = max(0.0, projected_raw)
        shortage = max(0.0, -projected_raw)
        stockout = projected_raw <= 0.0
        if stockout and first_stockout_date is None:
            first_stockout_date = forecast_date.isoformat()
            first_stockout_day = day_index
        daily_projection.append(
            {
                "date": forecast_date.isoformat(),
                "openingStock": opening_stock,
                "predictedDemand": predicted_demand,
                "plannedIncomingQuantity": planned_incoming,
                "expiredQuantity": expired_quantity,
                "projectedEndingStockRaw": projected_raw,
                "projectedEndingStock": projected_ending,
                "shortageQuantity": shortage,
                "stockout": stockout,
            },
        )
        total_predicted_demand += predicted_demand
        total_shortage += shortage

    remaining_stock = float(sum(float(lot["remaining"]) for lot in projected_lots))
    return {
        "totalPredictedDemand": total_predicted_demand,
        "alreadyOutOfStock": already_out_of_stock,
        "stockoutExpected": already_out_of_stock or first_stockout_date is not None,
        "predictedStockoutDate": first_stockout_date,
        "daysUntilStockout": first_stockout_day,
        "remainingStockAfterHorizon": remaining_stock,
        "totalShortageAfterHorizon": total_shortage,
        "dailyProjection": daily_projection,
    }


def predict_stockout(
    product_id: int,
    forecast_days: int = 7,
    as_of_date: date | str | None = None,
    synthetic_batch: str | None = None,
    database_url: str | None = None,
    model_bundle: DeploymentModelBundle | None = None,
) -> dict[str, object]:
    """Forecast a product's stockout over the same 1-to-7-day demand horizon."""
    if not isinstance(forecast_days, int) or isinstance(forecast_days, bool) or not 1 <= forecast_days <= MAX_FORECAST_DAYS:
        raise PredictionValidationError(f"forecast_days doit être compris entre 1 et {MAX_FORECAST_DAYS}.")
    demand_forecast = predict_demand(
        product_id=product_id,
        forecast_days=forecast_days,
        as_of_date=as_of_date,
        synthetic_batch=synthetic_batch,
        database_url=database_url,
        model_bundle=model_bundle,
    )
    resolved_as_of = _coerce_date(str(demand_forecast["asOfDate"]))
    stock_snapshot = get_available_stock(product_id, resolved_as_of, database_url)
    projection = calculate_lot_stock_projection(
        stock_snapshot.usable_lots,
        demand_forecast["predictions"],
    )
    if projection["alreadyOutOfStock"]:
        projection["predictedStockoutDate"] = resolved_as_of.isoformat()
        projection["daysUntilStockout"] = 0
    return {
        "productId": product_id,
        "productReference": demand_forecast["productReference"],
        "asOfDate": demand_forecast["asOfDate"],
        "forecastDays": forecast_days,
        "currentStock": stock_snapshot.available_stock,
        **projection,
    }


def _coerce_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as error:
            raise StockUnavailableError("Les dates doivent respecter le format YYYY-MM-DD.") from error
    raise StockUnavailableError("as_of_date doit être une date ou une chaîne YYYY-MM-DD.")
