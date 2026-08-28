"""Build a daily demand dataset from MADIS PostgreSQL inventory movements."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
import os
from pathlib import Path
import statistics
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import psycopg


DEMAND_MOVEMENT_TYPES = ("SALE", "PROMOTION_GIFT")


class DatasetValidationError(RuntimeError):
    """Raised when a dataset does not meet its required daily-grain contract."""


@dataclass(frozen=True)
class DateRange:
    start_date: date
    end_date: date

    def __post_init__(self) -> None:
        if self.end_date < self.start_date:
            raise ValueError("La date de fin doit être postérieure ou égale à la date de début.")

    @property
    def day_count(self) -> int:
        return (self.end_date - self.start_date).days + 1

    @property
    def end_exclusive(self) -> date:
        return self.end_date + timedelta(days=1)

    def dates(self) -> list[date]:
        return [self.start_date + timedelta(days=index) for index in range(self.day_count)]


@dataclass(frozen=True)
class DemandRow:
    day: date
    product_id: int
    product_reference: str
    demand_quantity: int


@dataclass(frozen=True)
class ProductDemandStats:
    product_id: int
    product_reference: str
    day_count: int
    total_demand: int
    average_daily_demand: float
    minimum_demand: int
    maximum_demand: int
    standard_deviation: float
    zero_days: int
    zero_day_percentage: float


@dataclass(frozen=True)
class DatasetStats:
    product_count: int
    day_count: int
    expected_row_count: int
    row_count: int
    minimum_date: date
    maximum_date: date
    total_sale_quantity: int
    total_promotion_gift_quantity: int
    total_demand_quantity: int
    average_demand: float
    minimum_demand: int
    maximum_demand: int
    zero_rows: int
    zero_row_percentage: float
    duplicate_count: int
    null_date_count: int
    null_product_id_count: int
    null_demand_count: int
    products: tuple[ProductDemandStats, ...]


@dataclass(frozen=True)
class DemandDataset:
    rows: tuple[DemandRow, ...]
    stats: DatasetStats


def postgres_connection_settings(database_url: str) -> tuple[str, str]:
    """Use DATABASE_URL while translating Prisma's optional ``schema`` parameter."""
    parsed = urlsplit(database_url)
    query = parse_qsl(parsed.query, keep_blank_values=True)
    schema = next((value for key, value in query if key == "schema"), None)
    filtered_query = [(key, value) for key, value in query if key != "schema"]
    if schema and not schema.replace("_", "").isalnum():
        raise ValueError("Le paramètre schema de DATABASE_URL est invalide.")
    normalized_url = urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(filtered_query), parsed.fragment),
    )
    options = "-c default_transaction_read_only=on"
    if schema:
        options = f"-c search_path={schema} {options}"
    return normalized_url, options


def connect_read_only(database_url: str) -> psycopg.Connection:
    """Create a PostgreSQL connection whose transactions are read-only."""
    connection_url, options = postgres_connection_settings(database_url)
    connection = psycopg.connect(connection_url, options=options)
    connection.execute("SET TRANSACTION READ ONLY")
    return connection


def load_daily_aggregates(
    connection: psycopg.Connection,
    period: DateRange,
    synthetic_batch: str | None = None,
) -> tuple[dict[tuple[int, date], tuple[str, int, int]], int, int]:
    """Read and aggregate only SALE and PROMOTION_GIFT movements by day/product."""
    marker_filter = ""
    parameters: list[object] = [period.start_date, period.end_exclusive]
    if synthetic_batch:
        marker_filter = 'AND cart.reason = %s'
        parameters.append(synthetic_batch)

    query = f'''
        SELECT
            movement."createdAt"::date AS demand_date,
            inventory."productId" AS product_id,
            product.reference AS product_reference,
            SUM(CASE WHEN movement.type = 'SALE' THEN movement."outgoingQuantity" ELSE 0 END) AS sale_quantity,
            SUM(CASE WHEN movement.type = 'PROMOTION_GIFT' THEN movement."outgoingQuantity" ELSE 0 END) AS promotion_gift_quantity
        FROM "InventoryMovement" movement
        JOIN "Inventories" inventory ON inventory."ID" = movement."inventoryId"
        JOIN "Products" product ON product."ID" = inventory."productId"
        LEFT JOIN "Carts" cart ON cart."ID" = movement."cartId"
        WHERE movement.type IN ('SALE', 'PROMOTION_GIFT')
          AND movement."createdAt" >= %s
          AND movement."createdAt" < %s
          {marker_filter}
        GROUP BY movement."createdAt"::date, inventory."productId", product.reference
        ORDER BY inventory."productId", movement."createdAt"::date
    '''
    aggregates: dict[tuple[int, date], tuple[str, int, int]] = {}
    total_sale = 0
    total_gift = 0
    with connection.cursor() as cursor:
        cursor.execute(query, parameters)
        for demand_date, product_id, reference, sale_quantity, gift_quantity in cursor.fetchall():
            if None in (demand_date, product_id, reference, sale_quantity, gift_quantity):
                raise DatasetValidationError("L'agrégation PostgreSQL contient une valeur nulle inattendue.")
            if sale_quantity < 0 or gift_quantity < 0:
                raise DatasetValidationError("Une quantité de demande agrégée ne peut pas être négative.")
            key = (int(product_id), demand_date)
            if key in aggregates:
                raise DatasetValidationError(f"Doublon d'agrégation détecté pour {key}.")
            aggregates[key] = (str(reference), int(sale_quantity), int(gift_quantity))
            total_sale += int(sale_quantity)
            total_gift += int(gift_quantity)
    return aggregates, total_sale, total_gift


def build_daily_dataset(
    connection: psycopg.Connection,
    period: DateRange,
    synthetic_batch: str | None = None,
) -> DemandDataset:
    """Create the complete product × date grid and fill missing demand with zero."""
    aggregates, total_sale, total_gift = load_daily_aggregates(
        connection,
        period,
        synthetic_batch,
    )
    references = {
        product_id: reference
        for (product_id, _), (reference, _, _) in aggregates.items()
    }
    if not references:
        raise DatasetValidationError(
            "Aucun mouvement SALE ou PROMOTION_GIFT ne correspond à la période demandée.",
        )

    rows: list[DemandRow] = []
    for product_id in sorted(references):
        for current_day in period.dates():
            _, sale_quantity, gift_quantity = aggregates.get(
                (product_id, current_day),
                (references[product_id], 0, 0),
            )
            rows.append(
                DemandRow(
                    day=current_day,
                    product_id=product_id,
                    product_reference=references[product_id],
                    demand_quantity=sale_quantity + gift_quantity,
                ),
            )

    stats = validate_and_summarize(rows, period, total_sale, total_gift)
    return DemandDataset(tuple(rows), stats)


def extract_demand_dataset(
    start_date: date | str,
    end_date: date | str,
    synthetic_batch: str | None = None,
    database_url: str | None = None,
) -> pd.DataFrame:
    """Return the validated daily demand dataset directly as a DataFrame.

    ``synthetic_batch`` is an optional exact filter on ``Carts.reason``.  When
    omitted, the function is the normal production extractor and aggregates all
    business SALE and PROMOTION_GIFT movements in the requested period.
    """
    period = DateRange(_coerce_date(start_date), _coerce_date(end_date))
    connection_string = database_url or os.getenv("DATABASE_URL")
    if not connection_string:
        raise ValueError("DATABASE_URL est obligatoire.")
    with connect_read_only(connection_string) as connection:
        dataset = build_daily_dataset(connection, period, synthetic_batch)
    return demand_dataset_to_dataframe(dataset)


def get_latest_demand_date(
    synthetic_batch: str | None = None,
    database_url: str | None = None,
) -> date:
    """Return the newest SALE/PROMOTION_GIFT date available to the extractor."""
    connection_string = database_url or os.getenv("DATABASE_URL")
    if not connection_string:
        raise ValueError("DATABASE_URL est obligatoire.")
    marker_filter = ""
    parameters: list[object] = []
    if synthetic_batch:
        marker_filter = 'AND cart.reason = %s'
        parameters.append(synthetic_batch)
    query = f'''
        SELECT MAX(movement."createdAt"::date)
        FROM "InventoryMovement" movement
        LEFT JOIN "Carts" cart ON cart."ID" = movement."cartId"
        WHERE movement.type IN ('SALE', 'PROMOTION_GIFT')
          {marker_filter}
    '''
    with connect_read_only(connection_string) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters)
            latest_date = cursor.fetchone()[0]
    if latest_date is None:
        raise DatasetValidationError("Aucun mouvement de demande n'est disponible pour déterminer as_of_date.")
    return latest_date


def get_active_product_reference(
    product_id: int,
    database_url: str | None = None,
) -> str | None:
    """Return an active product reference, without creating or changing products."""
    connection_string = database_url or os.getenv("DATABASE_URL")
    if not connection_string:
        raise ValueError("DATABASE_URL est obligatoire.")
    with connect_read_only(connection_string) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                '''
                SELECT reference
                FROM "Products"
                WHERE "ID" = %s
                  AND "deletedAt" IS NULL
                ''',
                (product_id,),
            )
            row = cursor.fetchone()
    return str(row[0]) if row else None


def _coerce_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as error:
            raise ValueError("Les dates doivent respecter le format YYYY-MM-DD.") from error
    raise TypeError("Les dates doivent être des date ou des chaînes YYYY-MM-DD.")


def demand_dataset_to_dataframe(dataset: DemandDataset) -> pd.DataFrame:
    """Convert the validated domain dataset without writing an intermediate CSV."""
    dataframe = pd.DataFrame(
        {
            "date": [row.day for row in dataset.rows],
            "productId": [row.product_id for row in dataset.rows],
            "productReference": [row.product_reference for row in dataset.rows],
            "demandQty": [row.demand_quantity for row in dataset.rows],
        },
    ).sort_values(["productId", "date"], kind="stable", ignore_index=True)
    dataframe.attrs["dataset_stats"] = dataset.stats
    return dataframe


def validate_and_summarize(
    rows: list[DemandRow],
    period: DateRange,
    total_sale: int,
    total_gift: int,
) -> DatasetStats:
    """Validate uniqueness, completeness, types and accounting reconciliation."""
    null_date_count = sum(row.day is None for row in rows)
    null_product_id_count = sum(row.product_id is None for row in rows)
    null_demand_count = sum(row.demand_quantity is None for row in rows)
    if null_date_count or null_product_id_count or null_demand_count:
        raise DatasetValidationError("Le dataset contient une valeur obligatoire manquante.")
    if any(not isinstance(row.product_id, int) for row in rows):
        raise DatasetValidationError("productId doit être un entier PostgreSQL.")
    if any(not isinstance(row.demand_quantity, int) or row.demand_quantity < 0 for row in rows):
        raise DatasetValidationError("demandQty doit être un entier supérieur ou égal à zéro.")

    keys = [(row.product_id, row.day) for row in rows]
    duplicate_count = len(keys) - len(set(keys))
    if duplicate_count:
        raise DatasetValidationError(f"{duplicate_count} doublon(s) date × productId détecté(s).")

    products = sorted({row.product_id for row in rows})
    expected_row_count = len(products) * period.day_count
    if len(rows) != expected_row_count:
        raise DatasetValidationError(
            f"Grille incomplète : {len(rows)} ligne(s), {expected_row_count} attendue(s).",
        )
    expected_days = set(period.dates())
    for product_id in products:
        product_days = {row.day for row in rows if row.product_id == product_id}
        if product_days != expected_days:
            raise DatasetValidationError(
                f"La couverture chronologique du produit {product_id} est incomplète.",
            )

    total_demand = sum(row.demand_quantity for row in rows)
    if total_demand != total_sale + total_gift:
        raise DatasetValidationError(
            "Incohérence comptable : demandQty doit être égal à SALE + PROMOTION_GIFT.",
        )
    by_product: dict[int, list[DemandRow]] = defaultdict(list)
    for row in rows:
        by_product[row.product_id].append(row)
    product_stats: list[ProductDemandStats] = []
    for product_id in products:
        product_rows = by_product[product_id]
        demands = [row.demand_quantity for row in product_rows]
        zero_days = sum(value == 0 for value in demands)
        product_stats.append(
            ProductDemandStats(
                product_id=product_id,
                product_reference=product_rows[0].product_reference,
                day_count=len(product_rows),
                total_demand=sum(demands),
                average_daily_demand=sum(demands) / len(demands),
                minimum_demand=min(demands),
                maximum_demand=max(demands),
                standard_deviation=statistics.pstdev(demands),
                zero_days=zero_days,
                zero_day_percentage=zero_days / len(demands) * 100,
            ),
        )
    demand_values = [row.demand_quantity for row in rows]
    zero_rows = sum(value == 0 for value in demand_values)
    return DatasetStats(
        product_count=len(products),
        day_count=period.day_count,
        expected_row_count=expected_row_count,
        row_count=len(rows),
        minimum_date=min(row.day for row in rows),
        maximum_date=max(row.day for row in rows),
        total_sale_quantity=total_sale,
        total_promotion_gift_quantity=total_gift,
        total_demand_quantity=total_demand,
        average_demand=total_demand / len(rows),
        minimum_demand=min(demand_values),
        maximum_demand=max(demand_values),
        zero_rows=zero_rows,
        zero_row_percentage=zero_rows / len(rows) * 100,
        duplicate_count=duplicate_count,
        null_date_count=null_date_count,
        null_product_id_count=null_product_id_count,
        null_demand_count=null_demand_count,
        products=tuple(product_stats),
    )


def export_dataframe_csv(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Write an extracted DataFrame as an optional UTF-8 CSV export."""
    expected_columns = ["date", "productId", "productReference", "demandQty"]
    if list(dataframe.columns) != expected_columns:
        raise DatasetValidationError("Les colonnes du DataFrame de demande sont invalides.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=expected_columns,
        )
        writer.writeheader()
        for row in dataframe.itertuples(index=False):
            writer.writerow(
                {
                    "date": row.date.isoformat(),
                    "productId": row.productId,
                    "productReference": row.productReference,
                    "demandQty": row.demandQty,
                },
            )
