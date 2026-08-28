"""Thin HTTP layer delegating stockout forecasting to stockout_predictor."""

from __future__ import annotations

from datetime import date
import logging

from fastapi import APIRouter, Query, Request

from app.api.routes.common import get_settings_and_bundle, run_prediction
from app.api.schemas.demand import DemandForecastBatchRequest
from app.api.schemas.stockout import StockoutForecastBatchResponse, StockoutForecastResponse
from app.prediction.stockout_predictor import predict_stockout


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Stockout forecasting"])


def _forecast(request: Request, product_id: int, days: int, as_of_date: date | None) -> dict:
    settings, bundle = get_settings_and_bundle(request)
    logger.info("Stockout forecast requested: product_id=%s days=%s", product_id, days)
    return run_prediction(
        lambda: predict_stockout(
            product_id=product_id,
            forecast_days=days,
            as_of_date=as_of_date,
            synthetic_batch=settings.synthetic_batch,
            database_url=settings.database_url,
            model_bundle=bundle,
        ),
        product_id,
        "stockout forecast",
    )


@router.get(
    "/stockout/forecast/{product_id}",
    response_model=StockoutForecastResponse,
    summary="Forecast one product's stock evolution and possible stockout",
    description=(
        "Prévoit l'évolution du stock d'un produit à partir de la demande prédite "
        "et indique une éventuelle rupture dans l'horizon demandé. "
        "`expiredQuantity` représente le reliquat de lots devenu inutilisable ce jour."
    ),
)
def forecast_product_stockout(
    request: Request,
    product_id: int,
    days: int = Query(default=7, ge=1, le=7, description="Forecast horizon, from 1 to 7 days."),
    as_of_date: date | None = Query(default=None, description="Final known demand date (YYYY-MM-DD)."),
) -> dict:
    return _forecast(request, product_id, days, as_of_date)


@router.post(
    "/stockout/forecast",
    response_model=StockoutForecastBatchResponse,
    summary="Forecast stockout risk for multiple products with one shared horizon",
)
def forecast_products_stockout(request: Request, payload: DemandForecastBatchRequest) -> dict[str, object]:
    return {
        "forecasts": [
            _forecast(request, product_id, payload.days, payload.as_of_date)
            for product_id in payload.product_ids
        ],
    }
