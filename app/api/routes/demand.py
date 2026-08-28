"""Thin HTTP layer delegating all forecasting work to demand_predictor."""

from __future__ import annotations

from datetime import date
import logging

from fastapi import APIRouter, Query, Request

from app.api.routes.common import get_settings_and_bundle, run_prediction

from app.api.schemas.demand import (
    DemandForecastBatchRequest,
    DemandForecastBatchResponse,
    DemandForecastResponse,
    DemandModelInfoResponse,
    HealthResponse,
)
from app.prediction.demand_predictor import predict_demand


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Demand forecasting"])


def _forecast(request: Request, product_id: int, days: int, as_of_date: date | None) -> dict:
    settings, bundle = get_settings_and_bundle(request)
    logger.info("Demand forecast requested: product_id=%s days=%s", product_id, days)
    return run_prediction(
        lambda: predict_demand(
            product_id=product_id,
            forecast_days=days,
            as_of_date=as_of_date,
            synthetic_batch=settings.synthetic_batch,
            database_url=settings.database_url,
            model_bundle=bundle,
        ),
        product_id,
        "demand forecast",
    )


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health(request: Request) -> dict[str, object]:
    return {"status": "ok", "demandModelLoaded": request.app.state.demand_model_bundle is not None}


@router.get("/demand/model-info", response_model=DemandModelInfoResponse)
def model_info(request: Request) -> dict[str, object]:
    _, bundle = get_settings_and_bundle(request)
    metadata = bundle.metadata
    return {
        "modelName": metadata["modelName"],
        "modelPurpose": metadata["modelPurpose"],
        "trainingStartDate": metadata["trainingStartDate"],
        "trainingEndDate": metadata["trainingEndDate"],
        "syntheticData": metadata["syntheticData"],
        "maximumForecastDays": 7,
        "featureCount": len(metadata["features"]),
    }


@router.get(
    "/demand/forecast/{product_id}",
    response_model=DemandForecastResponse,
    summary="Forecast one product's demand for the next 1 to 7 days",
)
def forecast_product(
    request: Request,
    product_id: int,
    days: int = Query(default=7, ge=1, le=7, description="Forecast horizon, from 1 to 7 days."),
    as_of_date: date | None = Query(default=None, description="Final known demand date (YYYY-MM-DD)."),
) -> dict:
    return _forecast(request, product_id, days, as_of_date)


@router.post(
    "/demand/forecast",
    response_model=DemandForecastBatchResponse,
    summary="Forecast multiple products with one shared horizon",
)
def forecast_products(request: Request, payload: DemandForecastBatchRequest) -> dict[str, object]:
    return {
        "forecasts": [
            _forecast(request, product_id, payload.days, payload.as_of_date)
            for product_id in payload.product_ids
        ],
    }
