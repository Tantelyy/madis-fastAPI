"""Thin HTTP layer delegating all forecasting work to demand_predictor."""

from __future__ import annotations

from datetime import date
import logging

from fastapi import APIRouter, HTTPException, Query, Request, status

from app.api.schemas.demand import (
    DemandForecastBatchRequest,
    DemandForecastBatchResponse,
    DemandForecastResponse,
    DemandModelInfoResponse,
    HealthResponse,
)
from app.config import ApiSettings
from app.prediction.demand_predictor import (
    DemandModelUnavailableError,
    InsufficientHistoryError,
    PredictionValidationError,
    ProductNotFoundError,
    UnknownModelProductError,
    predict_demand,
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Demand forecasting"])


def _settings_and_bundle(request: Request):
    settings: ApiSettings = request.app.state.settings
    bundle = request.app.state.demand_model_bundle
    if bundle is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Demand model is unavailable")
    return settings, bundle


def _forecast(request: Request, product_id: int, days: int, as_of_date: date | None) -> dict:
    settings, bundle = _settings_and_bundle(request)
    logger.info("Demand forecast requested: product_id=%s days=%s", product_id, days)
    try:
        return predict_demand(
            product_id=product_id,
            forecast_days=days,
            as_of_date=as_of_date,
            synthetic_batch=settings.synthetic_batch,
            database_url=settings.database_url,
            model_bundle=bundle,
        )
    except ProductNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Product {product_id} not found") from error
    except UnknownModelProductError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Product has not been seen by the current demand model",
        ) from error
    except InsufficientHistoryError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error
    except DemandModelUnavailableError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Demand model is unavailable") from error
    except PredictionValidationError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error
    except Exception as error:
        logger.exception("Unexpected demand forecast error for product_id=%s", product_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Demand forecast failed") from error


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health(request: Request) -> dict[str, object]:
    return {"status": "ok", "demandModelLoaded": request.app.state.demand_model_bundle is not None}


@router.get("/demand/model-info", response_model=DemandModelInfoResponse)
def model_info(request: Request) -> dict[str, object]:
    _, bundle = _settings_and_bundle(request)
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
