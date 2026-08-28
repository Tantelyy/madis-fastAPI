"""Shared request state and business-error translation for prediction routes."""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import TypeVar

from fastapi import HTTPException, Request, status

from app.config import ApiSettings
from app.prediction.demand_predictor import (
    DemandModelUnavailableError,
    InsufficientHistoryError,
    PredictionValidationError,
    ProductNotFoundError,
    UnknownModelProductError,
)


logger = logging.getLogger(__name__)
Result = TypeVar("Result")


def get_settings_and_bundle(request: Request):
    """Return app-scoped settings and the deployment pipeline loaded at startup."""
    settings: ApiSettings = request.app.state.settings
    bundle = request.app.state.demand_model_bundle
    if bundle is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Demand model is unavailable")
    return settings, bundle


def run_prediction(operation: Callable[[], Result], product_id: int, forecast_name: str) -> Result:
    """Translate shared domain errors to the stable HTTP error contract."""
    try:
        return operation()
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
        logger.exception("Unexpected %s error for product_id=%s", forecast_name, product_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{forecast_name.capitalize()} failed",
        ) from error
