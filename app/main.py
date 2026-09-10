"""FastAPI application exposing the already-trained demand forecasting engine."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.demand import router as demand_router
from app.api.routes.stockout import router as stockout_router
from app.config import ApiSettings
from app.prediction.demand_predictor import DemandModelUnavailableError, load_deployment_model


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / "backend" / ".env", override=False)
load_dotenv(ROOT / ".env", override=False)
logger = logging.getLogger(__name__)


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    """Create the app; the pipeline is loaded once during lifespan startup."""
    settings = settings or ApiSettings.from_environment()
    logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO))

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.settings = settings
        application.state.demand_model_bundle = None
        try:
            application.state.demand_model_bundle = load_deployment_model(settings.model_path, settings.metadata_path)
            logger.info("Demand deployment model loaded at application startup.")
        except DemandModelUnavailableError:
            logger.exception("Demand deployment model is unavailable at startup.")
        yield

    application = FastAPI(
        title="MADIS Demand Forecast API",
        version="1.0.0",
        description="Read-only J+1 to J+7 demand forecasting using the frozen deployment pipeline.",
        lifespan=lifespan,
    )
    if settings.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )
    application.include_router(demand_router)
    application.include_router(stockout_router)
    return application


app = create_app()
