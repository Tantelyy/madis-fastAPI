"""Public Pydantic schemas for demand forecasting endpoints."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class DemandForecastItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    date: date
    predicted_demand: float = Field(alias="predictedDemand")
    predicted_quantity: int = Field(alias="predictedQuantity")


class DemandForecastResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    product_id: int = Field(alias="productId")
    product_reference: str = Field(alias="productReference")
    as_of_date: date = Field(alias="asOfDate")
    forecast_days: int = Field(alias="forecastDays")
    predictions: list[DemandForecastItem]


class DemandForecastBatchRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    product_ids: list[int] = Field(alias="productIds", min_length=1, max_length=100)
    days: int = Field(default=7, ge=1, le=7)
    as_of_date: date | None = Field(default=None, alias="asOfDate")


class DemandForecastBatchResponse(BaseModel):
    forecasts: list[DemandForecastResponse]


class HealthResponse(BaseModel):
    status: str
    demand_model_loaded: bool = Field(alias="demandModelLoaded")


class DemandModelInfoResponse(BaseModel):
    model_name: str = Field(alias="modelName")
    model_purpose: str = Field(alias="modelPurpose")
    training_start_date: date = Field(alias="trainingStartDate")
    training_end_date: date = Field(alias="trainingEndDate")
    synthetic_data: bool = Field(alias="syntheticData")
    maximum_forecast_days: int = Field(alias="maximumForecastDays")
    feature_count: int = Field(alias="featureCount")
