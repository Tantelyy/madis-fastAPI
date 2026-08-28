"""Public Pydantic schemas for read-only stockout forecasting endpoints."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class StockoutDailyProjection(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    date: date
    opening_stock: float = Field(alias="openingStock")
    predicted_demand: float = Field(alias="predictedDemand")
    planned_incoming_quantity: float = Field(alias="plannedIncomingQuantity")
    expired_quantity: float = Field(alias="expiredQuantity")
    projected_ending_stock_raw: float = Field(alias="projectedEndingStockRaw")
    projected_ending_stock: float = Field(alias="projectedEndingStock")
    shortage_quantity: float = Field(alias="shortageQuantity")
    stockout: bool


class StockoutForecastResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    product_id: int = Field(alias="productId")
    product_reference: str = Field(alias="productReference")
    as_of_date: date = Field(alias="asOfDate")
    forecast_days: int = Field(alias="forecastDays")
    current_stock: float = Field(alias="currentStock")
    total_predicted_demand: float = Field(alias="totalPredictedDemand")
    already_out_of_stock: bool = Field(alias="alreadyOutOfStock")
    stockout_expected: bool = Field(alias="stockoutExpected")
    predicted_stockout_date: date | None = Field(alias="predictedStockoutDate")
    days_until_stockout: int | None = Field(alias="daysUntilStockout")
    remaining_stock_after_horizon: float = Field(alias="remainingStockAfterHorizon")
    total_shortage_after_horizon: float = Field(alias="totalShortageAfterHorizon")
    daily_projection: list[StockoutDailyProjection] = Field(alias="dailyProjection")


class StockoutForecastBatchResponse(BaseModel):
    forecasts: list[StockoutForecastResponse]
