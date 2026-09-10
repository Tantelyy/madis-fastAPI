from __future__ import annotations

from datetime import date, datetime
from unittest.mock import patch
import unittest

from app.prediction.demand_predictor import (
    InsufficientHistoryError,
    PredictionValidationError,
    ProductNotFoundError,
)
from app.prediction.stockout_predictor import (
    InventoryLotSnapshot,
    ProductStockSnapshot,
    calculate_lot_stock_projection,
    calculate_stock_projection,
    predict_stockout,
)


START_DATE = date(2030, 1, 1)


def predictions(*demands: float) -> list[dict[str, object]]:
    return [
        {
            "date": date.fromordinal(START_DATE.toordinal() + offset).isoformat(),
            "predictedDemand": demand,
            # This intentionally differs for the float test below.
            "predictedQuantity": int(round(demand)),
        }
        for offset, demand in enumerate(demands)
    ]


class StockoutProjectionTestCase(unittest.TestCase):
    def test_projection_has_no_stockout_when_supply_covers_forecast(self) -> None:
        result = calculate_stock_projection(100.0, predictions(5, 5, 5, 5, 5, 5, 10))

        self.assertFalse(result["stockoutExpected"])
        self.assertEqual(result["remainingStockAfterHorizon"], 60.0)
        self.assertEqual(result["totalShortageAfterHorizon"], 0.0)
        self.assertTrue(all(day["plannedIncomingQuantity"] == 0.0 for day in result["dailyProjection"]))


    def test_projection_detects_exact_fourth_day_stockout(self) -> None:
        result = calculate_stock_projection(30.0, predictions(5, 5, 5, 20, 4, 4, 4))

        self.assertTrue(result["stockoutExpected"])
        self.assertEqual(result["predictedStockoutDate"], "2030-01-04")
        self.assertEqual(result["daysUntilStockout"], 4)
        self.assertEqual(result["dailyProjection"][3]["projectedEndingStockRaw"], -5.0)
        self.assertEqual(result["dailyProjection"][3]["projectedEndingStock"], 0.0)
        self.assertEqual(result["totalShortageAfterHorizon"], 17.0)


    def test_projection_detects_first_day_shortage(self) -> None:
        result = calculate_stock_projection(5.0, predictions(8.0))

        self.assertEqual(result["predictedStockoutDate"], "2030-01-01")
        self.assertEqual(result["daysUntilStockout"], 1)
        self.assertEqual(result["dailyProjection"][0]["shortageQuantity"], 3.0)


    def test_projection_uses_unrounded_float_predictions(self) -> None:
        result = calculate_stock_projection(10.0, predictions(3.25, 2.5))

        self.assertEqual(result["totalPredictedDemand"], 5.75)
        self.assertEqual(result["dailyProjection"][0]["projectedEndingStock"], 6.75)
        self.assertEqual(result["dailyProjection"][1]["projectedEndingStock"], 4.25)

    def test_future_expiry_removes_unconsumed_lot_stock(self) -> None:
        """A lot expiring at 2026-08-29 midnight cannot cover 29 August."""
        lots = (
            InventoryLotSnapshot(
                inventory_id=1,
                available_quantity=20.0,
                created_at=datetime(2026, 8, 1),
                expired_at=datetime(2026, 8, 29),
            ),
            InventoryLotSnapshot(
                inventory_id=2,
                available_quantity=20.0,
                created_at=datetime(2026, 8, 2),
                expired_at=datetime(2027, 1, 1),
            ),
        )
        result = calculate_lot_stock_projection(
            lots,
            [
                {"date": "2026-08-28", "predictedDemand": 5.0},
                {"date": "2026-08-29", "predictedDemand": 5.0},
            ],
        )

        first_day, second_day = result["dailyProjection"]
        self.assertEqual(first_day["openingStock"], 40.0)
        self.assertEqual(first_day["projectedEndingStock"], 35.0)
        self.assertEqual(second_day["expiredQuantity"], 15.0)
        self.assertEqual(second_day["openingStock"], 20.0)
        self.assertEqual(second_day["projectedEndingStock"], 15.0)
        self.assertEqual(result["remainingStockAfterHorizon"], 15.0)


    def test_predict_stockout_supports_allowed_horizons(self) -> None:
        for horizon in (1, 7):
            with self.subTest(horizon=horizon):
                demand = {
                    "productId": 52,
                    "productReference": "REF-52",
                    "asOfDate": "2026-08-27",
                    "forecastDays": horizon,
                    "predictions": predictions(*([1.0] * horizon)),
                }
                snapshot = ProductStockSnapshot(52, "REF-52", date(2026, 8, 27), 10.0, ())
                with patch("app.prediction.stockout_predictor.predict_demand", return_value=demand), patch(
                    "app.prediction.stockout_predictor.get_available_stock", return_value=snapshot
                ):
                    result = predict_stockout(52, forecast_days=horizon, database_url="postgresql://read-only")

                self.assertEqual(len(result["dailyProjection"]), horizon)

    def test_predict_stockout_rejects_invalid_horizons_before_any_database_read(self) -> None:
        for horizon in (0, 8):
            with self.subTest(horizon=horizon), patch(
                "app.prediction.stockout_predictor.predict_demand"
            ) as demand_forecast:
                with self.assertRaisesRegex(PredictionValidationError, "forecast_days"):
                    predict_stockout(52, forecast_days=horizon, database_url="postgresql://read-only")
                demand_forecast.assert_not_called()


    def test_zero_current_stock_uses_as_of_date_and_day_zero(self) -> None:
        demand = {
            "productId": 52,
            "productReference": "REF-52",
            "asOfDate": "2026-08-27",
            "forecastDays": 1,
            "predictions": predictions(2.0),
        }
        snapshot = ProductStockSnapshot(52, "REF-52", date(2026, 8, 27), 0.0, ())
        with patch("app.prediction.stockout_predictor.predict_demand", return_value=demand), patch(
            "app.prediction.stockout_predictor.get_available_stock", return_value=snapshot
        ):
            result = predict_stockout(52, forecast_days=1, database_url="postgresql://read-only")

        self.assertTrue(result["alreadyOutOfStock"])
        self.assertTrue(result["stockoutExpected"])
        self.assertEqual(result["predictedStockoutDate"], "2026-08-27")
        self.assertEqual(result["daysUntilStockout"], 0)


    def test_predict_stockout_propagates_demand_business_errors(self) -> None:
        for error in (ProductNotFoundError("missing"), InsufficientHistoryError("too short")):
            with self.subTest(error=type(error).__name__), patch(
                "app.prediction.stockout_predictor.predict_demand", side_effect=error
            ):
                with self.assertRaises(type(error)):
                    predict_stockout(52, forecast_days=1, database_url="postgresql://read-only")


    def test_projection_is_in_memory_and_rejects_negative_demand(self) -> None:
        """The calculation has no database dependency and never writes anything."""
        with self.assertRaisesRegex(ValueError, "predictedDemand"):
            calculate_stock_projection(10.0, predictions(-0.1))


if __name__ == "__main__":
    unittest.main()
