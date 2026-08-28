"""Read-only integration tests for the FastAPI demand endpoints."""

from __future__ import annotations

from datetime import date, timedelta
import os
from pathlib import Path
import unittest

from dotenv import load_dotenv
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / "backend" / ".env", override=False)
load_dotenv(ROOT / ".env", override=False)

from app.config import ApiSettings  # noqa: E402
from app.main import create_app  # noqa: E402


SYNTHETIC_BATCH = "[SYNTHETIC_DEMAND|v1|2025-08-28|2026-08-27|seed=42]"
MODEL_PATH = ROOT / "artifacts" / "models" / "demand_model_deployment.joblib"


class DemandApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise unittest.SkipTest("DATABASE_URL est requis pour les tests API d'intégration.")
        settings = ApiSettings(
            database_url=database_url,
            demand_data_mode="synthetic",
            synthetic_batch=SYNTHETIC_BATCH,
            model_path=MODEL_PATH,
            metadata_path=ROOT / "artifacts" / "models" / "demand_model_deployment_metadata.json",
            cors_origins=("http://localhost:3000",),
            log_level="INFO",
        )
        cls.app = create_app(settings)
        cls.client_context = TestClient(cls.app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_context.__exit__(None, None, None)

    def test_health(self) -> None:
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "demandModelLoaded": True})

    def test_forecast_one_and_seven_days_match_expected_contract(self) -> None:
        model_mtime = MODEL_PATH.stat().st_mtime
        first = self.client.get("/api/v1/demand/forecast/52", params={"days": 1, "as_of_date": "2026-08-27"})
        seven = self.client.get("/api/v1/demand/forecast/52", params={"days": 7, "as_of_date": "2026-08-27"})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(seven.status_code, 200)
        forecast = seven.json()
        self.assertEqual(forecast["productId"], 52)
        self.assertEqual(forecast["forecastDays"], 7)
        self.assertEqual(len(forecast["predictions"]), 7)
        self.assertAlmostEqual(forecast["predictions"][0]["predictedDemand"], 22.046272663729038)
        self.assertEqual(first.json()["predictions"][0], forecast["predictions"][0])
        dates = [date.fromisoformat(item["date"]) for item in forecast["predictions"]]
        self.assertEqual(dates, [date(2026, 8, 28) + timedelta(days=index) for index in range(7)])
        self.assertTrue(all(item["predictedDemand"] >= 0 for item in forecast["predictions"]))
        self.assertEqual(MODEL_PATH.stat().st_mtime, model_mtime)
        self.assertIsNotNone(self.app.state.demand_model_bundle)

    def test_invalid_horizon_and_date_are_rejected(self) -> None:
        self.assertEqual(self.client.get("/api/v1/demand/forecast/52", params={"days": 0}).status_code, 422)
        self.assertEqual(self.client.get("/api/v1/demand/forecast/52", params={"days": 8}).status_code, 422)
        self.assertEqual(
            self.client.get("/api/v1/demand/forecast/52", params={"as_of_date": "invalid-date"}).status_code,
            422,
        )

    def test_missing_product_returns_404(self) -> None:
        response = self.client.get("/api/v1/demand/forecast/999999", params={"days": 1})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Product 999999 not found")

    def test_batch_forecast(self) -> None:
        response = self.client.post(
            "/api/v1/demand/forecast",
            json={"productIds": [1, 2, 46, 52], "days": 7, "asOfDate": "2026-08-27"},
        )
        self.assertEqual(response.status_code, 200)
        forecasts = response.json()["forecasts"]
        self.assertEqual([item["productId"] for item in forecasts], [1, 2, 46, 52])
        self.assertTrue(all(len(item["predictions"]) == 7 for item in forecasts))


if __name__ == "__main__":
    unittest.main()
