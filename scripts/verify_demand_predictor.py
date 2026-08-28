#!/usr/bin/env python3
"""Run read-only functional checks for recursive J+1/J+7 demand inference."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.extractor import extract_demand_dataset  # noqa: E402
from app.features.demand_features import create_future_feature_row, load_promotion_activity  # noqa: E402
from app.prediction.demand_predictor import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    HISTORY_LOOKBACK_DAYS,
    predict_demand,
)


PROFILE_PRODUCTS = (52, 46, 2, 51)


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("La date doit respecter le format YYYY-MM-DD.") from error


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of-date", type=parse_date, default=date(2026, 8, 27))
    parser.add_argument("--synthetic-batch", required=True, help="Batch à utiliser pour ce contrôle fonctionnel.")
    return parser.parse_args()


def assert_forecast_contract(forecast: dict[str, object], as_of_date: date, expected_days: int) -> None:
    predictions = forecast["predictions"]
    assert len(predictions) == expected_days
    dates = [date.fromisoformat(item["date"]) for item in predictions]
    assert all(prediction_date > as_of_date for prediction_date in dates)
    assert dates == [as_of_date + timedelta(days=offset) for offset in range(1, expected_days + 1)]
    assert all(float(item["predictedDemand"]) >= 0 for item in predictions)


def assert_recursive_float_history(product_id: int, as_of_date: date, synthetic_batch: str, database_url: str) -> None:
    history_start = as_of_date - timedelta(days=HISTORY_LOOKBACK_DAYS - 1)
    raw = extract_demand_dataset(history_start, as_of_date, synthetic_batch, database_url)
    history = raw[raw["productId"] == product_id].copy().reset_index(drop=True)
    first_future = create_future_feature_row(history, as_of_date + timedelta(days=1), promotion_active=0)
    first_date = as_of_date + timedelta(days=1)
    assert first_future["dayOfWeek"] == first_date.weekday()
    assert first_future["dayOfMonth"] == first_date.day
    assert first_future["month"] == first_date.month
    assert first_future["weekOfYear"] == first_date.isocalendar().week
    assert first_future["isWeekend"] == int(first_date.weekday() >= 5)
    fractional_prediction = 7.25
    history.loc[len(history)] = {
        "date": as_of_date + timedelta(days=1),
        "productId": product_id,
        "productReference": history["productReference"].iloc[0],
        "demandQty": fractional_prediction,
    }
    second_future = create_future_feature_row(history, as_of_date + timedelta(days=2), promotion_active=0)
    assert second_future["lag1"] == fractional_prediction
    assert second_future["rollingMean7"] != first_future["rollingMean7"]


def main() -> int:
    load_dotenv(ROOT / "backend" / ".env", override=False)
    load_dotenv(ROOT / ".env", override=False)
    arguments = parse_arguments()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("Erreur : DATABASE_URL est obligatoire.", file=sys.stderr)
        return 1
    model_mtime = DEFAULT_MODEL_PATH.stat().st_mtime
    promotion_activity = load_promotion_activity(
        arguments.as_of_date + timedelta(days=1),
        arguments.as_of_date + timedelta(days=7),
        database_url,
    )
    active_promotions = {(int(row.productId), row.date.isoformat()) for row in promotion_activity.itertuples(index=False)}
    for product_id in PROFILE_PRODUCTS:
        forecast_j1 = predict_demand(product_id, 1, arguments.as_of_date, arguments.synthetic_batch, database_url)
        forecast_j7 = predict_demand(product_id, 7, arguments.as_of_date, arguments.synthetic_batch, database_url)
        assert_forecast_contract(forecast_j1, arguments.as_of_date, 1)
        assert_forecast_contract(forecast_j7, arguments.as_of_date, 7)
        assert forecast_j1["predictions"][0]["predictedDemand"] == forecast_j7["predictions"][0]["predictedDemand"]
        for item in forecast_j7["predictions"]:
            assert item["promotionActive"] == int((product_id, item["date"]) in active_promotions)
        print(f"Produit {product_id}: {[round(item['predictedDemand'], 2) for item in forecast_j7['predictions']]}")
    assert_recursive_float_history(PROFILE_PRODUCTS[0], arguments.as_of_date, arguments.synthetic_batch, database_url)
    assert DEFAULT_MODEL_PATH.stat().st_mtime == model_mtime
    print("Verification OK: J+1/J+7, dates, non-negativite, recurrence flottante et absence de fit/ecriture locale.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
