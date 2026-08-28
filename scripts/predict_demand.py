#!/usr/bin/env python3
"""Print an in-memory recursive demand forecast for one existing trained product."""

from __future__ import annotations

import argparse
from datetime import date
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.prediction.demand_predictor import PredictionValidationError, predict_demand  # noqa: E402


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("La date doit respecter le format YYYY-MM-DD.") from error


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-id", type=int, required=True)
    parser.add_argument("--days", type=int, default=7, help="Horizon entre 1 et 7 jours.")
    parser.add_argument("--as-of-date", type=parse_date, help="Dernière date de demande réellement connue.")
    parser.add_argument("--synthetic-batch", help="Filtre optionnel exact sur Carts.reason.")
    return parser.parse_args()


def main() -> int:
    load_dotenv(ROOT / "backend" / ".env", override=False)
    load_dotenv(ROOT / ".env", override=False)
    arguments = parse_arguments()
    try:
        forecast = predict_demand(
            product_id=arguments.product_id,
            forecast_days=arguments.days,
            as_of_date=arguments.as_of_date,
            synthetic_batch=arguments.synthetic_batch,
        )
        print(json.dumps(forecast, indent=2, ensure_ascii=False))
        return 0
    except (PredictionValidationError, ValueError, OSError) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
