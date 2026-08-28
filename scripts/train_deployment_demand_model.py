#!/usr/bin/env python3
"""Train the already-selected Extra Trees pipeline on all currently available history."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
import joblib
import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.extractor import extract_demand_dataset  # noqa: E402
from app.features.demand_features import create_demand_features, load_promotion_activity  # noqa: E402
from app.training.demand_training import (  # noqa: E402
    EXTRA_TREES_SMOOTH_PARAMETERS,
    TrainingValidationError,
    build_extra_trees_smooth_pipeline,
)


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("La date doit respecter le format YYYY-MM-DD.") from error


def parse_arguments() -> argparse.Namespace:
    default_end = date.today() - timedelta(days=1)
    default_start = default_end - timedelta(days=364)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=parse_date, default=default_start)
    parser.add_argument("--end-date", type=parse_date, default=default_end)
    parser.add_argument("--synthetic-batch", help="Filtre optionnel exact sur Carts.reason.")
    return parser.parse_args()


def main() -> int:
    load_dotenv(ROOT / "backend" / ".env", override=False)
    load_dotenv(ROOT / ".env", override=False)
    arguments = parse_arguments()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("Erreur : DATABASE_URL est obligatoire.", file=sys.stderr)
        return 1
    try:
        raw = extract_demand_dataset(
            arguments.start_date,
            arguments.end_date,
            synthetic_batch=arguments.synthetic_batch,
            database_url=database_url,
        )
        promotion_activity = load_promotion_activity(arguments.start_date, arguments.end_date, database_url)
        features = create_demand_features(raw, promotion_activity)
        pipeline, feature_columns = build_extra_trees_smooth_pipeline(
            features,
            include_promotion_active="promotionActive" in features.columns,
        )
        pipeline.fit(features.loc[:, feature_columns], features["demandQty"])
        artifacts = ROOT / "artifacts" / "models"
        artifacts.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, artifacts / "demand_model_deployment.joblib")
        metadata = {
            "modelName": "EXTRA_TREES_SMOOTH",
            "modelPurpose": "deployment",
            "trainedOnFullAvailableHistory": True,
            "trainedAt": datetime.now(timezone.utc).isoformat(),
            "sourceDataStartDate": arguments.start_date.isoformat(),
            "sourceDataEndDate": arguments.end_date.isoformat(),
            "trainingStartDate": features["date"].min().isoformat(),
            "trainingEndDate": features["date"].max().isoformat(),
            "trainingRowCount": len(features),
            "productCount": int(features["productId"].nunique()),
            "features": list(feature_columns),
            "hyperparameters": EXTRA_TREES_SMOOTH_PARAMETERS,
            "syntheticData": bool(arguments.synthetic_batch),
            "syntheticBatch": arguments.synthetic_batch,
            "tuningPerformed": False,
            "selectionWasFrozenBeforeDeploymentTraining": True,
        }
        (artifacts / "demand_model_deployment_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8",
        )
        print(
            f"Modele de deploiement entraine : {len(features)} lignes, "
            f"{features['date'].min()} -> {features['date'].max()}, "
            f"{features['productId'].nunique()} produits.",
        )
        return 0
    except (TrainingValidationError, ValueError, OSError, psycopg.Error) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
