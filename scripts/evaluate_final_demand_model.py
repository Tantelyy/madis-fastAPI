#!/usr/bin/env python3
"""Fit the frozen Extra Trees candidate on TRAIN+VALIDATION and evaluate TEST once."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
import joblib
import numpy as np
import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.extractor import extract_demand_dataset  # noqa: E402
from app.evaluation.baselines import calculate_metrics, chronological_split  # noqa: E402
from app.evaluation.model_evaluation import (  # noqa: E402
    BASELINE_NAME,
    analyze_prediction_by_demand_level,
    evaluate_frozen_hybrid_on_test,
    evaluate_test_per_product,
    export_final_mae_comparison_chart,
    export_test_prediction_example,
)
from app.features.demand_features import (  # noqa: E402
    FeatureValidationError,
    create_demand_features,
    load_promotion_activity,
)
from app.training.demand_training import (  # noqa: E402
    EXTRA_TREES_SMOOTH_PARAMETERS,
    TrainingValidationError,
    build_extra_trees_smooth_pipeline,
)


SELECTED_MODEL_NAME = "EXTRA_TREES_SMOOTH"


def format_dataset_date(value: object) -> str:
    """Serialize a pandas date/timestamp as the dataset's date-only ISO format."""
    return pd.Timestamp(value).date().isoformat()


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
    parser.add_argument(
        "--frozen-hybrid-selection",
        type=Path,
        default=ROOT / "reports" / "models" / "hybrid_per_product.csv",
        help="Mapping hybride cree sur VALIDATION, applique ici sans modification.",
    )
    return parser.parse_args()


def load_frozen_validation_metrics() -> tuple[dict[str, float], dict[str, float]]:
    """Read the already-selected validation scores; never recompute or select here."""
    report_path = ROOT / "reports" / "models" / "follow_up_final_comparison.csv"
    if not report_path.exists():
        raise FileNotFoundError(f"Rapport de validation fige introuvable : {report_path}")
    report = pd.read_csv(report_path)
    baseline = report.loc[report["method"] == BASELINE_NAME]
    model = report.loc[report["method"] == SELECTED_MODEL_NAME]
    if len(baseline) != 1 or len(model) != 1:
        raise ValueError("Le rapport de validation ne contient pas exactement la baseline et le modele selectionne.")
    return (
        {"MAE": float(baseline.iloc[0]["MAE"]), "RMSE": float(baseline.iloc[0]["RMSE"]), "WAPE": float(baseline.iloc[0]["WAPE"])},
        {"MAE": float(model.iloc[0]["MAE"]), "RMSE": float(model.iloc[0]["RMSE"]), "WAPE": float(model.iloc[0]["WAPE"])},
    )


def create_test_metrics(
    baseline_metrics: dict[str, float],
    model_metrics: dict[str, float],
    hybrid_metrics: dict[str, float] | None,
) -> pd.DataFrame:
    rows = [
        {"method": BASELINE_NAME, **baseline_metrics, "improvementVsBaseline": 0.0},
        {
            "method": SELECTED_MODEL_NAME,
            **model_metrics,
            "improvementVsBaseline": (baseline_metrics["MAE"] - model_metrics["MAE"])
            / baseline_metrics["MAE"]
            * 100,
        },
    ]
    if hybrid_metrics is not None:
        rows.append(
            {
                "method": "HYBRID_PER_PRODUCT_FROZEN",
                **hybrid_metrics,
                "improvementVsBaseline": (baseline_metrics["MAE"] - hybrid_metrics["MAE"])
                / baseline_metrics["MAE"]
                * 100,
            },
        )
    return pd.DataFrame(rows).sort_values("MAE", kind="stable", ignore_index=True)


def save_reports(
    test_metrics: pd.DataFrame,
    product_metrics: pd.DataFrame,
    demand_level_metrics: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    reports = ROOT / "reports" / "models"
    reports.mkdir(parents=True, exist_ok=True)
    test_metrics.to_csv(reports / "final_test_metrics.csv", index=False, encoding="utf-8")
    product_metrics.to_csv(reports / "final_test_product_metrics.csv", index=False, encoding="utf-8")
    demand_level_metrics.to_csv(reports / "final_test_demand_level_metrics.csv", index=False, encoding="utf-8")
    (reports / "final_test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_final_model(
    pipeline,
    feature_columns: tuple[str, ...],
    final_train: pd.DataFrame,
    test: pd.DataFrame,
    validation_baseline: dict[str, float],
    validation_model: dict[str, float],
    test_baseline: dict[str, float],
    test_model: dict[str, float],
    synthetic_batch: str | None,
) -> None:
    artifacts = ROOT / "artifacts" / "models"
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifacts / "demand_model.joblib")
    metadata = {
        "modelName": SELECTED_MODEL_NAME,
        "trainedAt": datetime.now(timezone.utc).isoformat(),
        "trainingStartDate": format_dataset_date(final_train["date"].min()),
        "trainingEndDate": format_dataset_date(final_train["date"].max()),
        "testStartDate": format_dataset_date(test["date"].min()),
        "testEndDate": format_dataset_date(test["date"].max()),
        "features": list(feature_columns),
        "hyperparameters": EXTRA_TREES_SMOOTH_PARAMETERS,
        "validationMAE": validation_model["MAE"],
        "testMAE": test_model["MAE"],
        "baselineValidationMAE": validation_baseline["MAE"],
        "baselineTestMAE": test_baseline["MAE"],
        "syntheticData": bool(synthetic_batch),
        "syntheticBatch": synthetic_batch,
        "testWasUsedForTuning": False,
        "selectionWasFrozenBeforeTest": True,
    }
    (artifacts / "demand_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def print_results(test_metrics: pd.DataFrame, product_metrics: pd.DataFrame, demand_levels: pd.DataFrame) -> None:
    print("TEST final - baseline et modele selectionne")
    print(test_metrics.to_string(index=False, formatters={
        "MAE": "{:.4f}".format,
        "RMSE": "{:.4f}".format,
        "WAPE": "{:.2f}".format,
        "improvementVsBaseline": "{:+.2f}%".format,
    }))
    print("\nTEST par produit")
    print(product_metrics.to_string(index=False, formatters={
        "meanActualDemand": "{:.4f}".format,
        "baselineMAE": "{:.4f}".format,
        "mlMAE": "{:.4f}".format,
        "baselineRMSE": "{:.4f}".format,
        "mlRMSE": "{:.4f}".format,
        "maeImprovementVsBaseline": "{:+.2f}%".format,
    }))
    print("\nTEST par niveau de demande")
    print(demand_levels.to_string(index=False, formatters={"MAE": "{:.4f}".format, "RMSE": "{:.4f}".format, "WAPE": "{:.2f}".format}))


def main() -> int:
    load_dotenv(ROOT / "backend" / ".env", override=False)
    load_dotenv(ROOT / ".env", override=False)
    arguments = parse_arguments()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("Erreur : DATABASE_URL est obligatoire.", file=sys.stderr)
        return 1
    try:
        validation_baseline, validation_model = load_frozen_validation_metrics()
        raw_dataframe = extract_demand_dataset(
            arguments.start_date,
            arguments.end_date,
            synthetic_batch=arguments.synthetic_batch,
            database_url=database_url,
        )
        promotion_activity = load_promotion_activity(arguments.start_date, arguments.end_date, database_url)
        features = create_demand_features(raw_dataframe, promotion_activity)
        split = chronological_split(features)
        final_train = pd.concat([split.train, split.validation], ignore_index=True).sort_values(
            ["productId", "date"], kind="stable", ignore_index=True,
        )
        pipeline, feature_columns = build_extra_trees_smooth_pipeline(
            final_train,
            include_promotion_active="promotionActive" in features.columns,
        )
        # Fit deliberately receives only TRAIN + VALIDATION. TEST is untouched until predict().
        pipeline.fit(final_train.loc[:, feature_columns], final_train["demandQty"])
        raw_prediction = np.asarray(pipeline.predict(split.test.loc[:, feature_columns]), dtype=float)
        prediction = pd.Series(np.maximum(raw_prediction, 0.0), index=split.test.index, name=SELECTED_MODEL_NAME)
        test_baseline = calculate_metrics(split.test["demandQty"], split.test["rollingMean7"])
        test_model = calculate_metrics(split.test["demandQty"], prediction)
        product_metrics = evaluate_test_per_product(split.test, prediction)
        demand_levels = pd.concat(
            [
                analyze_prediction_by_demand_level(split.test, split.test["rollingMean7"], BASELINE_NAME),
                analyze_prediction_by_demand_level(split.test, prediction, SELECTED_MODEL_NAME),
            ],
            ignore_index=True,
        )
        frozen_selection_path = arguments.frozen_hybrid_selection
        if not frozen_selection_path.is_absolute():
            frozen_selection_path = ROOT / frozen_selection_path
        frozen_selection = pd.read_csv(frozen_selection_path)
        _, hybrid_metrics = evaluate_frozen_hybrid_on_test(split.test, prediction, frozen_selection)
        test_metrics = create_test_metrics(test_baseline, test_model, hybrid_metrics)
        summary = {
            "selectedModel": SELECTED_MODEL_NAME,
            "selectionWasFrozenBeforeTest": True,
            "finalTrainingPeriod": {
                "start": format_dataset_date(final_train["date"].min()),
                "end": format_dataset_date(final_train["date"].max()),
                "rows": len(final_train),
            },
            "testPeriod": {
                "start": format_dataset_date(split.test["date"].min()),
                "end": format_dataset_date(split.test["date"].max()),
                "rows": len(split.test),
            },
            "validation": {"baseline": validation_baseline, "model": validation_model},
            "test": test_metrics.to_dict(orient="records"),
            "negativePredictionsClipped": int((raw_prediction < 0).sum()),
            "hybridSelectionSource": str(frozen_selection_path),
        }
        print_results(test_metrics, product_metrics, demand_levels)
        save_reports(test_metrics, product_metrics, demand_levels, summary)
        save_final_model(
            pipeline,
            feature_columns,
            final_train,
            split.test,
            validation_baseline,
            validation_model,
            test_baseline,
            test_model,
            arguments.synthetic_batch,
        )
        reports = ROOT / "reports" / "models"
        export_final_mae_comparison_chart(
            {"baseline": validation_baseline["MAE"], "model": validation_model["MAE"]},
            {"baseline": test_baseline["MAE"], "model": test_model["MAE"]},
            reports / "final_validation_test_mae.png",
        )
        export_test_prediction_example(split.test, prediction, reports / "final_test_prediction_typical.png")
        high_rotation_product = int(split.test.groupby("productId")["demandQty"].mean().idxmax())
        export_test_prediction_example(
            split.test,
            prediction,
            reports / "final_test_prediction_high_rotation.png",
            product_id=high_rotation_product,
        )
        print("\nTEST utilise une seule fois pour evaluation finale, sans tuning ni nouvelle selection.")
        return 0
    except (TrainingValidationError, FeatureValidationError, ValueError, OSError, FileNotFoundError, psycopg.Error) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
