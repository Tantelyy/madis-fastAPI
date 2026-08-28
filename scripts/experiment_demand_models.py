#!/usr/bin/env python3
"""Run small follow-up demand experiments on TRAIN and VALIDATION only."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.extractor import extract_demand_dataset  # noqa: E402
from app.evaluation.baselines import calculate_metrics, chronological_split  # noqa: E402
from app.evaluation.model_evaluation import (  # noqa: E402
    BASELINE_NAME,
    analyze_prediction_by_demand_level,
    analyze_zero_demand_by_product,
    build_hybrid_per_product,
    evaluate_candidate_models,
)
from app.features.demand_features import (  # noqa: E402
    FeatureValidationError,
    create_demand_features,
    load_promotion_activity,
)
from app.training.demand_training import (  # noqa: E402
    TrainingValidationError,
    train_and_evaluate_candidates,
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


def create_final_comparison(comparison: pd.DataFrame, hybrid_metrics: dict[str, float]) -> pd.DataFrame:
    """Combine baseline, candidates and exploratory hybrid under one common schema."""
    baseline_mae = float(comparison["baselineMAE"].iloc[0])
    baseline_rmse = calculate_metrics_placeholder(comparison, "RMSE")
    baseline_wape = calculate_metrics_placeholder(comparison, "WAPE")
    rows = [
        {
            "method": BASELINE_NAME,
            "MAE": baseline_mae,
            "RMSE": baseline_rmse,
            "WAPE": baseline_wape,
            "improvementVsRollingMean7": 0.0,
        },
    ]
    for row in comparison.itertuples(index=False):
        rows.append(
            {
                "method": row.model,
                "MAE": row.MAE,
                "RMSE": row.RMSE,
                "WAPE": row.WAPE,
                "improvementVsRollingMean7": row.improvementVsBaseline,
            },
        )
    rows.append(
        {
            "method": "HYBRID_PER_PRODUCT",
            "MAE": hybrid_metrics["MAE"],
            "RMSE": hybrid_metrics["RMSE"],
            "WAPE": hybrid_metrics["WAPE"],
            "improvementVsRollingMean7": (baseline_mae - hybrid_metrics["MAE"]) / baseline_mae * 100,
        },
    )
    return pd.DataFrame(rows).sort_values("MAE", kind="stable", ignore_index=True)


def calculate_metrics_placeholder(comparison: pd.DataFrame, metric_name: str) -> float:
    """Recover a baseline metric supplied separately by the caller via DataFrame attrs."""
    return float(comparison.attrs["baselineMetrics"][metric_name])


def print_results(
    error_by_level: pd.DataFrame,
    zero_by_product: pd.DataFrame,
    final_comparison: pd.DataFrame,
    hybrid_selection: pd.DataFrame,
    best_ml_name: str,
) -> None:
    print("Erreurs par niveau de demande reelle - VALIDATION uniquement")
    print(error_by_level.to_string(index=False, formatters={"MAE": "{:.4f}".format, "RMSE": "{:.4f}".format, "WAPE": "{:.2f}".format}))
    print("\nZeros par produit : baseline vs Random Forest original")
    print(zero_by_product.to_string(index=False, formatters={
        "zeroDemandPercentage": "{:.2f}".format,
        "meanActualDemand": "{:.4f}".format,
        "baselineMAE": "{:.4f}".format,
        "randomForestMAE": "{:.4f}".format,
    }))
    print("\nComparaison finale - VALIDATION uniquement")
    print(final_comparison.to_string(index=False, formatters={
        "MAE": "{:.4f}".format,
        "RMSE": "{:.4f}".format,
        "WAPE": "{:.2f}".format,
        "improvementVsRollingMean7": "{:+.2f}%".format,
    }))
    print(f"\nMeilleur candidat ML global : {best_ml_name}")
    print("\nChoix hybride exploratoire par produit")
    print(hybrid_selection.to_string(index=False, formatters={"baselineMAE": "{:.4f}".format, "mlMAE": "{:.4f}".format}))


def save_reports(
    comparison: pd.DataFrame,
    product_metrics: pd.DataFrame,
    error_by_level: pd.DataFrame,
    zero_by_product: pd.DataFrame,
    hybrid_selection: pd.DataFrame,
    final_comparison: pd.DataFrame,
    candidates,
) -> None:
    output_directory = ROOT / "reports" / "models"
    output_directory.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_directory / "follow_up_candidate_metrics.csv", index=False, encoding="utf-8")
    product_metrics.to_csv(output_directory / "follow_up_product_metrics.csv", index=False, encoding="utf-8")
    error_by_level.to_csv(output_directory / "validation_error_by_demand_level.csv", index=False, encoding="utf-8")
    zero_by_product.to_csv(output_directory / "zero_demand_by_product.csv", index=False, encoding="utf-8")
    hybrid_selection.to_csv(output_directory / "hybrid_per_product.csv", index=False, encoding="utf-8")
    final_comparison.to_csv(output_directory / "follow_up_final_comparison.csv", index=False, encoding="utf-8")
    (output_directory / "follow_up_model_parameters.json").write_text(
        json.dumps({candidate.name: candidate.parameters for candidate in candidates}, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    load_dotenv(ROOT / "backend" / ".env", override=False)
    load_dotenv(ROOT / ".env", override=False)
    arguments = parse_arguments()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("Erreur : DATABASE_URL est obligatoire.", file=sys.stderr)
        return 1
    try:
        raw_dataframe = extract_demand_dataset(
            arguments.start_date,
            arguments.end_date,
            synthetic_batch=arguments.synthetic_batch,
            database_url=database_url,
        )
        promotion_activity = load_promotion_activity(arguments.start_date, arguments.end_date, database_url)
        features = create_demand_features(raw_dataframe, promotion_activity)
        split = chronological_split(features)
        # Intentionally do not inspect, predict or score split.test in this experiment.
        candidates = train_and_evaluate_candidates(
            split.train,
            split.validation,
            include_promotion_active="promotionActive" in features.columns,
            include_follow_up_variants=True,
            include_two_stage=True,
        )
        comparison, product_metrics = evaluate_candidate_models(split.validation, candidates)
        baseline_metrics = calculate_metrics(split.validation["demandQty"], split.validation["rollingMean7"])
        comparison.attrs["baselineMetrics"] = baseline_metrics
        original_forest = next(candidate for candidate in candidates if candidate.name == "RANDOM_FOREST")
        error_by_level = pd.concat(
            [
                analyze_prediction_by_demand_level(
                    split.validation,
                    split.validation["rollingMean7"],
                    BASELINE_NAME,
                ),
                analyze_prediction_by_demand_level(
                    split.validation,
                    original_forest.validation_prediction,
                    "RANDOM_FOREST",
                ),
            ],
            ignore_index=True,
        )
        zero_by_product = analyze_zero_demand_by_product(split.validation, original_forest.validation_prediction)
        best_ml_name = str(comparison.iloc[0]["model"])
        best_ml = next(candidate for candidate in candidates if candidate.name == best_ml_name)
        hybrid_selection, _, hybrid_metrics = build_hybrid_per_product(split.validation, best_ml)
        final_comparison = create_final_comparison(comparison, hybrid_metrics)
        print_results(error_by_level, zero_by_product, final_comparison, hybrid_selection, best_ml_name)
        save_reports(
            comparison,
            product_metrics,
            error_by_level,
            zero_by_product,
            hybrid_selection,
            final_comparison,
            candidates,
        )
        print("\nTEST : aucune prediction, aucune metrique et aucune selection n'ont ete effectuees.")
        return 0
    except (TrainingValidationError, FeatureValidationError, ValueError, OSError, psycopg.Error) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
