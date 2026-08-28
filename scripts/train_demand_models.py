#!/usr/bin/env python3
"""Train three demand-model candidates on TRAIN and compare them on VALIDATION only."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
import joblib
import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.extractor import extract_demand_dataset  # noqa: E402
from app.evaluation.baselines import chronological_split, summarize_partition  # noqa: E402
from app.evaluation.model_evaluation import (  # noqa: E402
    analyze_errors_by_demand_level,
    evaluate_candidate_models,
    export_model_comparison_chart,
    export_validation_prediction_example,
)
from app.features.demand_features import (  # noqa: E402
    FeatureValidationError,
    create_demand_features,
    load_promotion_activity,
)
from app.training.demand_training import (  # noqa: E402
    RANDOM_STATE,
    TrainingValidationError,
    extract_feature_importances,
    train_and_evaluate_candidates,
)


PROMOTION_ACTIVITY_DECISION = (
    "conservée : elle est reconstruite depuis les dates planifiées de SpecialOffers, "
    "bornées par createdAt, jamais depuis SALE, PROMOTION_GIFT ou demandQty."
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
    parser.add_argument("--skip-plots", action="store_true", help="N'exporte pas les deux graphiques de validation.")
    return parser.parse_args()


def print_split_summary(split) -> None:
    print("Résumé du split (TEST isolé)")
    for name, partition in (("TRAIN", split.train), ("VALIDATION", split.validation), ("TEST", split.test)):
        summary = summarize_partition(name, partition)
        print(
            f"  {name}: {summary.minimum_date} -> {summary.maximum_date}; "
            f"dates={summary.date_count}; lignes={summary.row_count}; produits={summary.product_count}",
        )


def print_results(comparison: pd.DataFrame, product_metrics: pd.DataFrame, error_analysis: pd.DataFrame, importances: pd.DataFrame) -> None:
    print("\nCandidats ML - VALIDATION uniquement")
    print(comparison.to_string(index=False, formatters={
        "MAE": "{:.4f}".format,
        "RMSE": "{:.4f}".format,
        "WAPE": "{:.2f}".format,
        "baselineMAE": "{:.4f}".format,
        "improvementVsBaseline": "{:+.2f}%".format,
    }))
    print("\nMétriques par produit")
    print(product_metrics.to_string(index=False, formatters={
        "meanActualDemand": "{:.4f}".format,
        "MAE": "{:.4f}".format,
        "RMSE": "{:.4f}".format,
        "WAPE": "{:.2f}".format,
        "baselineMAE": "{:.4f}".format,
        "maeImprovementVsBaseline": "{:+.2f}%".format,
    }))
    print("\nErreurs du meilleur candidat par niveau de demande réelle")
    print(error_analysis.to_string(index=False, formatters={"MAE": "{:.4f}".format, "RMSE": "{:.4f}".format, "WAPE": "{:.2f}".format}))
    if importances.empty:
        print("\nAucune importance native disponible pour le meilleur candidat.")
    else:
        print("\nPrincipales importances de features (non causales)")
        print(importances.head(12).to_string(index=False, formatters={"importance": "{:.4f}".format}))


def save_outputs(best, comparison: pd.DataFrame, product_metrics: pd.DataFrame, error_analysis: pd.DataFrame, importances: pd.DataFrame) -> None:
    reports = ROOT / "reports" / "models"
    reports.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(reports / "model_comparison.csv", index=False, encoding="utf-8")
    product_metrics.to_csv(reports / "product_metrics.csv", index=False, encoding="utf-8")
    error_analysis.to_csv(reports / "error_by_demand_level.csv", index=False, encoding="utf-8")
    importances.to_csv(reports / "feature_importances.csv", index=False, encoding="utf-8")

    artifact_directory = ROOT / "artifacts" / "candidates"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.pipeline, artifact_directory / "demand_model_candidate.joblib")
    metadata = {
        "selectionSet": "VALIDATION only",
        "model": best.name,
        "randomState": RANDOM_STATE,
        "featureColumns": list(best.feature_columns),
        "parameters": best.parameters,
    }
    (artifact_directory / "demand_model_candidate_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
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
        include_promotion_active = "promotionActive" in features.columns
        print("Colonnes exactes du dataset final :")
        print("  " + ", ".join(features.columns))
        print(f"promotionActive : {PROMOTION_ACTIVITY_DECISION}")
        split = chronological_split(features)
        print_split_summary(split)
        candidates = train_and_evaluate_candidates(
            split.train,
            split.validation,
            include_promotion_active=include_promotion_active,
        )
        comparison, product_metrics = evaluate_candidate_models(split.validation, candidates)
        best_name = comparison.iloc[0]["model"]
        best = next(candidate for candidate in candidates if candidate.name == best_name)
        error_analysis = analyze_errors_by_demand_level(split.validation, best)
        importances = extract_feature_importances(best)
        print_results(comparison, product_metrics, error_analysis, importances)
        print(
            f"\nMeilleur candidat VALIDATION : {best.name}; MAE={comparison.iloc[0]['MAE']:.4f}; "
            f"baseline={comparison.iloc[0]['baselineMAE']:.4f}; "
            f"amélioration={comparison.iloc[0]['improvementVsBaseline']:+.2f}%",
        )
        print("Le TEST a seulement été préparé dans le split ; aucune métrique TEST n'a été calculée.")
        save_outputs(best, comparison, product_metrics, error_analysis, importances)
        if not arguments.skip_plots:
            reports = ROOT / "reports" / "models"
            export_model_comparison_chart(comparison, reports / "validation_mae_comparison.png")
            export_validation_prediction_example(split.validation, best, reports / "validation_prediction_example.png")
        return 0
    except (TrainingValidationError, FeatureValidationError, ValueError, OSError, psycopg.Error) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
