#!/usr/bin/env python3
"""Prepare demand features, create a chronological split, and evaluate validation baselines."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.extractor import extract_demand_dataset  # noqa: E402
from app.evaluation.baselines import (  # noqa: E402
    EvaluationValidationError,
    chronological_split,
    evaluate_baselines,
    export_split_plot,
    summarize_partition,
)
from app.features.demand_features import (  # noqa: E402
    FeatureValidationError,
    create_demand_features,
    load_promotion_activity,
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
    parser.add_argument("--export-splits", action="store_true", help="Exporte les splits pour inspection seulement.")
    parser.add_argument("--split-directory", type=Path, default=ROOT / "datasets" / "splits")
    parser.add_argument("--skip-split-plot", action="store_true")
    return parser.parse_args()


def print_columns(dataframe: pd.DataFrame) -> None:
    print("Colonnes du DataFrame final :")
    print("  " + ", ".join(dataframe.columns))


def print_split_summary(split) -> None:
    print("\nRésumé du split")
    for name, partition in (("TRAIN", split.train), ("VALIDATION", split.validation), ("TEST", split.test)):
        summary = summarize_partition(name, partition)
        print(
            f"  {name}: dates={summary.minimum_date} au {summary.maximum_date}, "
            f"nombre de dates={summary.date_count}, lignes={summary.row_count}, produits={summary.product_count}",
        )
    print("  Vérifications : ordre strict, dates disjointes et conservation de toutes les lignes : OK")


def print_evaluation(global_results: pd.DataFrame, product_results: pd.DataFrame) -> None:
    print("\nBaselines — VALIDATION uniquement (MAE principale)")
    print(global_results.to_string(index=False, formatters={
        "MAE": "{:.4f}".format,
        "RMSE": "{:.4f}".format,
        "WAPE": "{:.2f}".format,
    }))
    print("\nRésultats par produit")
    print(product_results.to_string(index=False, formatters={
        "MAE": "{:.4f}".format,
        "RMSE": "{:.4f}".format,
        "meanActualDemand": "{:.4f}".format,
    }))
    print(f"\nRéférence de validation (MAE minimale) : {global_results.iloc[0]['baseline']}")
    print("Le TEST est préparé mais n'est pas utilisé pour choisir cette référence.")


def export_splits(split, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, dataframe in (("train", split.train), ("validation", split.validation), ("test", split.test)):
        output = directory / f"{name}.csv"
        dataframe.to_csv(output, index=False, encoding="utf-8")
        print(f"  Split exporté : {output}")


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
        promotion_activity = load_promotion_activity(
            arguments.start_date,
            arguments.end_date,
            database_url,
        )
        features = create_demand_features(raw_dataframe, promotion_activity)
        print_columns(features)
        split = chronological_split(features)
        print_split_summary(split)
        global_results, product_results = evaluate_baselines(split.validation)
        print_evaluation(global_results, product_results)
        if arguments.export_splits:
            export_directory = arguments.split_directory if arguments.split_directory.is_absolute() else ROOT / arguments.split_directory
            export_splits(split, export_directory)
        if not arguments.skip_split_plot:
            plot_path = ROOT / "reports" / "evaluation" / "chronological_split.png"
            export_split_plot(split, plot_path)
            print(f"  Graphique du split : {plot_path}")
        return 0
    except (EvaluationValidationError, FeatureValidationError, ValueError, OSError, psycopg.Error) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
