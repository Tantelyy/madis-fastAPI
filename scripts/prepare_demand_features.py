#!/usr/bin/env python3
"""Run demand EDA and create leakage-safe temporal features without training a model."""

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
from app.features.demand_features import (  # noqa: E402
    FeatureValidationError,
    analyze_demand_dataset,
    create_demand_features,
    export_eda_charts,
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
    parser.add_argument("--output", type=Path, default=ROOT / "datasets" / "demand_features.csv")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "eda")
    parser.add_argument("--skip-plots", action="store_true", help="N'exporte pas les graphiques EDA.")
    return parser.parse_args()


def print_eda(eda) -> None:
    values = eda.global_summary
    print("EDA globale")
    print(
        f"  lignes={values['row_count']}, produits={values['product_count']}, "
        f"période={values['minimum_date']} au {values['maximum_date']}",
    )
    print(
        f"  total={values['total_demand']}, moyenne={values['mean_demand']:.2f}, "
        f"médiane={values['median_demand']:.2f}, écart-type={values['standard_deviation']:.2f}, "
        f"min/max={values['minimum_demand']}/{values['maximum_demand']}, "
        f"zéros={values['zero_count']} ({values['zero_percentage']:.1f} %)",
    )
    print("  Moyenne par jour de semaine :")
    print(eda.weekday_summary.to_string(index=False, formatters={"meanDemand": "{:.2f}".format}))
    print("  Moyenne par mois :")
    print(eda.month_summary.to_string(index=False, formatters={"meanDemand": "{:.2f}".format}))
    print("  Statistiques par produit :")
    print(eda.product_summary.to_string(index=False, formatters={
        "meanDemand": "{:.2f}".format,
        "medianDemand": "{:.2f}".format,
        "standardDeviation": "{:.2f}".format,
        "zeroDayPercentage": "{:.1f}".format,
    }))


def print_feature_summary(features: pd.DataFrame) -> None:
    summary = features.attrs["feature_summary"]
    print("\nFeatures créées")
    print(f"  lignes brutes / finales / supprimées : {summary.raw_row_count} / {summary.feature_row_count} / {summary.removed_for_history_count}")
    print(f"  produits={summary.product_count}, période={summary.minimum_date} au {summary.maximum_date}")
    print(f"  colonnes={summary.column_count}, nulls={summary.null_count}, doublons={summary.duplicate_count}")
    sample_product = int(features["productId"].min())
    sample_columns = [
        "date", "demandQty", "lag1", "lag7", "lag14", "lag28",
        "rollingMean7", "rollingMean14", "rollingMean28",
    ]
    print(f"  Échantillon de contrôle — produit {sample_product} :")
    print(features.loc[features["productId"] == sample_product, sample_columns].head(5).to_string(index=False))


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
        eda = analyze_demand_dataset(raw_dataframe)
        promotion_activity = load_promotion_activity(
            arguments.start_date,
            arguments.end_date,
            database_url,
        )
        features = create_demand_features(raw_dataframe, promotion_activity)
        output = arguments.output if arguments.output.is_absolute() else ROOT / arguments.output
        output.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(output, index=False, encoding="utf-8")
        print_eda(eda)
        print_feature_summary(features)
        print(f"  CSV features : {output}")
        if not arguments.skip_plots:
            report_directory = arguments.report_dir if arguments.report_dir.is_absolute() else ROOT / arguments.report_dir
            charts = export_eda_charts(raw_dataframe, eda, report_directory)
            print("  Graphiques EDA :")
            for chart in charts:
                print(f"    - {chart}")
        return 0
    except (FeatureValidationError, ValueError, OSError, psycopg.Error) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
