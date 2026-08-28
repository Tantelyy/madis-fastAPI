#!/usr/bin/env python3
"""Export the daily SALE + PROMOTION_GIFT demand dataset from PostgreSQL."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.extractor import (  # noqa: E402
    DatasetValidationError,
    export_dataframe_csv,
    extract_demand_dataset,
)


def parse_date(value: str, argument_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{argument_name} doit respecter le format YYYY-MM-DD.",
        ) from error


def parse_arguments() -> argparse.Namespace:
    default_end = date.today() - timedelta(days=1)
    default_start = default_end - timedelta(days=364)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-date",
        type=lambda value: parse_date(value, "--start-date"),
        default=default_start,
        help=f"Date incluse de début (défaut : {default_start.isoformat()}).",
    )
    parser.add_argument(
        "--end-date",
        type=lambda value: parse_date(value, "--end-date"),
        default=default_end,
        help=f"Date incluse de fin (défaut : {default_end.isoformat()}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "datasets" / "demand_dataset.csv",
        help="Chemin du CSV UTF-8 (défaut : datasets/demand_dataset.csv).",
    )
    parser.add_argument(
        "--synthetic-batch",
        "--synthetic-marker",
        dest="synthetic_batch",
        help="Filtre optionnel exact sur Carts.reason ; --synthetic-marker reste un alias compatible.",
    )
    return parser.parse_args()


def print_summary(dataset_path: Path, dataframe) -> None:
    stats = dataframe.attrs["dataset_stats"]
    print("Extraction terminée")
    print(f"  CSV : {dataset_path}")
    print(f"  Produits : {stats.product_count}")
    print(f"  Jours : {stats.day_count}")
    print(f"  Lignes attendues / obtenues : {stats.expected_row_count} / {stats.row_count}")
    print(f"  Dates min / max : {stats.minimum_date} / {stats.maximum_date}")
    print(
        "  SALE / PROMOTION_GIFT / demande : "
        f"{stats.total_sale_quantity} / {stats.total_promotion_gift_quantity} / "
        f"{stats.total_demand_quantity}",
    )
    print(
        f"  Demande moyenne / min / max : {stats.average_demand:.2f} / "
        f"{stats.minimum_demand} / {stats.maximum_demand}",
    )
    print(
        f"  Lignes à zéro : {stats.zero_rows} ({stats.zero_row_percentage:.1f} %) | "
        f"doublons : {stats.duplicate_count} | null date/productId/demandQty : "
        f"{stats.null_date_count}/{stats.null_product_id_count}/{stats.null_demand_count}",
    )
    print("  Statistiques par produit :")
    for product in stats.products:
        print(
            f"    - {product.product_id} | {product.product_reference} | "
            f"jours={product.day_count}, total={product.total_demand}, "
            f"moyenne={product.average_daily_demand:.2f}, min/max="
            f"{product.minimum_demand}/{product.maximum_demand}, "
            f"écart-type={product.standard_deviation:.2f}, "
            f"zéros={product.zero_days} ({product.zero_day_percentage:.1f} %)",
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
        dataframe = extract_demand_dataset(
            arguments.start_date,
            arguments.end_date,
            synthetic_batch=arguments.synthetic_batch,
            database_url=database_url,
        )
        output_path = arguments.output if arguments.output.is_absolute() else ROOT / arguments.output
        export_dataframe_csv(dataframe, output_path)
        print_summary(output_path, dataframe)
        return 0
    except (DatasetValidationError, ValueError, OSError, psycopg.Error) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
