"""Date-safe dataset splits and non-ML demand baseline evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import pandas as pd


BASELINE_FEATURES = {
    "NAIVE_LAG1": "lag1",
    "NAIVE_LAG7": "lag7",
    "ROLLING_MEAN_7": "rollingMean7",
}


class EvaluationValidationError(RuntimeError):
    """Raised when a chronological split or baseline prediction is invalid."""


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class SplitPartitionSummary:
    name: str
    minimum_date: object
    maximum_date: object
    date_count: int
    row_count: int
    product_count: int


def chronological_split(
    feature_dataframe: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> ChronologicalSplit:
    """Split by unique dates, keeping every product of a date together."""
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1:
        raise ValueError("Les proportions train et validation doivent être entre 0 et 1.")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("Les proportions train + validation doivent rester inférieures à 1.")
    required_columns = {"date", "productId", "demandQty", *BASELINE_FEATURES.values()}
    missing = required_columns - set(feature_dataframe.columns)
    if missing:
        raise EvaluationValidationError(
            f"Colonnes nécessaires aux baselines manquantes : {', '.join(sorted(missing))}.",
        )
    dataframe = feature_dataframe.copy()
    dataframe["date"] = pd.to_datetime(dataframe["date"], errors="raise").dt.normalize()
    if dataframe.duplicated(["date", "productId"]).any():
        raise EvaluationValidationError("Le dataset contient des doublons date × productId.")
    date_product_counts = dataframe.groupby("date")["productId"].nunique()
    if date_product_counts.nunique() != 1:
        raise EvaluationValidationError(
            "Chaque date doit contenir le même ensemble complet de produits avant le split.",
        )
    expected_products = set(dataframe["productId"])
    if any(
        set(group["productId"]) != expected_products
        for _, group in dataframe.groupby("date", sort=False)
    ):
        raise EvaluationValidationError(
            "L'ensemble des produits diffère entre au moins deux dates.",
        )
    dates = pd.Index(sorted(dataframe["date"].unique()))
    if len(dates) < 3:
        raise EvaluationValidationError("Au moins trois dates sont nécessaires au split.")
    train_date_count = round(len(dates) * train_ratio)
    validation_date_count = round(len(dates) * validation_ratio)
    test_date_count = len(dates) - train_date_count - validation_date_count
    if min(train_date_count, validation_date_count, test_date_count) < 1:
        raise EvaluationValidationError("Une partition chronologique serait vide.")

    train_dates = dates[:train_date_count]
    validation_dates = dates[train_date_count : train_date_count + validation_date_count]
    test_dates = dates[train_date_count + validation_date_count :]
    split = ChronologicalSplit(
        train=dataframe[dataframe["date"].isin(train_dates)].copy(),
        validation=dataframe[dataframe["date"].isin(validation_dates)].copy(),
        test=dataframe[dataframe["date"].isin(test_dates)].copy(),
    )
    validate_chronological_split(split, len(dataframe), date_product_counts.iloc[0])
    return split


def summarize_partition(name: str, dataframe: pd.DataFrame) -> SplitPartitionSummary:
    dates = pd.to_datetime(dataframe["date"])
    return SplitPartitionSummary(
        name=name,
        minimum_date=dates.min().date(),
        maximum_date=dates.max().date(),
        date_count=dates.nunique(),
        row_count=len(dataframe),
        product_count=dataframe["productId"].nunique(),
    )


def validate_chronological_split(
    split: ChronologicalSplit,
    original_row_count: int,
    expected_product_count: int,
) -> None:
    """Assert strict order, disjoint dates and absence of dropped rows."""
    summaries = [
        summarize_partition("TRAIN", split.train),
        summarize_partition("VALIDATION", split.validation),
        summarize_partition("TEST", split.test),
    ]
    if any(summary.product_count != expected_product_count for summary in summaries):
        raise EvaluationValidationError("Une partition ne contient pas tous les produits.")
    if sum(summary.row_count for summary in summaries) != original_row_count:
        raise EvaluationValidationError("Des lignes ont été perdues ou dupliquées pendant le split.")
    train_dates = set(pd.to_datetime(split.train["date"]))
    validation_dates = set(pd.to_datetime(split.validation["date"]))
    test_dates = set(pd.to_datetime(split.test["date"]))
    if train_dates & validation_dates or train_dates & test_dates or validation_dates & test_dates:
        raise EvaluationValidationError("Des dates sont partagées entre les partitions.")
    if not (
        summaries[0].maximum_date < summaries[1].minimum_date
        and summaries[1].maximum_date < summaries[2].minimum_date
    ):
        raise EvaluationValidationError("L'ordre TRAIN < VALIDATION < TEST n'est pas strict.")


def calculate_metrics(y_true: pd.Series, y_prediction: pd.Series) -> dict[str, float]:
    """Calculate unweighted row-level MAE, RMSE and WAPE."""
    actual = pd.to_numeric(y_true, errors="raise").astype(float)
    predicted = pd.to_numeric(y_prediction, errors="raise").astype(float)
    if actual.isna().any() or predicted.isna().any():
        raise EvaluationValidationError("Une baseline contient une valeur nulle.")
    if (predicted < 0).any():
        raise EvaluationValidationError("Une baseline a produit une prédiction négative.")
    absolute_errors = (actual - predicted).abs()
    total_actual = actual.sum()
    return {
        "MAE": float(absolute_errors.mean()),
        "RMSE": float(math.sqrt(((actual - predicted) ** 2).mean())),
        "WAPE": float(absolute_errors.sum() / total_actual * 100) if total_actual else float("nan"),
    }


def evaluate_baselines(validation_dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all simple baselines on validation only, globally and per product."""
    global_rows: list[dict[str, object]] = []
    product_rows: list[dict[str, object]] = []
    for baseline_name, feature_name in BASELINE_FEATURES.items():
        prediction = validation_dataframe[feature_name]
        global_rows.append({"baseline": baseline_name, **calculate_metrics(validation_dataframe["demandQty"], prediction)})
        for product_id, product_data in validation_dataframe.groupby("productId", sort=True):
            metrics = calculate_metrics(product_data["demandQty"], product_data[feature_name])
            product_rows.append(
                {
                    "baseline": baseline_name,
                    "productId": int(product_id),
                    "productReference": product_data["productReference"].iloc[0],
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "meanActualDemand": float(product_data["demandQty"].mean()),
                },
            )
    global_results = pd.DataFrame(global_rows).sort_values("MAE", kind="stable", ignore_index=True)
    product_results = pd.DataFrame(product_rows).sort_values(
        ["baseline", "productId"],
        kind="stable",
        ignore_index=True,
    )
    return global_results, product_results


def export_split_plot(split: ChronologicalSplit, output_path: Path) -> Path:
    """Export one compact chart showing temporal TRAIN / VALIDATION / TEST coverage."""
    import os

    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 3.5))
    for label, dataframe, color in (
        ("TRAIN", split.train, "#2563eb"),
        ("VALIDATION", split.validation, "#f59e0b"),
        ("TEST", split.test, "#dc2626"),
    ):
        daily_total = dataframe.groupby("date", as_index=True)["demandQty"].sum()
        axis.plot(pd.to_datetime(daily_total.index), daily_total.values, label=label, color=color)
    axis.set(title="Séparation chronologique de la demande", xlabel="Date", ylabel="Demande totale")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
