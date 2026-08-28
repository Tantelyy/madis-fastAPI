"""Validation-only metrics, diagnostics and charts for ML demand candidates."""

from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pandas as pd

from app.evaluation.baselines import calculate_metrics
from app.training.demand_training import CandidateResult


BASELINE_NAME = "ROLLING_MEAN_7"
BASELINE_COLUMN = "rollingMean7"


def evaluate_candidate_models(
    validation_dataframe: pd.DataFrame,
    candidates: tuple[CandidateResult, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate model candidates versus the same rolling baseline on validation only."""
    if BASELINE_COLUMN not in validation_dataframe.columns:
        raise ValueError(f"La baseline nécessite la colonne {BASELINE_COLUMN}.")
    baseline_metrics = calculate_metrics(validation_dataframe["demandQty"], validation_dataframe[BASELINE_COLUMN])
    comparison_rows: list[dict[str, object]] = []
    product_rows: list[dict[str, object]] = []
    for candidate in candidates:
        prediction = candidate.validation_prediction.reindex(validation_dataframe.index)
        model_metrics = calculate_metrics(validation_dataframe["demandQty"], prediction)
        comparison_rows.append(
            {
                "model": candidate.name,
                **model_metrics,
                "baseline": BASELINE_NAME,
                "baselineMAE": baseline_metrics["MAE"],
                "improvementVsBaseline": (baseline_metrics["MAE"] - model_metrics["MAE"])
                / baseline_metrics["MAE"]
                * 100,
                "clippedPredictionCount": candidate.clipped_prediction_count,
            },
        )
        for product_id, product_data in validation_dataframe.groupby("productId", sort=True):
            product_prediction = prediction.loc[product_data.index]
            metrics = calculate_metrics(product_data["demandQty"], product_prediction)
            baseline_product_metrics = calculate_metrics(
                product_data["demandQty"],
                product_data[BASELINE_COLUMN],
            )
            product_rows.append(
                {
                    "model": candidate.name,
                    "productId": int(product_id),
                    "productReference": product_data["productReference"].iloc[0],
                    "meanActualDemand": float(product_data["demandQty"].mean()),
                    **metrics,
                    "baselineMAE": baseline_product_metrics["MAE"],
                    "maeImprovementVsBaseline": (baseline_product_metrics["MAE"] - metrics["MAE"])
                    / baseline_product_metrics["MAE"]
                    * 100,
                },
            )
    comparison = pd.DataFrame(comparison_rows).sort_values("MAE", kind="stable", ignore_index=True)
    product_metrics = pd.DataFrame(product_rows).sort_values(["model", "productId"], kind="stable", ignore_index=True)
    return comparison, product_metrics


def analyze_errors_by_demand_level(
    validation_dataframe: pd.DataFrame,
    candidate: CandidateResult,
) -> pd.DataFrame:
    """Describe validation errors for zero, low, medium and high actual demand."""
    return analyze_prediction_by_demand_level(
        validation_dataframe,
        candidate.validation_prediction,
        candidate.name,
    )


def analyze_prediction_by_demand_level(
    validation_dataframe: pd.DataFrame,
    prediction: pd.Series,
    model_name: str,
) -> pd.DataFrame:
    """Calculate identical demand-level diagnostics for any prediction series."""
    analysis = validation_dataframe.loc[:, ["demandQty"]].copy()
    analysis["prediction"] = prediction.reindex(validation_dataframe.index)
    analysis["demandLevel"] = pd.cut(
        analysis["demandQty"],
        bins=[-1, 0, 3, 10, np.inf],
        labels=["0", "1-3", "4-10", ">10"],
    )
    rows: list[dict[str, object]] = []
    for level, group in analysis.groupby("demandLevel", observed=False):
        metrics = calculate_metrics(group["demandQty"], group["prediction"])
        rows.append({"model": model_name, "demandLevel": str(level), "observationCount": len(group), **metrics})
    return pd.DataFrame(rows)


def analyze_zero_demand_by_product(
    validation_dataframe: pd.DataFrame,
    random_forest_prediction: pd.Series,
) -> pd.DataFrame:
    """Compare original RF and rolling baseline on each product's zero-demand profile."""
    rows: list[dict[str, object]] = []
    for product_id, group in validation_dataframe.groupby("productId", sort=True):
        baseline_metrics = calculate_metrics(group["demandQty"], group[BASELINE_COLUMN])
        forest_metrics = calculate_metrics(
            group["demandQty"],
            random_forest_prediction.reindex(group.index),
        )
        zero_days = int((group["demandQty"] == 0).sum())
        rows.append(
            {
                "productId": int(product_id),
                "productReference": group["productReference"].iloc[0],
                "dayCount": len(group),
                "zeroDemandDays": zero_days,
                "zeroDemandPercentage": zero_days / len(group) * 100,
                "meanActualDemand": float(group["demandQty"].mean()),
                "baselineMAE": baseline_metrics["MAE"],
                "randomForestMAE": forest_metrics["MAE"],
            },
        )
    return pd.DataFrame(rows)


def build_hybrid_per_product(
    validation_dataframe: pd.DataFrame,
    candidate: CandidateResult,
) -> tuple[pd.DataFrame, pd.Series, dict[str, float]]:
    """Choose baseline or one global ML candidate once per product on validation.

    This is deliberately an exploratory validation-only oracle: its choices must
    be frozen before a later, single TEST evaluation.
    """
    prediction = candidate.validation_prediction.reindex(validation_dataframe.index)
    selection_rows: list[dict[str, object]] = []
    hybrid_prediction = validation_dataframe[BASELINE_COLUMN].astype(float).copy()
    for product_id, group in validation_dataframe.groupby("productId", sort=True):
        baseline_metrics = calculate_metrics(group["demandQty"], group[BASELINE_COLUMN])
        ml_metrics = calculate_metrics(group["demandQty"], prediction.loc[group.index])
        winner = "ML" if ml_metrics["MAE"] < baseline_metrics["MAE"] else "BASELINE"
        if winner == "ML":
            hybrid_prediction.loc[group.index] = prediction.loc[group.index]
        selection_rows.append(
            {
                "productId": int(product_id),
                "productReference": group["productReference"].iloc[0],
                "baselineMAE": baseline_metrics["MAE"],
                "mlMAE": ml_metrics["MAE"],
                "winner": winner,
            },
        )
    return pd.DataFrame(selection_rows), hybrid_prediction, calculate_metrics(
        validation_dataframe["demandQty"],
        hybrid_prediction,
    )


def evaluate_test_per_product(
    test_dataframe: pd.DataFrame,
    ml_prediction: pd.Series,
) -> pd.DataFrame:
    """Compare the frozen ML candidate and baseline per product on TEST."""
    rows: list[dict[str, object]] = []
    for product_id, group in test_dataframe.groupby("productId", sort=True):
        baseline = calculate_metrics(group["demandQty"], group[BASELINE_COLUMN])
        ml = calculate_metrics(group["demandQty"], ml_prediction.reindex(group.index))
        winner = "ML" if ml["MAE"] < baseline["MAE"] else "BASELINE"
        rows.append(
            {
                "productId": int(product_id),
                "productReference": group["productReference"].iloc[0],
                "meanActualDemand": float(group["demandQty"].mean()),
                "baselineMAE": baseline["MAE"],
                "mlMAE": ml["MAE"],
                "baselineRMSE": baseline["RMSE"],
                "mlRMSE": ml["RMSE"],
                "maeImprovementVsBaseline": (baseline["MAE"] - ml["MAE"]) / baseline["MAE"] * 100,
                "winner": winner,
            },
        )
    return pd.DataFrame(rows)


def evaluate_frozen_hybrid_on_test(
    test_dataframe: pd.DataFrame,
    ml_prediction: pd.Series,
    frozen_selection: pd.DataFrame,
) -> tuple[pd.Series, dict[str, float]]:
    """Apply an existing validation-frozen product mapping without inspecting TEST errors."""
    expected = {"productId", "winner"}
    if not expected.issubset(frozen_selection.columns):
        raise ValueError("Le mapping hybride figé est incomplet.")
    mapping = frozen_selection.set_index("productId")["winner"]
    products = set(test_dataframe["productId"])
    if set(mapping.index) != products or not mapping.isin(("ML", "BASELINE")).all():
        raise ValueError("Le mapping hybride figé ne correspond pas exactement aux produits TEST.")
    hybrid_prediction = test_dataframe[BASELINE_COLUMN].astype(float).copy()
    for product_id, indices in test_dataframe.groupby("productId", sort=False).groups.items():
        if mapping.loc[product_id] == "ML":
            hybrid_prediction.loc[indices] = ml_prediction.loc[indices]
    return hybrid_prediction, calculate_metrics(test_dataframe["demandQty"], hybrid_prediction)


def export_final_mae_comparison_chart(
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
    output_path: Path,
) -> Path:
    """Export the selected baseline/ML MAE comparison on validation and test."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Validation", "Test"]
    positions = np.arange(len(labels))
    width = 0.34
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(
        positions - width / 2,
        [validation_metrics["baseline"], test_metrics["baseline"]],
        width,
        label=BASELINE_NAME,
        color="#64748b",
    )
    axis.bar(
        positions + width / 2,
        [validation_metrics["model"], test_metrics["model"]],
        width,
        label="EXTRA_TREES_SMOOTH",
        color="#2563eb",
    )
    axis.set_xticks(positions, labels)
    axis.set(title="MAE validation et TEST", ylabel="MAE")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def export_test_prediction_example(
    test_dataframe: pd.DataFrame,
    ml_prediction: pd.Series,
    output_path: Path,
    product_id: int | None = None,
) -> Path:
    """Export one TEST-only real-versus-frozen-model trace for a selected product."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if product_id is None:
        means = test_dataframe.groupby("productId")["demandQty"].mean().sort_values()
        product_id = int(means.index[len(means) // 2])
    sample = test_dataframe[test_dataframe["productId"] == product_id].copy()
    if sample.empty:
        raise ValueError(f"Le produit {product_id} est absent du TEST.")
    sample["prediction"] = ml_prediction.loc[sample.index]
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(pd.to_datetime(sample["date"]), sample["demandQty"], label="Réel", color="#111827")
    axis.plot(pd.to_datetime(sample["date"]), sample["prediction"], label="EXTRA_TREES_SMOOTH", color="#2563eb")
    axis.set(
        title=f"TEST — produit {product_id} ({sample['productReference'].iloc[0]})",
        xlabel="Date",
        ylabel="DemandQty",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def export_model_comparison_chart(comparison: pd.DataFrame, output_path: Path) -> Path:
    """Export one concise MAE comparison chart, including the baseline reference."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = comparison.loc[:, ["model", "MAE", "baselineMAE"]].copy()
    labels = [BASELINE_NAME, *data["model"].tolist()]
    values = [float(data["baselineMAE"].iloc[0]), *data["MAE"].tolist()]
    colors = ["#64748b", *["#2563eb"] * len(data)]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(labels, values, color=colors)
    axis.set(title="MAE de validation : baseline et candidats ML", ylabel="MAE")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def export_validation_prediction_example(
    validation_dataframe: pd.DataFrame,
    candidate: CandidateResult,
    output_path: Path,
) -> Path:
    """Export one validation-only actual-versus-prediction example for a typical product."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    product_means = validation_dataframe.groupby("productId")["demandQty"].mean().sort_values()
    product_id = int(product_means.index[len(product_means) // 2])
    sample = validation_dataframe[validation_dataframe["productId"] == product_id].copy()
    sample["prediction"] = candidate.validation_prediction.loc[sample.index]
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(pd.to_datetime(sample["date"]), sample["demandQty"], label="Réel", color="#111827")
    axis.plot(pd.to_datetime(sample["date"]), sample["prediction"], label=candidate.name, color="#2563eb")
    axis.set(
        title=f"Validation — produit {product_id} ({sample['productReference'].iloc[0]})",
        xlabel="Date",
        ylabel="DemandQty",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
