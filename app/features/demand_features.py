"""Exploratory analysis and leakage-safe demand feature engineering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import psycopg

from app.data.extractor import connect_read_only


RAW_COLUMNS = ("date", "productId", "productReference", "demandQty")
LAG_WINDOWS = (1, 7, 14, 28)
ROLLING_WINDOWS = (7, 14, 28)
REQUIRED_HISTORY_COLUMNS = (
    "lag1",
    "lag7",
    "lag14",
    "lag28",
    "rollingMean7",
    "rollingMean14",
    "rollingMean28",
    "rollingStd7",
    "rollingStd28",
)
MINIMUM_FORECAST_HISTORY_DAYS = max(LAG_WINDOWS + ROLLING_WINDOWS)


class FeatureValidationError(RuntimeError):
    """Raised when raw demand data or generated features are invalid."""


@dataclass(frozen=True)
class EdaSummary:
    global_summary: dict[str, object]
    product_summary: pd.DataFrame
    weekday_summary: pd.DataFrame
    month_summary: pd.DataFrame


@dataclass(frozen=True)
class FeatureSummary:
    raw_row_count: int
    feature_row_count: int
    removed_for_history_count: int
    product_count: int
    minimum_date: date
    maximum_date: date
    column_count: int
    null_count: int
    duplicate_count: int


def validate_raw_demand_dataset(raw_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a complete product × daily-date demand dataset."""
    missing_columns = set(RAW_COLUMNS) - set(raw_dataframe.columns)
    if missing_columns:
        raise FeatureValidationError(
            f"Colonnes brutes manquantes : {', '.join(sorted(missing_columns))}.",
        )
    dataframe = raw_dataframe.loc[:, RAW_COLUMNS].copy()
    dataframe["date"] = pd.to_datetime(dataframe["date"], errors="raise").dt.normalize()
    dataframe["productId"] = pd.to_numeric(dataframe["productId"], errors="raise").astype("int64")
    dataframe["demandQty"] = pd.to_numeric(dataframe["demandQty"], errors="raise").astype("int64")
    if dataframe[list(RAW_COLUMNS)].isna().any().any():
        raise FeatureValidationError("Le dataset brut contient une valeur nulle.")
    if (dataframe["demandQty"] < 0).any():
        raise FeatureValidationError("demandQty ne peut pas être négatif.")
    if dataframe.duplicated(["productId", "date"]).any():
        raise FeatureValidationError("Le dataset brut contient des doublons date × productId.")
    dataframe = dataframe.sort_values(["productId", "date"], kind="stable", ignore_index=True)
    _validate_daily_continuity(dataframe)
    return dataframe


def _validate_daily_continuity(dataframe: pd.DataFrame) -> None:
    for product_id, group in dataframe.groupby("productId", sort=False):
        gaps = group["date"].diff().dropna()
        if not (gaps == pd.Timedelta(days=1)).all():
            raise FeatureValidationError(
                f"Le produit {product_id} ne possède pas une série journalière continue.",
            )


def analyze_demand_dataset(raw_dataframe: pd.DataFrame) -> EdaSummary:
    """Produce EDA aggregates without modifying the input data."""
    dataframe = validate_raw_demand_dataset(raw_dataframe)
    demand = dataframe["demandQty"]
    global_summary: dict[str, object] = {
        "row_count": len(dataframe),
        "product_count": dataframe["productId"].nunique(),
        "minimum_date": dataframe["date"].min().date(),
        "maximum_date": dataframe["date"].max().date(),
        "total_demand": int(demand.sum()),
        "mean_demand": float(demand.mean()),
        "median_demand": float(demand.median()),
        "standard_deviation": float(demand.std(ddof=0)),
        "minimum_demand": int(demand.min()),
        "maximum_demand": int(demand.max()),
        "zero_count": int((demand == 0).sum()),
        "zero_percentage": float((demand == 0).mean() * 100),
    }
    product_summary = (
        dataframe.groupby(["productId", "productReference"], as_index=False)["demandQty"]
        .agg(
            totalDemand="sum",
            meanDemand="mean",
            medianDemand="median",
            standardDeviation=lambda values: values.std(ddof=0),
            minDemand="min",
            maxDemand="max",
            zeroDays=lambda values: (values == 0).sum(),
        )
        .sort_values("productId", kind="stable", ignore_index=True)
    )
    product_summary["zeroDayPercentage"] = (
        product_summary["zeroDays"]
        / dataframe.groupby("productId").size().reindex(product_summary["productId"]).to_numpy()
        * 100
    )
    weekday_summary = (
        dataframe.assign(dayOfWeek=dataframe["date"].dt.dayofweek)
        .groupby("dayOfWeek", as_index=False)["demandQty"]
        .mean()
        .rename(columns={"demandQty": "meanDemand"})
    )
    month_summary = (
        dataframe.assign(month=dataframe["date"].dt.month)
        .groupby("month", as_index=False)["demandQty"]
        .mean()
        .rename(columns={"demandQty": "meanDemand"})
    )
    return EdaSummary(global_summary, product_summary, weekday_summary, month_summary)


def load_promotion_activity(
    start_date: date | str,
    end_date: date | str,
    database_url: str,
) -> pd.DataFrame:
    """Reconstruct planned promotion activity from the existing promotion schema.

    A product is active when it is either attached to an offer through a lot or
    explicitly configured as that offer's gifted product. Both are facts known
    from the offer dates, not inferred from demand.
    """
    start = _coerce_date(start_date)
    end = _coerce_date(end_date)
    if end < start:
        raise ValueError("La date de fin doit être postérieure ou égale à la date de début.")
    query = '''
        WITH offer_products AS (
            SELECT association."specialOfferId", inventory."productId"
            FROM "InventorySpecialOffers" association
            JOIN "Inventories" inventory ON inventory."ID" = association."inventoryId"
            UNION
            SELECT offer."ID", offer."productIdOffer"
            FROM "SpecialOffers" offer
            WHERE offer."productIdOffer" IS NOT NULL
        )
        SELECT DISTINCT
            offer_products."productId",
            offer."createdAt"::date,
            offer."startDateTime"::date,
            offer."endDateTime"::date
        FROM "SpecialOffers" offer
        JOIN offer_products ON offer_products."specialOfferId" = offer."ID"
        WHERE offer."deletedAt" IS NULL
          AND offer."startDateTime" < %s
          AND offer."endDateTime" >= %s
        ORDER BY offer_products."productId", offer."startDateTime"::date
    '''
    active_keys: set[tuple[int, date]] = set()
    with connect_read_only(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (end + timedelta(days=1), start))
            for product_id, created_date, active_start, active_end in cursor.fetchall():
                # A promotion cannot be a known feature before its record has
                # been created, even if its configured start date is earlier.
                current = max(created_date, active_start, start)
                last = min(active_end, end)
                while current <= last:
                    active_keys.add((int(product_id), current))
                    current += timedelta(days=1)
    return pd.DataFrame(
        [
            {"productId": product_id, "date": active_date, "promotionActive": 1}
            for product_id, active_date in sorted(active_keys)
        ],
        columns=["productId", "date", "promotionActive"],
    )


def create_demand_features(
    raw_dataframe: pd.DataFrame,
    promotion_activity: pd.DataFrame | None = None,
    drop_incomplete_history: bool = True,
) -> pd.DataFrame:
    """Create calendar and strictly past-looking demand features per product."""
    dataframe = validate_raw_demand_dataset(raw_dataframe)
    dataframe["dayOfWeek"] = dataframe["date"].dt.dayofweek.astype("int8")
    dataframe["dayOfMonth"] = dataframe["date"].dt.day.astype("int8")
    dataframe["month"] = dataframe["date"].dt.month.astype("int8")
    dataframe["weekOfYear"] = dataframe["date"].dt.isocalendar().week.astype("int16")
    dataframe["isWeekend"] = (dataframe["dayOfWeek"] >= 5).astype("int8")

    grouped_demand = dataframe.groupby("productId", sort=False)["demandQty"]
    for window in LAG_WINDOWS:
        dataframe[f"lag{window}"] = grouped_demand.shift(window)
    for window in ROLLING_WINDOWS:
        dataframe[f"rollingMean{window}"] = grouped_demand.transform(
            lambda values, size=window: values.shift(1).rolling(size, min_periods=size).mean(),
        )
    for window in (7, 28):
        dataframe[f"rollingStd{window}"] = grouped_demand.transform(
            lambda values, size=window: values.shift(1).rolling(size, min_periods=size).std(ddof=0),
        )

    if promotion_activity is not None:
        dataframe = _merge_promotion_activity(dataframe, promotion_activity)

    assert_no_temporal_leakage(dataframe)
    raw_row_count = len(dataframe)
    if drop_incomplete_history:
        dataframe = dataframe.dropna(subset=REQUIRED_HISTORY_COLUMNS).copy()
    dataframe = dataframe.sort_values(["productId", "date"], kind="stable", ignore_index=True)
    dataframe["date"] = dataframe["date"].dt.date
    summary = summarize_feature_dataframe(dataframe, raw_row_count)
    dataframe.attrs["feature_summary"] = summary
    return dataframe


def create_future_feature_row(
    product_history: pd.DataFrame,
    forecast_date: date | str,
    promotion_active: int,
) -> dict[str, object]:
    """Build one model-ready future row from strictly prior real/predicted demand.

    Unlike the training dataframe validator, this helper intentionally accepts
    floating demand in ``product_history`` because recursive predictions must
    retain their precision for subsequent lag and rolling features.
    """
    expected = set(RAW_COLUMNS)
    if not expected.issubset(product_history.columns):
        raise FeatureValidationError("L'historique de prévision est incomplet.")
    history = product_history.loc[:, RAW_COLUMNS].copy()
    history["date"] = pd.to_datetime(history["date"], errors="raise").dt.normalize()
    history["productId"] = pd.to_numeric(history["productId"], errors="raise").astype("int64")
    history["demandQty"] = pd.to_numeric(history["demandQty"], errors="raise").astype(float)
    if history.isna().any().any() or (history["demandQty"] < 0).any():
        raise FeatureValidationError("L'historique de prévision contient une demande invalide.")
    if history["productId"].nunique() != 1 or history["productReference"].nunique() != 1:
        raise FeatureValidationError("Une prévision doit être construite pour un seul produit.")
    if history.duplicated(["date", "productId"]).any():
        raise FeatureValidationError("L'historique de prévision contient des doublons.")
    history = history.sort_values("date", kind="stable", ignore_index=True)
    _validate_daily_continuity(history)
    target_date = pd.Timestamp(_coerce_date(forecast_date)).normalize()
    if target_date != history["date"].iloc[-1] + pd.Timedelta(days=1):
        raise FeatureValidationError("La date à prévoir doit suivre immédiatement le dernier jour historique.")
    if len(history) < MINIMUM_FORECAST_HISTORY_DAYS:
        raise FeatureValidationError(
            f"Au moins {MINIMUM_FORECAST_HISTORY_DAYS} jours d'historique sont nécessaires à la prévision.",
        )
    if promotion_active not in (0, 1):
        raise FeatureValidationError("promotion_active doit valoir 0 ou 1.")

    demand = history["demandQty"].to_numpy(dtype=float)
    product_id = int(history["productId"].iloc[0])
    reference = str(history["productReference"].iloc[0])
    day_of_week = int(target_date.dayofweek)
    row: dict[str, object] = {
        "date": target_date.date(),
        "productId": product_id,
        "productReference": reference,
        "dayOfWeek": day_of_week,
        "dayOfMonth": int(target_date.day),
        "month": int(target_date.month),
        "weekOfYear": int(target_date.isocalendar().week),
        "isWeekend": int(day_of_week >= 5),
        "promotionActive": int(promotion_active),
    }
    for window in LAG_WINDOWS:
        row[f"lag{window}"] = float(demand[-window])
    for window in ROLLING_WINDOWS:
        row[f"rollingMean{window}"] = float(demand[-window:].mean())
    for window in (7, 28):
        row[f"rollingStd{window}"] = float(demand[-window:].std(ddof=0))
    return row


def _merge_promotion_activity(
    dataframe: pd.DataFrame,
    promotion_activity: pd.DataFrame,
) -> pd.DataFrame:
    expected = {"productId", "date", "promotionActive"}
    if not expected.issubset(promotion_activity.columns):
        raise FeatureValidationError("Le DataFrame promotion_activity est incomplet.")
    activity = promotion_activity.loc[:, ["productId", "date", "promotionActive"]].copy()
    activity["date"] = pd.to_datetime(activity["date"], errors="raise").dt.normalize()
    activity["productId"] = pd.to_numeric(activity["productId"], errors="raise").astype("int64")
    activity["promotionActive"] = pd.to_numeric(
        activity["promotionActive"],
        errors="raise",
    ).astype("int8")
    if activity.duplicated(["productId", "date"]).any():
        raise FeatureValidationError("promotion_activity contient des doublons date × productId.")
    if not activity["promotionActive"].isin((0, 1)).all():
        raise FeatureValidationError("promotionActive doit valoir 0 ou 1.")
    merged = dataframe.merge(activity, on=["productId", "date"], how="left", validate="one_to_one")
    merged["promotionActive"] = merged["promotionActive"].fillna(0).astype("int8")
    return merged


def assert_no_temporal_leakage(feature_dataframe: pd.DataFrame) -> None:
    """Independently verify lags and rolling windows against strictly prior rows."""
    validation_data = feature_dataframe.sort_values(["productId", "date"], kind="stable")
    _validate_daily_continuity(validation_data)
    for product_id, group in validation_data.groupby("productId", sort=False):
        demand = group["demandQty"].to_numpy(dtype=float)
        for window in LAG_WINDOWS:
            values = group[f"lag{window}"].to_numpy(dtype=float)
            for index in range(window, len(group)):
                if not math.isclose(values[index], demand[index - window], rel_tol=0, abs_tol=1e-9):
                    raise FeatureValidationError(
                        f"Fuite ou lag{window} incorrect pour le produit {product_id}.",
                    )
        for window in ROLLING_WINDOWS:
            values = group[f"rollingMean{window}"].to_numpy(dtype=float)
            for index in range(window, len(group)):
                expected = demand[index - window:index].mean()
                if not math.isclose(values[index], expected, rel_tol=0, abs_tol=1e-9):
                    raise FeatureValidationError(
                        f"Fuite ou rollingMean{window} incorrect pour le produit {product_id}.",
                    )
        for window in (7, 28):
            values = group[f"rollingStd{window}"].to_numpy(dtype=float)
            for index in range(window, len(group)):
                expected = demand[index - window:index].std(ddof=0)
                if not math.isclose(values[index], expected, rel_tol=0, abs_tol=1e-9):
                    raise FeatureValidationError(
                        f"Fuite ou rollingStd{window} incorrect pour le produit {product_id}.",
                    )


def summarize_feature_dataframe(
    dataframe: pd.DataFrame,
    raw_row_count: int,
) -> FeatureSummary:
    duplicate_count = int(dataframe.duplicated(["productId", "date"]).sum())
    null_count = int(dataframe.isna().sum().sum())
    if duplicate_count or null_count:
        raise FeatureValidationError("Le dataset de features contient des doublons ou des valeurs nulles.")
    if (dataframe["demandQty"] < 0).any():
        raise FeatureValidationError("Le dataset de features contient une demande négative.")
    return FeatureSummary(
        raw_row_count=raw_row_count,
        feature_row_count=len(dataframe),
        removed_for_history_count=raw_row_count - len(dataframe),
        product_count=int(dataframe["productId"].nunique()),
        minimum_date=min(dataframe["date"]),
        maximum_date=max(dataframe["date"]),
        column_count=len(dataframe.columns),
        null_count=null_count,
        duplicate_count=duplicate_count,
    )


def export_eda_charts(raw_dataframe: pd.DataFrame, eda: EdaSummary, output_directory: Path) -> tuple[Path, ...]:
    """Export four compact EDA charts for human inspection."""
    output_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_directory / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataframe = validate_raw_demand_dataset(raw_dataframe)
    outputs: list[Path] = []

    daily = dataframe.groupby("date", as_index=True)["demandQty"].sum()
    figure, axis = plt.subplots(figsize=(10, 4))
    daily.plot(ax=axis, color="#2563eb")
    axis.set(title="Demande totale journalière", xlabel="Date", ylabel="Unités")
    figure.tight_layout()
    path = output_directory / "daily_total_demand.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    outputs.append(path)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(eda.weekday_summary["dayOfWeek"], eda.weekday_summary["meanDemand"], color="#16a34a")
    axis.set(title="Demande moyenne par jour de semaine", xlabel="Jour (0=lundi)", ylabel="Unités")
    figure.tight_layout()
    path = output_directory / "mean_demand_by_weekday.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    outputs.append(path)

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.bar(eda.product_summary["productReference"], eda.product_summary["meanDemand"], color="#9333ea")
    axis.set(title="Demande moyenne par produit", xlabel="Référence", ylabel="Unités")
    axis.tick_params(axis="x", rotation=70, labelsize=7)
    figure.tight_layout()
    path = output_directory / "mean_demand_by_product.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    outputs.append(path)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(dataframe["demandQty"], bins=range(int(dataframe["demandQty"].max()) + 2), color="#ea580c", align="left")
    axis.set(title="Distribution de demandQty", xlabel="DemandQty", ylabel="Nombre de lignes")
    figure.tight_layout()
    path = output_directory / "demand_distribution.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    outputs.append(path)
    return tuple(outputs)


def _coerce_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError("Les dates doivent être des date ou des chaînes YYYY-MM-DD.")
