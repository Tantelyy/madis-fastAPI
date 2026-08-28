"""Recursive J+1 to J+7 demand inference without writing inventory movements."""

from __future__ import annotations

from datetime import date, timedelta
import json
import math
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.data.extractor import (
    DatasetValidationError,
    extract_demand_dataset,
    get_active_product_reference,
    get_latest_demand_date,
)
from app.features.demand_features import (
    MINIMUM_FORECAST_HISTORY_DAYS,
    FeatureValidationError,
    create_future_feature_row,
    load_promotion_activity,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "artifacts" / "models" / "demand_model_deployment.joblib"
DEFAULT_METADATA_PATH = ROOT / "artifacts" / "models" / "demand_model_deployment_metadata.json"
MAX_FORECAST_DAYS = 7
HISTORY_LOOKBACK_DAYS = 60


class PredictionValidationError(RuntimeError):
    """Raised when a reliable demand forecast cannot be built."""


class ProductNotFoundError(PredictionValidationError):
    """Raised when the requested active product does not exist."""


class InsufficientHistoryError(PredictionValidationError):
    """Raised when no complete historical window is available for a product."""


class UnknownModelProductError(PredictionValidationError):
    """Raised when the product was not represented while the model was trained."""


class DemandModelUnavailableError(PredictionValidationError):
    """Raised when the local deployment artifact cannot safely be used."""


class DeploymentModelBundle:
    """The loaded inference pipeline and its public metadata, kept in memory."""

    def __init__(self, pipeline: Any, metadata: dict[str, Any]) -> None:
        self.pipeline = pipeline
        self.metadata = metadata


def predict_demand(
    product_id: int,
    forecast_days: int = 7,
    as_of_date: date | str | None = None,
    synthetic_batch: str | None = None,
    database_url: str | None = None,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    metadata_path: Path | str = DEFAULT_METADATA_PATH,
    model_bundle: DeploymentModelBundle | None = None,
) -> dict[str, Any]:
    """Forecast 1 to 7 consecutive days using recursive, in-memory features.

    ``as_of_date`` is the final date whose actual demand is known. Predictions
    are never inserted into PostgreSQL; only the local ``history`` dataframe is
    extended between recursive steps.
    """
    if not isinstance(product_id, int) or isinstance(product_id, bool) or product_id <= 0:
        raise PredictionValidationError("product_id doit être un entier strictement positif.")
    if not isinstance(forecast_days, int) or isinstance(forecast_days, bool) or not 1 <= forecast_days <= MAX_FORECAST_DAYS:
        raise PredictionValidationError(f"forecast_days doit être compris entre 1 et {MAX_FORECAST_DAYS}.")
    database_url = database_url or os.getenv("DATABASE_URL")
    if not database_url:
        raise PredictionValidationError("DATABASE_URL est obligatoire.")

    bundle = model_bundle or load_deployment_model(Path(model_path), Path(metadata_path))
    model, metadata = bundle.pipeline, bundle.metadata
    active_reference = get_active_product_reference(product_id, database_url)
    if active_reference is None:
        raise ProductNotFoundError(f"Le produit actif {product_id} est introuvable.")
    _validate_product_seen_during_training(model, product_id)

    latest_available_date = get_latest_demand_date(synthetic_batch, database_url)
    effective_as_of = _coerce_date(as_of_date) if as_of_date is not None else latest_available_date
    if effective_as_of > latest_available_date:
        raise PredictionValidationError(
            f"as_of_date ({effective_as_of}) est postérieure à la dernière demande disponible ({latest_available_date}).",
        )

    history_start = effective_as_of - timedelta(days=HISTORY_LOOKBACK_DAYS - 1)
    history = _load_product_history(
        product_id,
        active_reference,
        history_start,
        effective_as_of,
        synthetic_batch,
        database_url,
    )
    feature_columns = tuple(metadata["features"])
    promotion_dates = load_promotion_activity(
        effective_as_of + timedelta(days=1),
        effective_as_of + timedelta(days=forecast_days),
        database_url,
    )
    active_promotions = {
        (int(row.productId), _coerce_date(row.date))
        for row in promotion_dates.itertuples(index=False)
    }

    predictions: list[dict[str, object]] = []
    for offset in range(1, forecast_days + 1):
        forecast_date = effective_as_of + timedelta(days=offset)
        promotion_active = int((product_id, forecast_date) in active_promotions)
        feature_row = create_future_feature_row(history, forecast_date, promotion_active)
        missing_features = set(feature_columns) - set(feature_row)
        if missing_features:
            raise PredictionValidationError(
                f"Le modèle requiert des features indisponibles : {', '.join(sorted(missing_features))}.",
            )
        model_input = pd.DataFrame([{column: feature_row[column] for column in feature_columns}])
        raw_prediction = float(np.asarray(model.predict(model_input), dtype=float)[0])
        predicted_demand = max(0.0, raw_prediction)
        predictions.append(
            {
                "date": forecast_date.isoformat(),
                "predictedDemand": predicted_demand,
                "predictedQuantity": int(math.floor(predicted_demand + 0.5)),
                "promotionActive": promotion_active,
            },
        )
        # Preserve the unrounded float for all subsequent lag/rolling features.
        history.loc[len(history)] = {
            "date": forecast_date,
            "productId": product_id,
            "productReference": active_reference,
            "demandQty": predicted_demand,
        }

    return {
        "productId": product_id,
        "productReference": active_reference,
        "asOfDate": effective_as_of.isoformat(),
        "forecastDays": forecast_days,
        "modelName": metadata["modelName"],
        "predictions": predictions,
    }


def load_deployment_model(model_path: Path = DEFAULT_MODEL_PATH, metadata_path: Path = DEFAULT_METADATA_PATH) -> DeploymentModelBundle:
    """Load deployment artifacts once; callers can reuse the returned bundle."""
    if not model_path.exists() or not metadata_path.exists():
        raise DemandModelUnavailableError("Le pipeline de déploiement ou ses métadonnées sont introuvables.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("modelPurpose") != "deployment" or not metadata.get("trainedOnFullAvailableHistory"):
        raise DemandModelUnavailableError("Les métadonnées ne correspondent pas à un modèle de déploiement complet.")
    if not isinstance(metadata.get("features"), list) or not metadata["features"]:
        raise DemandModelUnavailableError("Les métadonnées ne définissent pas les features du modèle.")
    try:
        pipeline = joblib.load(model_path)
    except (OSError, ValueError, TypeError) as error:
        raise DemandModelUnavailableError("Le pipeline de déploiement ne peut pas être chargé.") from error
    return DeploymentModelBundle(pipeline, metadata)


def _validate_product_seen_during_training(model: Any, product_id: int) -> None:
    try:
        encoder = model.named_steps["preprocessor"].named_transformers_["product"]
        trained_product_ids = {int(value) for value in encoder.categories_[0]}
    except (AttributeError, KeyError, IndexError, TypeError) as error:
        raise DemandModelUnavailableError("Le pipeline de déploiement ne contient pas le OneHotEncoder attendu.") from error
    if product_id not in trained_product_ids:
        raise UnknownModelProductError(
            f"Le produit {product_id} n'était pas présent pendant l'entraînement du modèle ; prévision refusée.",
        )


def _load_product_history(
    product_id: int,
    expected_reference: str,
    start_date: date,
    end_date: date,
    synthetic_batch: str | None,
    database_url: str,
) -> pd.DataFrame:
    try:
        demand = extract_demand_dataset(
            start_date,
            end_date,
            synthetic_batch=synthetic_batch,
            database_url=database_url,
        )
    except DatasetValidationError as error:
        raise InsufficientHistoryError("Aucun historique de demande exploitable n'a été trouvé.") from error
    history = demand[demand["productId"] == product_id].copy()
    if history.empty:
        raise InsufficientHistoryError(
            f"Le produit {product_id} ne possède pas de demande sur la fenêtre d'historique requise.",
        )
    if history["productReference"].iloc[0] != expected_reference:
        raise PredictionValidationError("La référence historique du produit ne correspond pas à la référence active.")
    history = history.sort_values("date", kind="stable", ignore_index=True)
    if len(history) < MINIMUM_FORECAST_HISTORY_DAYS:
        raise PredictionValidationError(
            f"Historique insuffisant pour le produit {product_id} : {len(history)} jour(s), "
            f"{MINIMUM_FORECAST_HISTORY_DAYS} requis.",
        )
    if history["date"].min() != start_date or history["date"].max() != end_date:
        raise InsufficientHistoryError("La fenêtre d'historique du produit est incomplète.")
    return history


def _coerce_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as error:
            raise PredictionValidationError("Les dates doivent respecter le format YYYY-MM-DD.") from error
    raise PredictionValidationError("as_of_date doit être une date ou une chaîne YYYY-MM-DD.")
