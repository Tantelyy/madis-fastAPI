"""Leakage-safe training of a small, reproducible set of demand candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


RANDOM_STATE = 42
TARGET_COLUMN = "demandQty"
CATEGORICAL_FEATURES = ("productId",)
CALENDAR_FEATURES = (
    "dayOfWeek",
    "dayOfMonth",
    "month",
    "weekOfYear",
    "isWeekend",
)
HISTORICAL_FEATURES = (
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
OPTIONAL_PROMOTION_FEATURE = "promotionActive"
EXTRA_TREES_SMOOTH_PARAMETERS = {
    "n_estimators": 400,
    "max_depth": 12,
    "min_samples_leaf": 4,
    "max_features": 0.7,
    "n_jobs": 1,
    "random_state": RANDOM_STATE,
}


class TrainingValidationError(RuntimeError):
    """Raised when candidate training would use invalid or leaking data."""


@dataclass(frozen=True)
class CandidateResult:
    """A fitted candidate and its validation-only prediction information."""

    name: str
    pipeline: Any
    parameters: dict[str, Any]
    feature_columns: tuple[str, ...]
    validation_prediction: pd.Series
    clipped_prediction_count: int


def select_feature_columns(
    dataframe: pd.DataFrame,
    include_promotion_active: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return categorical and numerical features that are known at prediction time."""
    required = {TARGET_COLUMN, *CATEGORICAL_FEATURES, *CALENDAR_FEATURES, *HISTORICAL_FEATURES}
    missing = required - set(dataframe.columns)
    if missing:
        raise TrainingValidationError(
            f"Colonnes nécessaires à l'entraînement absentes : {', '.join(sorted(missing))}.",
        )
    if dataframe[TARGET_COLUMN].isna().any() or (dataframe[TARGET_COLUMN] < 0).any():
        raise TrainingValidationError("La cible demandQty doit être non nulle et supérieure ou égale à zéro.")

    numerical = list(CALENDAR_FEATURES + HISTORICAL_FEATURES)
    if include_promotion_active:
        if OPTIONAL_PROMOTION_FEATURE not in dataframe.columns:
            raise TrainingValidationError("promotionActive a été demandée mais est absente du dataset.")
        if not dataframe[OPTIONAL_PROMOTION_FEATURE].isin((0, 1)).all():
            raise TrainingValidationError("promotionActive doit uniquement contenir 0 ou 1.")
        numerical.append(OPTIONAL_PROMOTION_FEATURE)
    return CATEGORICAL_FEATURES, tuple(numerical)


def _build_preprocessor(categorical: tuple[str, ...], numerical: tuple[str, ...]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "product",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical),
            ),
            ("numeric", "passthrough", list(numerical)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _candidate_estimators() -> tuple[tuple[str, Any, dict[str, Any]], ...]:
    """Return three deliberately small first-pass configurations (not a grid search)."""
    hist_parameters = {
        "loss": "poisson",
        "learning_rate": 0.07,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 12,
        "l2_regularization": 0.1,
        "random_state": RANDOM_STATE,
    }
    forest_parameters = {
        "n_estimators": 300,
        "max_depth": 16,
        "min_samples_leaf": 2,
        "max_features": 0.8,
        "n_jobs": 1,
        "random_state": RANDOM_STATE,
    }
    xgboost_parameters = {
        "objective": "count:poisson",
        "eval_metric": "poisson-nloglik",
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 2,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "n_jobs": 1,
        "random_state": RANDOM_STATE,
    }
    return (
        ("HIST_GRADIENT_BOOSTING_POISSON", HistGradientBoostingRegressor(**hist_parameters), hist_parameters),
        ("RANDOM_FOREST", RandomForestRegressor(**forest_parameters), forest_parameters),
        ("XGBOOST_POISSON", XGBRegressor(**xgboost_parameters), xgboost_parameters),
    )


def _extended_candidate_estimators() -> tuple[tuple[str, Any, dict[str, Any]], ...]:
    """Return a deliberately limited set of validation-only follow-up variants."""
    return (
        (
            "HIST_GRADIENT_BOOSTING_SQUARED_ERROR",
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.07,
                max_iter=300,
                max_leaf_nodes=31,
                min_samples_leaf=12,
                l2_regularization=0.1,
                random_state=RANDOM_STATE,
            ),
            {
                "loss": "squared_error",
                "learning_rate": 0.07,
                "max_iter": 300,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 12,
                "l2_regularization": 0.1,
                "random_state": RANDOM_STATE,
            },
        ),
        (
            "RANDOM_FOREST_SMOOTH",
            RandomForestRegressor(
                n_estimators=400,
                max_depth=12,
                min_samples_leaf=4,
                max_features=0.7,
                n_jobs=1,
                random_state=RANDOM_STATE,
            ),
            {
                "n_estimators": 400,
                "max_depth": 12,
                "min_samples_leaf": 4,
                "max_features": 0.7,
                "random_state": RANDOM_STATE,
            },
        ),
        (
            "RANDOM_FOREST_DEEP",
            RandomForestRegressor(
                n_estimators=400,
                max_depth=22,
                min_samples_leaf=1,
                max_features=1.0,
                n_jobs=1,
                random_state=RANDOM_STATE,
            ),
            {
                "n_estimators": 400,
                "max_depth": 22,
                "min_samples_leaf": 1,
                "max_features": 1.0,
                "random_state": RANDOM_STATE,
            },
        ),
        (
            "RANDOM_FOREST_BALANCED",
            RandomForestRegressor(
                n_estimators=500,
                max_depth=18,
                min_samples_leaf=2,
                max_features=0.6,
                n_jobs=1,
                random_state=RANDOM_STATE,
            ),
            {
                "n_estimators": 500,
                "max_depth": 18,
                "min_samples_leaf": 2,
                "max_features": 0.6,
                "random_state": RANDOM_STATE,
            },
        ),
        (
            "EXTRA_TREES",
            ExtraTreesRegressor(
                n_estimators=400,
                max_depth=16,
                min_samples_leaf=2,
                max_features=0.8,
                n_jobs=1,
                random_state=RANDOM_STATE,
            ),
            {
                "n_estimators": 400,
                "max_depth": 16,
                "min_samples_leaf": 2,
                "max_features": 0.8,
                "random_state": RANDOM_STATE,
            },
        ),
        (
            "EXTRA_TREES_SMOOTH",
            build_extra_trees_smooth_regressor(),
            EXTRA_TREES_SMOOTH_PARAMETERS.copy(),
        ),
        (
            "XGBOOST_SQUARED_ERROR",
            XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=2,
                reg_lambda=1.0,
                tree_method="hist",
                n_jobs=1,
                random_state=RANDOM_STATE,
            ),
            {
                "objective": "reg:squarederror",
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "min_child_weight": 2,
                "reg_lambda": 1.0,
                "random_state": RANDOM_STATE,
            },
        ),
    )


def build_extra_trees_smooth_regressor() -> ExtraTreesRegressor:
    """Build the frozen validation-selected Extra Trees configuration."""
    return ExtraTreesRegressor(**EXTRA_TREES_SMOOTH_PARAMETERS)


def build_extra_trees_smooth_pipeline(
    dataframe: pd.DataFrame,
    include_promotion_active: bool,
) -> tuple[Pipeline, tuple[str, ...]]:
    """Create the unfitted frozen candidate pipeline and its fixed feature order."""
    categorical, numerical = select_feature_columns(dataframe, include_promotion_active)
    feature_columns = categorical + numerical
    return Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(categorical, numerical)),
            ("model", build_extra_trees_smooth_regressor()),
        ],
    ), feature_columns


class TwoStageDemandModel:
    """Predict zero/non-zero first, then regress only predicted positive demand."""

    def __init__(self, categorical: tuple[str, ...], numerical: tuple[str, ...]) -> None:
        self.classifier = Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor(categorical, numerical)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=12,
                        min_samples_leaf=2,
                        max_features=0.8,
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ],
        )
        self.regressor = Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor(categorical, numerical)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=16,
                        min_samples_leaf=2,
                        max_features=0.8,
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ],
        )

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "TwoStageDemandModel":
        positive = target > 0
        if positive.sum() == 0:
            raise TrainingValidationError("Le modèle two-stage nécessite au moins une demande positive dans TRAIN.")
        self.classifier.fit(features, positive.astype(int))
        self.regressor.fit(features.loc[positive], target.loc[positive])
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        probabilities = self.classifier.predict_proba(features)
        positive_class_index = int(np.where(self.classifier.named_steps["model"].classes_ == 1)[0][0])
        positive_prediction = probabilities[:, positive_class_index] >= 0.5
        prediction = np.zeros(len(features), dtype=float)
        if positive_prediction.any():
            prediction[positive_prediction] = self.regressor.predict(features.loc[positive_prediction])
        return prediction


def train_and_evaluate_candidates(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    include_promotion_active: bool,
    include_follow_up_variants: bool = False,
    include_two_stage: bool = False,
) -> tuple[CandidateResult, ...]:
    """Fit candidates on TRAIN only and return unrounded validation predictions."""
    categorical, numerical = select_feature_columns(train_dataframe, include_promotion_active)
    select_feature_columns(validation_dataframe, include_promotion_active)
    feature_columns = categorical + numerical
    if train_dataframe[list(feature_columns)].isna().any().any():
        raise TrainingValidationError("TRAIN contient une valeur nulle dans les features.")
    if validation_dataframe[list(feature_columns)].isna().any().any():
        raise TrainingValidationError("VALIDATION contient une valeur nulle dans les features.")

    train_x = train_dataframe.loc[:, feature_columns]
    train_y = train_dataframe[TARGET_COLUMN]
    validation_x = validation_dataframe.loc[:, feature_columns]
    results: list[CandidateResult] = []
    estimators = _candidate_estimators()
    if include_follow_up_variants:
        estimators += _extended_candidate_estimators()
    for name, estimator, parameters in estimators:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor(categorical, numerical)),
                ("model", estimator),
            ],
        )
        # No validation data is supplied here: fitting uses TRAIN exclusively.
        pipeline.fit(train_x, train_y)
        raw_prediction = np.asarray(pipeline.predict(validation_x), dtype=float)
        clipped_prediction = np.maximum(raw_prediction, 0.0)
        results.append(
            CandidateResult(
                name=name,
                pipeline=pipeline,
                parameters=parameters,
                feature_columns=feature_columns,
                validation_prediction=pd.Series(clipped_prediction, index=validation_dataframe.index, name=name),
                clipped_prediction_count=int((raw_prediction < 0).sum()),
            ),
        )
    if include_two_stage:
        two_stage = TwoStageDemandModel(categorical, numerical).fit(train_x, train_y)
        raw_prediction = np.asarray(two_stage.predict(validation_x), dtype=float)
        clipped_prediction = np.maximum(raw_prediction, 0.0)
        results.append(
            CandidateResult(
                name="TWO_STAGE_RANDOM_FOREST",
                pipeline=two_stage,
                parameters={
                    "classifier": "RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=2)",
                    "regressor": "RandomForestRegressor(n_estimators=300, max_depth=16, min_samples_leaf=2)",
                    "positiveProbabilityThreshold": 0.5,
                },
                feature_columns=feature_columns,
                validation_prediction=pd.Series(clipped_prediction, index=validation_dataframe.index, name="TWO_STAGE_RANDOM_FOREST"),
                clipped_prediction_count=int((raw_prediction < 0).sum()),
            ),
        )
    return tuple(results)


def extract_feature_importances(candidate: CandidateResult) -> pd.DataFrame:
    """Return aggregated feature importances when the fitted estimator exposes them."""
    if not isinstance(candidate.pipeline, Pipeline):
        return pd.DataFrame(columns=["model", "feature", "importance"])
    model = candidate.pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["model", "feature", "importance"])
    preprocessor = candidate.pipeline.named_steps["preprocessor"]
    transformed_names = list(preprocessor.get_feature_names_out())
    importances = np.asarray(model.feature_importances_, dtype=float)
    if len(transformed_names) != len(importances):
        raise TrainingValidationError("Les importances ne correspondent pas aux features transformées.")
    rows: list[dict[str, object]] = []
    for transformed_name, importance in zip(transformed_names, importances, strict=True):
        feature_name = "productId" if transformed_name.startswith("productId_") else transformed_name
        rows.append({"model": candidate.name, "feature": feature_name, "importance": float(importance)})
    return (
        pd.DataFrame(rows)
        .groupby(["model", "feature"], as_index=False)["importance"]
        .sum()
        .sort_values(["model", "importance"], ascending=[True, False], kind="stable", ignore_index=True)
    )
