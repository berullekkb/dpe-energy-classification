"""Préparation des variables pour les modèles DPE."""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from .config import TARGET


def split_features_target(df: pd.DataFrame):
    """Sépare les variables explicatives de la cible."""

    if TARGET not in df.columns:
        raise ValueError(
            f"Colonne cible absente : {TARGET}"
        )

    X = df.drop(columns=[TARGET]).copy()
    y = df[TARGET].copy()

    if X.empty or X.shape[1] == 0:
        raise ValueError(
            "Aucune variable explicative disponible."
        )

    return X, y


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Construit un prétraitement non encore entraîné."""

    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "X doit être un DataFrame pandas."
        )

    numeric = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    unsupported = (
        set(X.columns)
        - set(numeric)
        - set(categorical)
    )

    if unsupported:
        raise ValueError(
            f"Types non pris en charge : {sorted(unsupported)}"
        )

    transformers = []

    if numeric:
        numeric_pipe = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True
                )
            ),
            ("scaler", RobustScaler())
        ])

        transformers.append(
            ("num", numeric_pipe, numeric)
        )

    if categorical:
        categorical_pipe = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Manquant"
                )
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ])

        transformers.append(
            ("cat", categorical_pipe, categorical)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0
    )


def make_model_pipeline(
    X: pd.DataFrame,
    estimator
) -> Pipeline:
    """Associe le prétraitement à un modèle ML."""

    return Pipeline([
        ("pre", make_preprocessor(X)),
        ("clf", estimator)
    ])