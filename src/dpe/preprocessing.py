"""Préparation des variables pour les modèles DPE."""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from .config import (
    CONTINUOUS_FEATURES,
    RANDOM_STATE,
    TARGET,
    TARGET_ORDINAL_CODES,
)

ENCODINGS = ("onehot", "target")


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


class OrdinalTargetEncoder(BaseEstimator, TransformerMixin):
    """Encodage cible des notebooks historiques, sans dépendance externe.

    Reproduit ``category_encoders.TargetEncoder(smoothing, min_samples_leaf)``
    appliqué à la cible textuelle DPE : les classes sont converties en score
    ordinal (Energivore = 0, Passable = 1, Performant = 2) et chaque modalité
    est remplacée par la moyenne lissée de ce score ::

        lambda = 1 / (1 + exp(-(n - min_samples_leaf) / smoothing))
        code   = prior * (1 - lambda) + moyenne_modalite * lambda

    Une modalité vue une seule fois ou inconnue reçoit la moyenne globale.

    Amélioration optionnelle : ``cv`` (entier >= 2) active un encodage
    « hors pli » lors de ``fit_transform`` : chaque ligne d'entraînement est
    encodée avec des statistiques calculées sans elle, ce qui supprime le biais
    optimiste de l'encodage historique. ``transform`` utilise toujours les
    statistiques de tout le jeu d'entraînement.
    """

    def __init__(self, smoothing=3.0, min_samples_leaf=50, cv=None,
                 random_state=RANDOM_STATE):
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.cv = cv
        self.random_state = random_state

    @staticmethod
    def _codes(y):
        y = pd.Series(np.asarray(y))
        unknown = set(y.unique()) - set(TARGET_ORDINAL_CODES)
        if unknown:
            raise ValueError(f"Classes inattendues : {sorted(map(str, unknown))}")
        return y.map(TARGET_ORDINAL_CODES).astype(float).to_numpy()

    def _learn(self, X: pd.DataFrame, codes: np.ndarray):
        prior = float(codes.mean())
        mappings = []
        for col in X.columns:
            stats = (
                pd.DataFrame({"x": X[col].to_numpy(), "t": codes})
                .groupby("x", dropna=False)["t"]
                .agg(["count", "mean"])
            )
            smoove = 1.0 / (
                1.0 + np.exp(-(stats["count"] - self.min_samples_leaf) / self.smoothing)
            )
            encoded = prior * (1.0 - smoove) + stats["mean"] * smoove
            encoded[stats["count"] == 1] = prior
            mappings.append(encoded.to_dict())
        return prior, mappings

    @staticmethod
    def _apply(X: pd.DataFrame, prior, mappings) -> np.ndarray:
        out = np.empty(X.shape, dtype=float)
        for j, col in enumerate(X.columns):
            out[:, j] = (
                X[col].map(mappings[j]).astype(float).fillna(prior).to_numpy()
            )
        return out

    def _as_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.reset_index(drop=True)
        return pd.DataFrame(np.asarray(X, dtype=object))

    def fit(self, X, y):
        X = self._as_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.prior_, self.mappings_ = self._learn(X, self._codes(y))
        return self

    def transform(self, X):
        return self._apply(self._as_frame(X), self.prior_, self.mappings_)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        if not self.cv:
            return self.transform(X)

        X = self._as_frame(X)
        codes = self._codes(y)
        out = np.empty(X.shape, dtype=float)
        splitter = StratifiedKFold(
            n_splits=int(self.cv), shuffle=True, random_state=self.random_state
        )
        for fit_idx, enc_idx in splitter.split(X, codes):
            prior, mappings = self._learn(X.iloc[fit_idx], codes[fit_idx])
            out[enc_idx] = self._apply(X.iloc[enc_idx], prior, mappings)
        return out

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return self.feature_names_in_


def _column_groups(X: pd.DataFrame):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    unsupported = set(X.columns) - set(numeric) - set(categorical)
    if unsupported:
        raise ValueError(
            f"Types non pris en charge : {sorted(unsupported)}"
        )
    return numeric, categorical


def make_preprocessor(
    X: pd.DataFrame,
    encoding: str = "onehot",
    scale: bool = True,
    target_cv=None,
) -> ColumnTransformer:
    """Construit un prétraitement non encore entraîné.

    encoding='onehot' (défaut, comportement d'origine du package) :
        numériques -> imputation médiane + RobustScaler (si scale),
        qualitatives -> imputation 'Manquant' + OneHotEncoder.

    encoding='target' (protocole des notebooks historiques) :
        variables continues -> imputation médiane + RobustScaler (si scale),
        autres numériques (ordinales `_enc`, binaires) -> imputation médiane,
        qualitatives -> imputation 'Manquant' + OrdinalTargetEncoder
        (smoothing=3, min_samples_leaf=50).

    scale=False convient aux modèles à base d'arbres (insensibles à l'échelle).
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "X doit être un DataFrame pandas."
        )

    if encoding not in ENCODINGS:
        raise ValueError(f"encoding doit être parmi {ENCODINGS}.")

    numeric, categorical = _column_groups(X)

    def numeric_pipe(with_scaler):
        steps = [(
            "imputer",
            SimpleImputer(strategy="median", keep_empty_features=True),
        )]
        if with_scaler:
            steps.append(("scaler", RobustScaler()))
        return Pipeline(steps)

    transformers = []

    if encoding == "onehot":
        if numeric:
            transformers.append(("num", numeric_pipe(scale), numeric))
    else:
        continuous = [c for c in numeric if c in CONTINUOUS_FEATURES]
        others = [c for c in numeric if c not in CONTINUOUS_FEATURES]
        if continuous:
            transformers.append(("num", numeric_pipe(scale), continuous))
        if others:
            transformers.append(("pass", numeric_pipe(False), others))

    if categorical:
        if encoding == "onehot":
            encoder = OneHotEncoder(handle_unknown="ignore")
        else:
            encoder = OrdinalTargetEncoder(cv=target_cv)

        categorical_pipe = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="Manquant"),
            ),
            ("encoder", encoder),
        ])

        transformers.append(
            ("cat", categorical_pipe, categorical)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0,
    )


def make_model_pipeline(
    X: pd.DataFrame,
    estimator,
    encoding: str = "onehot",
    scale: bool = True,
    target_cv=None,
) -> Pipeline:
    """Associe le prétraitement à un modèle ML."""

    return Pipeline([
        ("pre", make_preprocessor(X, encoding, scale, target_cv)),
        ("clf", estimator)
    ])
