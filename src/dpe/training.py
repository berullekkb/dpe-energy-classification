"""Séparation reproductible des données DPE."""

import hashlib

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    GroupShuffleSplit,
    train_test_split,
)

from .config import RANDOM_STATE, TEST_SIZE, TARGET


def make_profile_groups(X: pd.DataFrame) -> np.ndarray:
    """
    Identifie les profils identiques à partir des variables
    explicatives, sans utiliser la cible.

    Deux lignes identiques reçoivent le même identifiant.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X doit être un DataFrame pandas.")

    if X.empty:
        raise ValueError("X ne peut pas être vide.")

    if TARGET in X.columns:
        raise ValueError(
            "La variable cible ne doit pas être utilisée "
            "pour identifier les profils."
        )

    if X.columns.duplicated().any():
        raise ValueError(
            "Les noms de colonnes doivent être uniques."
        )

    def fingerprint(row):
        values = tuple(
            (str(col), type(value).__name__, repr(value))
            for col, value in zip(X.columns, row)
        )

        payload = repr(values).encode("utf-8")

        return hashlib.sha256(payload).hexdigest()

    return np.array([
        fingerprint(row)
        for row in X.itertuples(index=False, name=None)
    ])


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    strategy: str = "group",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Sépare les données en train et test.

    strategy='group' :
        Les profils identiques restent dans un même ensemble.

    strategy='stratified' :
        Préserve approximativement la proportion des classes,
        sans garantir la séparation des profils identiques.
    """

    if len(X) != len(y):
        raise ValueError(
            "X et y doivent avoir le même nombre de lignes."
        )

    if len(X) < 2:
        raise ValueError(
            "Au moins deux observations sont nécessaires."
        )

    if strategy == "stratified":
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

    if strategy == "group":
        groups = make_profile_groups(X)

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )

        train_idx, test_idx = next(
            splitter.split(X, y, groups=groups)
        )

        return (
            X.iloc[train_idx].copy(),
            X.iloc[test_idx].copy(),
            y.iloc[train_idx].copy(),
            y.iloc[test_idx].copy(),
        )

    raise ValueError(
        "strategy doit être 'group' ou 'stratified'."
    )