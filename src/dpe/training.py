
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


def split_train_validation_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    validation_size: float = 0.20,
    random_state: int = RANDOM_STATE,
):
    """
    Sépare les données en train, validation et test.

    Garanties :
    - Aucun profil identique entre les trois ensembles.
    - Le test final conserve la séparation de split_data().
    - La séparation est reproductible.

    validation_size représente une proportion du jeu complet.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X doit être un DataFrame pandas.")

    if not isinstance(y, pd.Series):
        raise TypeError("y doit être une Series pandas.")

    if len(X) != len(y):
        raise ValueError(
            "X et y doivent avoir le même nombre de lignes."
        )

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size doit être entre 0 et 1."
        )

    if not 0 < validation_size < 1:
        raise ValueError(
            "validation_size doit être entre 0 et 1."
        )

    if test_size + validation_size >= 1:
        raise ValueError(
            "La somme test_size + validation_size "
            "doit être inférieure à 1."
        )

    # 1. Conserver le même test final que précédemment.
    X_dev, X_test, y_dev, y_test = split_data(
        X,
        y,
        strategy="group",
        test_size=test_size,
        random_state=random_state,
    )

    # 2. Calculer la part de validation dans le développement.
    relative_validation_size = (
        validation_size / (1 - test_size)
    )

    # 3. Séparer train et validation par profils.
    X_train, X_val, y_train, y_val = split_data(
        X_dev,
        y_dev,
        strategy="group",
        test_size=relative_validation_size,
        random_state=random_state + 1,
    )

    # 4. Vérifier l'absence de profils communs.
    train_groups = set(make_profile_groups(X_train))
    val_groups = set(make_profile_groups(X_val))
    test_groups = set(make_profile_groups(X_test))

    if train_groups & val_groups:
        raise RuntimeError(
            "Profils communs entre train et validation."
        )

    if train_groups & test_groups:
        raise RuntimeError(
            "Profils communs entre train et test."
        )

    if val_groups & test_groups:
        raise RuntimeError(
            "Profils communs entre validation et test."
        )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )
