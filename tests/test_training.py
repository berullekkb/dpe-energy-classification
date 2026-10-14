
"""Tests de séparation des données DPE."""

import pandas as pd
import pytest

from dpe.training import (
    make_profile_groups,
    split_data,
    split_train_validation_test,
)


def sample_data():
    """Crée un petit jeu de données pour les tests."""

    X = pd.DataFrame({
        "chauffage": [
            "Gaz", "Gaz",
            "Bois", "Bois",
            "Electricite", "Electricite",
            "Fioul", "Fioul",
            "Pompe", "Pompe",
        ],
        "surface": [
            50, 50,
            80, 80,
            60, 60,
            100, 100,
            70, 70,
        ],
    })

    y = pd.Series([
        "Passable", "Energivore",
        "Performant", "Performant",
        "Passable", "Passable",
        "Energivore", "Energivore",
        "Performant", "Performant",
    ])

    return X, y


# ============================================================
# TESTS : SEPARATION TRAIN / TEST
# ============================================================

def test_identical_profiles_share_group():
    """Deux profils identiques doivent avoir le même groupe."""

    X, _ = sample_data()

    groups = make_profile_groups(X)

    assert groups[0] == groups[1]
    assert groups[2] == groups[3]
    assert groups[0] != groups[2]


def test_group_split_has_no_shared_profiles():
    """Aucun profil identique entre train et test."""

    X, y = sample_data()

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        strategy="group",
        test_size=0.4,
    )

    train_groups = set(make_profile_groups(X_train))
    test_groups = set(make_profile_groups(X_test))

    assert train_groups.isdisjoint(test_groups)

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)


def test_split_is_reproducible():
    """La même graine doit produire la même séparation."""

    X, y = sample_data()

    result_1 = split_data(X, y, strategy="group")
    result_2 = split_data(X, y, strategy="group")

    assert (
        result_1[0].index.tolist()
        == result_2[0].index.tolist()
    )

    assert (
        result_1[1].index.tolist()
        == result_2[1].index.tolist()
    )


def test_invalid_strategy_raises():
    """Une stratégie inconnue doit déclencher une erreur."""

    X, y = sample_data()

    with pytest.raises(ValueError):
        split_data(
            X,
            y,
            strategy="unknown",
        )


# ============================================================
# TESTS : SEPARATION TRAIN / VALIDATION / TEST
# ============================================================

def test_three_way_split_no_shared_profiles():
    """Aucun profil identique entre les trois ensembles."""

    X, y = sample_data()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_train_validation_test(
        X,
        y,
        test_size=0.2,
        validation_size=0.2,
    )

    train_groups = set(make_profile_groups(X_train))
    val_groups = set(make_profile_groups(X_val))
    test_groups = set(make_profile_groups(X_test))

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    assert (
        len(X_train) + len(X_val) + len(X_test)
        == len(X)
    )

    assert (
        len(y_train) + len(y_val) + len(y_test)
        == len(y)
    )


def test_three_way_split_is_reproducible():
    """La séparation en trois ensembles doit être reproductible."""

    X, y = sample_data()

    first = split_train_validation_test(X, y)
    second = split_train_validation_test(X, y)

    for first_part, second_part in zip(first, second):
        assert (
            first_part.index.tolist()
            == second_part.index.tolist()
        )


def test_three_way_split_preserves_original_test():
    """Le test final doit rester identique à celui de split_data."""

    X, y = sample_data()

    _, X_test_original, _, y_test_original = split_data(
        X,
        y,
        strategy="group",
    )

    (
        _,
        _,
        X_test_new,
        _,
        _,
        y_test_new,
    ) = split_train_validation_test(X, y)

    assert (
        X_test_original.index.tolist()
        == X_test_new.index.tolist()
    )

    assert (
        y_test_original.index.tolist()
        == y_test_new.index.tolist()
    )


def test_three_way_split_invalid_sizes():
    """La somme validation + test doit être inférieure à 1."""

    X, y = sample_data()

    with pytest.raises(ValueError):
        split_train_validation_test(
            X,
            y,
            test_size=0.8,
            validation_size=0.3,
        )
