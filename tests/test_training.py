"""Tests de séparation des données DPE."""

import pandas as pd
import pytest

from dpe.training import (
    make_profile_groups,
    split_data,
)


def sample_data():
    X = pd.DataFrame({
        "chauffage": [
            "Gaz", "Gaz", "Bois", "Bois",
            "Electricite", "Electricite",
            "Fioul", "Fioul",
            "Pompe", "Pompe",
        ],
        "surface": [
            50, 50, 80, 80, 60, 60,
            100, 100, 70, 70,
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


def test_identical_profiles_share_group():
    X, _ = sample_data()

    groups = make_profile_groups(X)

    assert groups[0] == groups[1]
    assert groups[2] == groups[3]
    assert groups[0] != groups[2]


def test_group_split_has_no_shared_profiles():
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
    X, y = sample_data()

    result_1 = split_data(X, y, strategy="group")
    result_2 = split_data(X, y, strategy="group")

    assert result_1[0].index.tolist() == result_2[0].index.tolist()
    assert result_1[1].index.tolist() == result_2[1].index.tolist()


def test_invalid_strategy_raises():
    X, y = sample_data()

    with pytest.raises(ValueError):
        split_data(X, y, strategy="unknown")