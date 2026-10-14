
"""Tests de l'optimisation de la régression logistique."""

import pandas as pd
import pytest

from scripts import tune_logistic


def sample_data():
    """Crée des données simples avec trois classes DPE."""

    X_train = pd.DataFrame({
        "surface": [35, 40, 45, 65, 70, 75, 95, 100, 105],
        "chauffage": [
            "Electrique", "Electrique", "Electrique",
            "Gaz", "Gaz", "Gaz",
            "Fioul", "Fioul", "Fioul",
        ],
    })

    y_train = pd.Series([
        "Performant", "Performant", "Performant",
        "Passable", "Passable", "Passable",
        "Energivore", "Energivore", "Energivore",
    ])

    X_val = pd.DataFrame({
        "surface": [38, 42, 68, 72, 98, 102],
        "chauffage": [
            "Electrique", "Electrique",
            "Gaz", "Gaz",
            "Fioul", "Fioul",
        ],
    })

    y_val = pd.Series([
        "Performant", "Performant",
        "Passable", "Passable",
        "Energivore", "Energivore",
    ])

    return X_train, y_train, X_val, y_val


def test_tuning_evaluates_all_configurations(monkeypatch):
    """Chaque valeur de C doit être évaluée."""

    monkeypatch.setattr(
        tune_logistic,
        "C_VALUES",
        (0.1, 1.0, 10.0),
    )

    X_train, y_train, X_val, y_val = sample_data()

    results = tune_logistic.compare_logistic_configurations(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    assert len(results) == 3
    assert set(results["C"]) == {0.1, 1.0, 10.0}


def test_tuning_sorts_by_f1_macro(monkeypatch):
    """Le meilleur F1 macro doit apparaître en premier."""

    monkeypatch.setattr(
        tune_logistic,
        "C_VALUES",
        (0.1, 1.0, 10.0),
    )

    X_train, y_train, X_val, y_val = sample_data()

    results = tune_logistic.compare_logistic_configurations(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    scores = results["f1_macro"].tolist()

    assert scores == sorted(scores, reverse=True)


def test_tuning_metrics_are_valid(monkeypatch):
    """Les métriques doivent être comprises entre 0 et 1."""

    monkeypatch.setattr(
        tune_logistic,
        "C_VALUES",
        (1.0,),
    )

    X_train, y_train, X_val, y_val = sample_data()

    results = tune_logistic.compare_logistic_configurations(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
    ]:
        assert results[metric].between(0, 1).all()


def test_tuning_does_not_modify_input_data(monkeypatch):
    """L'optimisation ne doit pas modifier les données reçues."""

    monkeypatch.setattr(
        tune_logistic,
        "C_VALUES",
        (1.0,),
    )

    X_train, y_train, X_val, y_val = sample_data()

    original_X_train = X_train.copy(deep=True)
    original_y_train = y_train.copy(deep=True)
    original_X_val = X_val.copy(deep=True)
    original_y_val = y_val.copy(deep=True)

    tune_logistic.compare_logistic_configurations(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    pd.testing.assert_frame_equal(X_train, original_X_train)
    pd.testing.assert_series_equal(y_train, original_y_train)
    pd.testing.assert_frame_equal(X_val, original_X_val)
    pd.testing.assert_series_equal(y_val, original_y_val)
