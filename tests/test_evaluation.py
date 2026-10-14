"""Tests du module d'évaluation DPE."""

import pytest

from dpe.evaluation import evaluate_classifier


def test_perfect_predictions():
    y_true = [
        "Performant",
        "Passable",
        "Energivore",
    ]

    y_pred = [
        "Performant",
        "Passable",
        "Energivore",
    ]

    metrics, report, matrix = evaluate_classifier(
        y_true,
        y_pred,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert matrix.to_numpy().trace() == 3
    assert report["Performant"]["recall"] == 1.0


def test_imperfect_predictions():
    y_true = [
        "Performant",
        "Passable",
        "Energivore",
    ]

    y_pred = [
        "Passable",
        "Passable",
        "Energivore",
    ]

    metrics, report, matrix = evaluate_classifier(
        y_true,
        y_pred,
    )

    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["f1_macro"] < 1.0
    assert report["Performant"]["recall"] == 0.0
    assert matrix.loc["Performant", "Passable"] == 1


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        evaluate_classifier(
            ["Passable", "Performant"],
            ["Passable"],
        )


def test_empty_predictions_raise():
    with pytest.raises(ValueError):
        evaluate_classifier([], [])