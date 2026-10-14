"""Évaluation commune des modèles de classification DPE."""

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
)

from .config import GROUPS


def evaluate_classifier(y_true, y_pred):
    """Calcule les métriques d'un classifieur DPE."""

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true et y_pred doivent avoir la même longueur."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Impossible d'évaluer des prédictions vides."
        )

    labels = list(GROUPS)

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    return metrics, report, confusion_df