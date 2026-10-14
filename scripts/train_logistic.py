"""Entraînement et évaluation de la régression logistique DPE."""

import json
from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression

from dpe.config import PROJECT_ROOT, RANDOM_STATE, GROUPS
from dpe.data_loader import load_processed_data
from dpe.preprocessing import (
    split_features_target,
    make_model_pipeline,
)
from dpe.training import split_data, make_profile_groups
from dpe.evaluation import evaluate_classifier


def main():
    print("\n=== 1. CHARGEMENT DES DONNEES ===")

    df = load_processed_data()
    X, y = split_features_target(df)

    print(f"Observations : {len(df)}")
    print(f"Variables explicatives : {X.shape[1]}")

    print("\n=== 2. SEPARATION TRAIN / TEST ===")

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        strategy="group",
    )

    print(f"Train : {len(X_train)}")
    print(f"Test  : {len(X_test)}")

    train_profiles = set(make_profile_groups(X_train))
    test_profiles = set(make_profile_groups(X_test))

    overlap = len(train_profiles & test_profiles)

    print(f"Profils communs : {overlap}")

    if overlap != 0:
        raise RuntimeError(
            "Fuite détectée : profils communs entre train et test."
        )

    print("\n=== 3. CREATION DU MODELE ===")

    estimator = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    pipeline = make_model_pipeline(
        X_train,
        estimator,
    )

    print("Modele : LogisticRegression")
    print("Ponderation : balanced")
    print("Pretraitement integre au pipeline")

    print("\n=== 4. ENTRAINEMENT ===")

    pipeline.fit(X_train, y_train)

    print("Entrainement termine.")

    print("\n=== 5. PREDICTIONS ===")

    y_pred = pipeline.predict(X_test)

    print(f"Predictions realisees : {len(y_pred)}")

    print("\n=== 6. EVALUATION ===")

    metrics, report, confusion_df = evaluate_classifier(
        y_test,
        y_pred,
    )

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\n=== RAPPORT PAR CLASSE ===")

    for label in GROUPS:
        values = report[label]

        print(
            f"{label}: "
            f"precision={values['precision']:.4f}, "
            f"recall={values['recall']:.4f}, "
            f"f1={values['f1-score']:.4f}, "
            f"support={int(values['support'])}"
        )

    print("\n=== MATRICE DE CONFUSION ===")
    print("Lignes = classes reelles")
    print("Colonnes = classes predites")
    print(confusion_df.to_string())

    print("\n=== 7. SAUVEGARDE DES RESULTATS ===")

    output_dir = PROJECT_ROOT / "results" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": "logistic_regression",
        "split_strategy": "group",
        "random_state": RANDOM_STATE,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "metrics": metrics,
        "classification_report": report,
    }

    metrics_path = output_dir / "logistic_regression.json"
    confusion_path = output_dir / "logistic_confusion_matrix.csv"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

    confusion_df.to_csv(
        confusion_path,
        encoding="utf-8-sig",
    )

    print(f"Metriques : {metrics_path}")
    print(f"Matrice   : {confusion_path}")


if __name__ == "__main__":
    main()