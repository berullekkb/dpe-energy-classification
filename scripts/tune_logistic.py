
"""Optimisation de la régression logistique pour la classification DPE.

Objectif :
- Entraîner plusieurs configurations de régression logistique.
- Comparer leurs performances sur le jeu de validation.
- Sélectionner la meilleure configuration selon le F1 macro.
- Ne jamais utiliser le jeu de test final pour la sélection.
"""

import json

import pandas as pd
from sklearn.linear_model import LogisticRegression

from dpe.config import PROJECT_ROOT, RANDOM_STATE
from dpe.data_loader import load_processed_data
from dpe.evaluation import evaluate_classifier
from dpe.preprocessing import (
    make_model_pipeline,
    split_features_target,
)
from dpe.training import split_train_validation_test


# ============================================================
# CONFIGURATION
# ============================================================

C_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)

OUTPUT_DIR = PROJECT_ROOT / "results" / "metrics"


# ============================================================
# COMPARAISON DES CONFIGURATIONS
# ============================================================

def compare_logistic_configurations(
    X_train,
    y_train,
    X_val,
    y_val,
):
    """Compare plusieurs valeurs de C sur le jeu de validation.

    Le prétraitement et le modèle sont ajustés uniquement
    sur les données d'entraînement.
    """

    results = []

    for c_value in C_VALUES:

        print(f"\nEntraînement avec C = {c_value}")

        # 1. Définir le modèle.
        estimator = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE,
        )

        # 2. Construire le pipeline de prétraitement.
        pipeline = make_model_pipeline(
            X_train,
            estimator,
        )

        # 3. Entraîner uniquement sur TRAIN.
        pipeline.fit(
            X_train,
            y_train,
        )

        # 4. Prédire uniquement sur VALIDATION.
        predictions = pipeline.predict(X_val)

        # 5. Récupérer les métriques.
        #
        # evaluate_classifier() renvoie :
        # (metrics, report, confusion_df)
        #
        # Nous utilisons uniquement metrics ici.

        metrics, _, _ = evaluate_classifier(
            y_val,
            predictions,
        )

        # 6. Enregistrer les performances.
        result = {
            "C": float(c_value),
            "accuracy": float(
                metrics["accuracy"]
            ),
            "balanced_accuracy": float(
                metrics["balanced_accuracy"]
            ),
            "f1_macro": float(
                metrics["f1_macro"]
            ),
            "f1_weighted": float(
                metrics["f1_weighted"]
            ),
        }

        results.append(result)

        print(
            f"Accuracy = {result['accuracy']:.4f} | "
            f"Balanced accuracy = "
            f"{result['balanced_accuracy']:.4f} | "
            f"F1 macro = {result['f1_macro']:.4f}"
        )

    # 7. Transformer les résultats en tableau.
    results_df = pd.DataFrame(results)

    # 8. Trier par F1 macro décroissant.
    # En cas d'égalité, préférer la plus petite valeur de C.
    results_df = results_df.sort_values(
        by=["f1_macro", "C"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return results_df


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=== CHARGEMENT DES DONNEES ===")

    # 1. Charger les données préparées.
    df = load_processed_data()

    # 2. Séparer variables explicatives et cible.
    X, y = split_features_target(df)

    # 3. Créer les trois ensembles.
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
    )

    print(f"Train      : {len(X_train)}")
    print(f"Validation : {len(X_val)}")
    print(f"Test final : {len(X_test)} (non utilisé)")

    # 4. Comparer les configurations.
    print("\n=== COMPARAISON DES CONFIGURATIONS ===")

    results_df = compare_logistic_configurations(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    # 5. Afficher le classement.
    print("\n=== CLASSEMENT SUR VALIDATION ===")

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # 6. Sélectionner la meilleure configuration.
    best = results_df.iloc[0]

    print("\n=== MEILLEURE CONFIGURATION ===")
    print(f"C = {best['C']}")
    print(
        f"F1 macro validation = "
        f"{best['f1_macro']:.4f}"
    )
    print(
        f"Balanced accuracy validation = "
        f"{best['balanced_accuracy']:.4f}"
    )

    # 7. Préparer le dossier des résultats.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        OUTPUT_DIR / "logistic_validation_comparison.csv"
    )

    json_path = (
        OUTPUT_DIR / "logistic_best_params.json"
    )

    # 8. Sauvegarder toutes les configurations.
    results_df.to_csv(
        csv_path,
        index=False,
    )

    # 9. Sauvegarder les meilleurs paramètres.
    best_config = {
        "model": "LogisticRegression",
        "selection_metric": "f1_macro",
        "selected_on": "validation",
        "best_C": float(best["C"]),
        "class_weight": "balanced",
        "max_iter": 3000,
        "random_state": RANDOM_STATE,
        "train_size": len(X_train),
        "validation_size": len(X_val),
        "test_size": len(X_test),
        "validation_accuracy": float(
            best["accuracy"]
        ),
        "validation_balanced_accuracy": float(
            best["balanced_accuracy"]
        ),
        "validation_f1_macro": float(
            best["f1_macro"]
        ),
        "validation_f1_weighted": float(
            best["f1_weighted"]
        ),
        "test_evaluated": False,
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            best_config,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # 10. Confirmer la sauvegarde.
    print("\n=== FICHIERS GENERES ===")
    print(csv_path)
    print(json_path)

    print("\n=== CONTROLE METHODOLOGIQUE ===")
    print("Entraînement : TRAIN uniquement")
    print("Sélection    : VALIDATION uniquement")
    print("Test final   : NON UTILISE")

    print("\nOptimisation terminée.")


if __name__ == "__main__":
    main()
