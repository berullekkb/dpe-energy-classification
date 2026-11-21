"""Évaluation finale, unique, sur le jeu de TEST.

Le modèle évalué est celui figé dans results/metrics/selected_model.json par
compare_models.py (meilleur F1 macro en VALIDATION). Ce script ne compare
aucun modèle sur le test et ne modifie pas la sélection.

Deux variantes du modèle retenu, mêmes hyperparamètres :
- `train_plus_validation` (variante principale) : réentraîné sur
  TRAIN + VALIDATION, ce qui exploite toutes les données de développement ;
- `train_only` : le pipeline exactement évalué en validation, pour mesurer
  l'écart validation -> test sans changement de données d'entraînement.

Rappel : le test a déjà servi à une ancienne baseline (régression logistique,
results/metrics/logistic_regression.json). Il n'est donc pas historiquement
vierge, mais il n'a jamais servi à choisir un modèle ou un hyperparamètre.

Le script refuse de s'exécuter une seconde fois sans --force, pour éviter les
allers-retours entre le test et la modélisation.

Usage :
    python scripts/evaluate_final.py [--force]
"""

import argparse
from datetime import datetime, timezone

import joblib
import pandas as pd

from dpe.benchmarking import evaluate_pipeline, per_class_table, read_json, write_json
from dpe.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR
from dpe.data_loader import load_processed_data
from dpe.models import get_spec
from dpe.optimization import fit_pipeline
from dpe.preprocessing import split_features_target
from dpe.training import make_profile_groups, split_train_validation_test
from dpe.visualization import plot_confusion

OUTPUT = METRICS_DIR / "final_test_evaluation.json"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if OUTPUT.exists() and not args.force:
        raise SystemExit(
            f"{OUTPUT} existe déjà : le test a déjà été évalué. "
            "Utiliser --force uniquement pour reproduire le même calcul."
        )

    selection = read_json(METRICS_DIR / "selected_model.json")
    name = selection["selected_model"]
    spec = get_spec(name)
    params = read_json(METRICS_DIR / f"{name}_best_params.json")["params"]
    validation = read_json(METRICS_DIR / f"{name}_validation.json")

    df = load_processed_data()
    X, y = split_features_target(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_validation_test(X, y)

    overlap = set(make_profile_groups(pd.concat([X_train, X_val]))) & set(
        make_profile_groups(X_test))
    if overlap:
        raise RuntimeError("Profils communs entre développement et test.")

    print(f"Modèle retenu en validation : {spec.label} ({name})")
    print(f"Test : {len(X_test)} lignes, jamais utilisé pour la sélection.")

    results = {}

    train_only = joblib.load(MODELS_DIR / f"{name}.joblib")
    m, rep, conf, _ = evaluate_pipeline(train_only, X_test, y_test)
    results["train_only"] = {"fit_rows": len(X_train), "metrics": m,
                             "per_class": per_class_table(rep).to_dict(orient="index")}
    conf.to_csv(METRICS_DIR / "final_test_confusion_train_only.csv")

    X_dev = pd.concat([X_train, X_val])
    y_dev = pd.concat([y_train, y_val])
    final = fit_pipeline(spec, params, X_dev, y_dev)
    m, rep, conf, _ = evaluate_pipeline(final, X_test, y_test)
    results["train_plus_validation"] = {
        "fit_rows": len(X_dev), "metrics": m,
        "per_class": per_class_table(rep).to_dict(orient="index"),
    }
    conf.to_csv(METRICS_DIR / "final_test_confusion.csv")
    joblib.dump(final, MODELS_DIR / f"final_{name}.joblib")
    plot_confusion(conf, f"{spec.label} - TEST (réentraîné sur train + validation)",
                   FIGURES_DIR / "final_test_confusion.png")

    baseline = read_json(METRICS_DIR / "logistic_regression.json")
    payload = {
        "model": name,
        "label": spec.label,
        "params": params,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection": {k: selection[k] for k in
                      ("criterion", "validation_f1_macro", "test_used_for_selection")},
        "validation_metrics": validation["metrics"],
        "test": results,
        "test_size": len(X_test),
        "reference_baseline_test": {
            "model": "Régression logistique L2, one-hot, poids balanced "
                     "(baseline initiale, train/test par profils)",
            "metrics": baseline["metrics"],
        },
        "note": (
            "Le test avait déjà servi à la baseline logistique ; il n'a jamais "
            "servi à sélectionner un modèle ou un hyperparamètre."
        ),
    }
    write_json(OUTPUT, payload)

    for variant, res in results.items():
        mm = res["metrics"]
        print(f"{variant:<22} F1 macro={mm['f1_macro']:.4f} "
              f"balanced acc={mm['balanced_accuracy']:.4f} acc={mm['accuracy']:.4f}")
    print(pd.DataFrame(results["train_plus_validation"]["per_class"]).T.round(3))
    print(conf)


if __name__ == "__main__":
    main()
