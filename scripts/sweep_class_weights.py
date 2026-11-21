"""Effet du poids de la classe Performant sur le Random Forest retenu.

Tous les hyperparamètres optimisés sont conservés ; seul le poids de la
classe Performant varie de 1 à 25 (espace de recherche des notebooks).
Chaque modèle est entraîné sur TRAIN et évalué sur VALIDATION.

Sortie : results/metrics/class_weight_sweep.csv
"""

import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from dpe.benchmarking import read_json
from dpe.config import GROUPS, METRICS_DIR
from dpe.models import get_spec
from dpe.optimization import fit_pipeline

import optimize_models as om

MODEL = "random_forest"


def main():
    X_train, y_train, X_val, y_val, _ = om.load_splits()
    spec = get_spec(MODEL)
    params = read_json(METRICS_DIR / f"{MODEL}_best_params.json")["params"]

    rows = []
    for weight in range(1, 26):
        pred = fit_pipeline(spec, {**params, "weight_Performant": weight},
                            X_train, y_train).predict(X_val)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, pred, labels=list(GROUPS), zero_division=0)
        row = {"model": MODEL, "weight_Performant": weight,
               "weight_Energivore": params["weight_Energivore"],
               "f1_macro": f1_score(y_val, pred, average="macro")}
        for i, label in enumerate(GROUPS):
            row[f"{label}_precision"] = precision[i]
            row[f"{label}_recall"] = recall[i]
            row[f"{label}_f1"] = f1[i]
        rows.append(row)
        print(f"poids {weight:>2} : rappel Performant={recall[0]:.3f} "
              f"précision={precision[0]:.3f} F1 macro={row['f1_macro']:.4f}",
              flush=True)

    pd.DataFrame(rows).to_csv(METRICS_DIR / "class_weight_sweep.csv", index=False)


if __name__ == "__main__":
    main()
