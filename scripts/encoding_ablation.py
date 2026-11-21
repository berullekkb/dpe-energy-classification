"""Nouvelle expérience : effet de l'encodage des variables qualitatives.

Expérience diagnostique, distincte des expériences historiques et lancée
après la sélection du modèle (elle ne modifie pas selected_model.json).

Pour quelques modèles, à hyperparamètres fixés (meilleurs paramètres Optuna) :
- `target_historique` : TargetEncoder des notebooks (statistiques calculées
  sur les lignes mêmes qu'elles encodent) ;
- `target_hors_pli` : même encodage, mais chaque ligne du train est encodée
  avec des statistiques calculées sans elle (5 plis) ;
- `onehot` : encodage one-hot (prétraitement d'origine du package).

Mesures : F1 macro en CV par groupes sur TRAIN (5 plis) et sur VALIDATION.

Sortie : results/metrics/encoding_ablation.csv
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score

from dpe.benchmarking import read_json
from dpe.config import METRICS_DIR
from dpe.models import get_spec, make_sample_weight, split_model_params
from dpe.preprocessing import make_model_pipeline
from dpe.training import make_group_cv, make_profile_groups

import optimize_models as om

MODELS = ("random_forest", "gradient_boosting", "lightgbm", "lr_lasso")
VARIANTS = {
    "target_historique": {"encoding": "target", "target_cv": None},
    "target_hors_pli": {"encoding": "target", "target_cv": 5},
    "onehot": {"encoding": "onehot", "target_cv": None},
}


def fit(spec, params, variant, X, y):
    _, class_weight = split_model_params(params)
    pipe = make_model_pipeline(X, spec.make_estimator(params), scale=spec.scale,
                               **variant)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipe.fit(X, y, clf__sample_weight=make_sample_weight(y, class_weight))
    return pipe


def main():
    X_train, y_train, X_val, y_val, _ = om.load_splits()
    groups = make_profile_groups(X_train)
    cv = make_group_cv(n_splits=5)
    folds = list(cv.split(X_train, y_train, groups))

    rows = []
    for name in MODELS:
        spec = get_spec(name)
        if not spec.is_available():
            continue
        params = read_json(METRICS_DIR / f"{name}_best_params.json")["params"]
        for variant_name, variant in VARIANTS.items():
            scores = []
            for tr, va in folds:
                pipe = fit(spec, params, variant, X_train.iloc[tr], y_train.iloc[tr])
                scores.append(f1_score(y_train.iloc[va],
                                       pipe.predict(X_train.iloc[va]),
                                       average="macro"))
            pipe = fit(spec, params, variant, X_train, y_train)
            val = f1_score(y_val, pipe.predict(X_val), average="macro")
            rows.append({
                "model": name, "label": spec.label, "encodage": variant_name,
                "f1_macro_cv": float(np.mean(scores)),
                "f1_macro_cv_std": float(np.std(scores)),
                "f1_macro_validation": float(val),
            })
            print(f"{spec.label:<20} {variant_name:<18} CV={np.mean(scores):.4f} "
                  f"val={val:.4f}", flush=True)

    table = pd.DataFrame(rows)
    table["origine"] = "new"
    table.to_csv(METRICS_DIR / "encoding_ablation.csv", index=False)


if __name__ == "__main__":
    main()
