"""Optimisation des hyperparamètres par Optuna avec validation croisée par groupes.

Protocole commun à tous les modèles :

- uniquement les données d'entraînement (jamais la validation ni le test) ;
- ``StratifiedGroupKFold`` : les profils identiques restent dans le même pli ;
- le prétraitement est ajusté dans chaque pli (aucune fuite vers le pli
  d'évaluation) ;
- score optimisé : F1 macro moyen sur les plis ;
- échantillonneur TPE à graine fixe, élagage des essais peu prometteurs
  (``MedianPruner``) pour limiter le temps de calcul.
"""

import time
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score

from .config import RANDOM_STATE
from .models import make_sample_weight, split_model_params
from .preprocessing import make_model_pipeline
from .training import make_group_cv


def fit_pipeline(spec, params: dict, X: pd.DataFrame, y: pd.Series):
    """Construit et entraîne le pipeline (prétraitement + modèle) d'une spec.

    Les poids de classe de ``params`` sont appliqués via ``sample_weight``.
    """
    _, class_weight = split_model_params(params)
    class_weight = params.get("class_weight", class_weight)
    pipeline = make_model_pipeline(
        X, spec.make_estimator(params), encoding=spec.encoding, scale=spec.scale,
    )
    sample_weight = make_sample_weight(y, class_weight)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipeline.fit(X, y, clf__sample_weight=sample_weight)
    return pipeline


def cross_validate_f1(spec, params, X, y, groups, n_splits=5, trial=None,
                      random_state=RANDOM_STATE):
    """F1 macro par pli, CV stratifiée par groupes de profils."""
    cv = make_group_cv(n_splits=n_splits, random_state=random_state)
    scores = []
    for step, (tr, va) in enumerate(cv.split(X, y, groups)):
        pipeline = fit_pipeline(spec, params, X.iloc[tr], y.iloc[tr])
        pred = pipeline.predict(X.iloc[va])
        scores.append(f1_score(y.iloc[va], pred, average="macro"))
        if trial is not None:
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return np.asarray(scores)


def optimize_model(spec, X, y, groups, n_trials=25, n_splits=5, timeout=None,
                   random_state=RANDOM_STATE, verbose=True):
    """Lance une étude Optuna pour une spec et renvoie l'étude terminée."""

    def objective(trial):
        params = spec.suggest(trial)
        scores = cross_validate_f1(
            spec, params, X, y, groups, n_splits=n_splits, trial=trial,
            random_state=random_state,
        )
        trial.set_user_attr("f1_macro_std", float(scores.std()))
        trial.set_user_attr("f1_macro_folds", [float(s) for s in scores])
        return float(scores.mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        study_name=f"dpe_{spec.name}",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    start = time.time()

    def log(study, trial):
        if not verbose:
            return
        state = trial.state.name
        value = f"{trial.value:.4f}" if trial.value is not None else "  -   "
        best = study.best_value if study.best_trial else float("nan")
        print(
            f"[{spec.name}] essai {trial.number + 1:>3}/{n_trials} {state:<8} "
            f"f1={value} meilleur={best:.4f} "
            f"({(time.time() - start) / 60:.1f} min)",
            flush=True,
        )

    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[log])
    return study


def trials_dataframe(study) -> pd.DataFrame:
    """Tableau des essais (paramètres, score, durée, état)."""
    df = study.trials_dataframe(
        attrs=("number", "value", "duration", "params", "state", "user_attrs")
    )
    if "duration" in df:
        df["duration_s"] = df["duration"].dt.total_seconds()
        df = df.drop(columns="duration")
    return df.rename(columns={"value": "f1_macro_cv"})
