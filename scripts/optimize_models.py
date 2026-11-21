"""Optimisation Optuna puis évaluation sur la validation de plusieurs modèles.

Exemples (PowerShell, depuis la racine du projet) :

    $env:PYTHONPATH = (Resolve-Path .\\src).Path
    python scripts/optimize_models.py --family tree
    python scripts/optimize_models.py --models svm_rbf --n-trials 25
    python scripts/optimize_models.py --family boosting --n-trials 25

Pour chaque modèle :
1. Optuna sur TRAIN uniquement (StratifiedGroupKFold par profils, F1 macro) ;
2. réentraînement du meilleur pipeline sur tout TRAIN ;
3. évaluation sur VALIDATION ;
4. sauvegarde : results/metrics/<modele>_{best_params,validation}.json,
   <modele>_trials.csv, <modele>_validation_confusion.csv et le modèle
   entraîné dans results/models/ (non versionné).

Le jeu de TEST n'est jamais chargé dans les calculs de ce script.
"""

import argparse
import time

import joblib

from dpe.benchmarking import (
    evaluate_pipeline,
    per_class_table,
    save_validation_results,
)
from dpe.config import METRICS_DIR, MODELS_DIR, RANDOM_STATE
from dpe.data_loader import load_processed_data
from dpe.models import get_spec, resolve_models
from dpe.optimization import fit_pipeline, optimize_model, trials_dataframe
from dpe.preprocessing import split_features_target
from dpe.training import (
    make_profile_groups,
    split_train_validation_test,
    stratified_subsample,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--family", nargs="*", default=None,
                        help="linear, svm, tree, boosting")
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=None,
                        help="Durée maximale d'optimisation par modèle (s).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Ignore les modèles déjà évalués.")
    return parser.parse_args(argv)


def load_splits():
    df = load_processed_data()
    X, y = split_features_target(df)
    X_train, X_val, X_test, y_train, y_val, y_test = (
        split_train_validation_test(X, y)
    )
    sizes = {
        "train_size": len(X_train),
        "validation_size": len(X_val),
        "test_size": len(X_test),
    }
    return X_train, y_train, X_val, y_val, sizes


def run_model(name, X_train, y_train, X_val, y_val, sizes, args):
    spec = get_spec(name)
    if not spec.is_available():
        print(f"[{name}] ignoré : dépendance manquante {spec.requires}")
        return None

    print(f"\n=== {spec.label} ({name}, origine : {spec.origin}) ===")

    X_tune, y_tune = stratified_subsample(
        X_train, y_train, spec.tuning_sample, random_state=RANDOM_STATE
    )
    groups = make_profile_groups(X_tune)
    print(f"Optimisation sur {len(X_tune)} lignes de TRAIN, "
          f"{len(set(groups))} profils, {args.n_splits} plis par groupes")

    t0 = time.time()
    study = optimize_model(
        spec, X_tune, y_tune, groups, n_trials=args.n_trials,
        n_splits=args.n_splits, timeout=args.timeout,
    )
    tuning_seconds = time.time() - t0

    trials = trials_dataframe(study)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    trials.to_csv(METRICS_DIR / f"{name}_trials.csv", index=False)

    best = study.best_trial
    params = dict(best.params)
    completed = int((trials["state"] == "COMPLETE").sum())
    cv_summary = {
        "f1_macro_cv_mean": float(best.value),
        "f1_macro_cv_std": float(best.user_attrs.get("f1_macro_std", float("nan"))),
        "f1_macro_cv_folds": best.user_attrs.get("f1_macro_folds"),
        "n_trials": len(trials),
        "n_trials_complete": completed,
        "n_splits": args.n_splits,
        "tuning_rows": len(X_tune),
        "tuning_seconds": round(tuning_seconds, 1),
    }

    t0 = time.time()
    pipeline = fit_pipeline(spec, params, X_train, y_train)
    fit_seconds = time.time() - t0

    metrics, report, confusion, _ = evaluate_pipeline(pipeline, X_val, y_val)

    save_validation_results(
        name, params, cv_summary, metrics, report, confusion, sizes,
        extra={
            "fit_seconds": round(fit_seconds, 2),
            "final_fit_rows": len(X_train),
            "notes": spec.notes,
        },
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / f"{name}.joblib")

    print(f"Meilleurs paramètres : {params}")
    print(f"F1 macro CV (train)  : {best.value:.4f}")
    print("Validation : " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    print(per_class_table(report).round(3).to_string())
    print(confusion.to_string())
    return metrics


def main(argv=None):
    args = parse_args(argv)
    names = resolve_models(args.models, args.family)
    X_train, y_train, X_val, y_val, sizes = load_splits()
    print(f"Train : {sizes['train_size']} | Validation : {sizes['validation_size']}"
          f" | Test : {sizes['test_size']} (non utilisé)")

    for name in names:
        if args.skip_existing and (METRICS_DIR / f"{name}_validation.json").exists():
            print(f"[{name}] déjà évalué, ignoré.")
            continue
        run_model(name, X_train, y_train, X_val, y_val, sizes, args)


if __name__ == "__main__":
    main()
