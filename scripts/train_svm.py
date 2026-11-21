"""Reprise du notebook 04_svm.ipynb avec le protocole commun.

Étapes (toutes sur TRAIN, évaluation sur VALIDATION, TEST jamais utilisé) :

1. Sous-échantillon stratifié de 8 000 lignes du train (comme le notebook).
2. Criblage des noyaux linéaire, polynomial (degré 3) et RBF avec C = 1 et
   poids 'balanced', en CV par groupes de profils (5 plis).
3. Optuna sur le noyau RBF (C, gamma, poids de classe ; espaces du notebook).
4. Évaluation sur la validation de deux variantes :
   - historique : modèle final ajusté sur les 8 000 lignes seulement ;
   - amélioration : même configuration ajustée sur tout le train.
5. Diagnostics : heatmap C x gamma (CV par groupes), vecteurs de support,
   frontière de décision dans le plan des 2 premières composantes principales.

Usage :
    python scripts/train_svm.py [--n-trials 25] [--skip-heatmap]
"""

import argparse
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from dpe.benchmarking import (
    evaluate_pipeline,
    per_class_table,
    save_validation_results,
    write_json,
)
from dpe.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, RANDOM_STATE
from dpe.models import SVM_KERNEL_SCREENING, get_spec, split_model_params
from dpe.optimization import (
    cross_validate_f1,
    fit_pipeline,
    optimize_model,
    trials_dataframe,
)
from dpe.training import make_profile_groups, stratified_subsample
from dpe.visualization import (
    plot_bar_scores,
    plot_confusion,
    plot_decision_regions,
    plot_heatmap,
)

import optimize_models as om

SAMPLE_SIZE = 8000
C_GRID = np.logspace(-1, 2, 6)
GAMMA_GRID = np.logspace(-3, 1, 6)


def kernel_screening(spec, X, y, groups, n_splits):
    rows = []
    for label, params in SVM_KERNEL_SCREENING.items():
        p = {**params, "class_weight": "balanced"}
        t0 = time.time()
        scores = cross_validate_f1(spec, p, X, y, groups, n_splits=n_splits)
        rows.append({
            "kernel": label,
            "f1_macro_cv_mean": scores.mean(),
            "f1_macro_cv_std": scores.std(),
            "seconds": round(time.time() - t0, 1),
        })
        print(f"  {label:<22} F1 = {scores.mean():.4f} ± {scores.std():.4f}")
    return pd.DataFrame(rows)


def heatmap_c_gamma(spec, params, X, y, groups, n_splits):
    values = np.zeros((len(GAMMA_GRID), len(C_GRID)))
    for i, gamma in enumerate(GAMMA_GRID):
        for j, c in enumerate(C_GRID):
            p = {**params, "C": float(c), "gamma": float(gamma)}
            values[i, j] = cross_validate_f1(
                spec, p, X, y, groups, n_splits=n_splits
            ).mean()
        print(f"  gamma={gamma:.3g} : " + " ".join(f"{v:.3f}" for v in values[i]))
    return values


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--skip-heatmap", action="store_true")
    args = parser.parse_args(argv)

    spec = get_spec("svm_rbf")
    X_train, y_train, X_val, y_val, sizes = om.load_splits()
    X_s, y_s = stratified_subsample(X_train, y_train, SAMPLE_SIZE, RANDOM_STATE)
    groups = make_profile_groups(X_s)
    print(f"Sous-échantillon : {len(X_s)} lignes, {len(set(groups))} profils")

    # 1. Criblage des noyaux
    print("\n=== Criblage des noyaux (C = 1, poids balanced) ===")
    screening = kernel_screening(spec, X_s, y_s, groups, args.n_splits)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    screening.to_csv(METRICS_DIR / "svm_kernel_screening.csv", index=False)
    plot_bar_scores(screening, "kernel", "f1_macro_cv_mean",
                    "SVM : criblage des noyaux (CV par groupes, 8 000 lignes)",
                    FIGURES_DIR / "svm_kernel_screening.png",
                    err_col="f1_macro_cv_std")

    # 2. Optuna RBF
    print("\n=== Optuna : SVM RBF ===")
    t0 = time.time()
    study = optimize_model(spec, X_s, y_s, groups, n_trials=args.n_trials,
                           n_splits=args.n_splits)
    tuning_seconds = time.time() - t0
    trials = trials_dataframe(study)
    trials.to_csv(METRICS_DIR / "svm_rbf_trials.csv", index=False)
    params = dict(study.best_trial.params)
    cv_summary = {
        "f1_macro_cv_mean": float(study.best_value),
        "f1_macro_cv_std": float(study.best_trial.user_attrs["f1_macro_std"]),
        "f1_macro_cv_folds": study.best_trial.user_attrs["f1_macro_folds"],
        "n_trials": len(trials),
        "n_trials_complete": int((trials["state"] == "COMPLETE").sum()),
        "n_splits": args.n_splits,
        "tuning_rows": len(X_s),
        "tuning_seconds": round(tuning_seconds, 1),
    }
    print(f"Meilleurs paramètres : {params} (F1 CV {study.best_value:.4f})")

    # 3. Variantes d'entraînement final
    variants = {}
    for variant, (Xf, yf) in {
        "historical_8000": (X_s, y_s),
        "full_train": (X_train, y_train),
    }.items():
        t0 = time.time()
        pipe = fit_pipeline(spec, params, Xf, yf)
        fit_s = time.time() - t0
        metrics, report, confusion, _ = evaluate_pipeline(pipe, X_val, y_val)
        n_sv = pipe.named_steps["clf"].n_support_
        variants[variant] = {
            "fit_rows": len(Xf),
            "fit_seconds": round(fit_s, 1),
            "metrics": metrics,
            "per_class": per_class_table(report).to_dict(orient="index"),
            "n_support_total": int(n_sv.sum()),
            "n_support_per_class": dict(zip(
                pipe.named_steps["clf"].classes_, map(int, n_sv))),
            "support_ratio": float(n_sv.sum() / len(Xf)),
        }
        print(f"  {variant:<16} ({len(Xf)} lignes, {fit_s:.1f}s) : "
              f"F1 val = {metrics['f1_macro']:.4f}, "
              f"vecteurs de support = {n_sv.sum()} ({n_sv.sum() / len(Xf):.1%})")
        if variant == "full_train":
            final_pipe, final = pipe, (metrics, report, confusion, fit_s)

    metrics, report, confusion, fit_s = final
    save_validation_results(
        "svm_rbf", params, cv_summary, metrics, report, confusion, sizes,
        extra={
            "fit_seconds": round(fit_s, 1),
            "final_fit_rows": len(X_train),
            "notes": "Optuna sur 8 000 lignes ; modèle final sur tout le train.",
            "variants": variants,
            "kernel_screening": screening.to_dict(orient="records"),
        },
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipe, MODELS_DIR / "svm_rbf.joblib")
    plot_confusion(confusion, "SVM RBF - validation (modèle ajusté sur tout le train)",
                   FIGURES_DIR / "svm_rbf_validation_confusion.png")

    # 4. Heatmap C x gamma
    if not args.skip_heatmap:
        print("\n=== Heatmap C x gamma ===")
        values = heatmap_c_gamma(spec, params, X_s, y_s, groups, args.n_splits)
        pd.DataFrame(values, index=[f"{g:.3g}" for g in GAMMA_GRID],
                     columns=[f"{c:.3g}" for c in C_GRID]).to_csv(
            METRICS_DIR / "svm_c_gamma_heatmap.csv")
        plot_heatmap(values, [f"{c:.3g}" for c in C_GRID],
                     [f"{g:.3g}" for g in GAMMA_GRID], "C", "gamma",
                     "SVM RBF : F1 macro selon C et gamma (poids optimaux)",
                     FIGURES_DIR / "svm_c_gamma_heatmap.png")

    # 5. Frontière de décision dans le plan ACP (illustration, comme le notebook)
    pre = final_pipe.named_steps["pre"]
    Z_s = pre.transform(X_s)
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(Z_s)
    P_s = pca.transform(Z_s)
    _, class_weight = split_model_params(params)
    svm_2d = SVC(kernel="rbf", C=params["C"], gamma=params["gamma"],
                 class_weight=class_weight, random_state=RANDOM_STATE).fit(P_s, y_s)
    var = pca.explained_variance_ratio_
    plot_decision_regions(
        P_s, y_s.to_numpy(), svm_2d.predict,
        f"SVM RBF ajusté dans le plan ACP ({var.sum():.1%} de variance, 8 000 lignes du train)",
        FIGURES_DIR / "svm_pca_decision_boundary.png",
        support=svm_2d.support_vectors_,
        axis_names=(f"CP1 ({var[0]:.1%})", f"CP2 ({var[1]:.1%})"),
    )
    write_json(METRICS_DIR / "svm_pca_2d.json", {
        "explained_variance_ratio": var.tolist(),
        "n_support_2d": int(len(svm_2d.support_)),
        "note": "Illustration uniquement : le SVM 2D n'est pas le modèle évalué.",
    })
    print("\nTerminé.")


if __name__ == "__main__":
    main()
