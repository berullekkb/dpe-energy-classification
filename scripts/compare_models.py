"""Benchmark global des modèles (équivalent amélioré de 07_model_comparison).

Entrées : les résultats de validation produits par optimize_models.py et
train_svm.py (results/metrics/*_validation.json).

Sorties :
- results/metrics/benchmark_validation.csv : tableau comparatif complet ;
- results/benchmark_validation.md : tableau lisible (README) ;
- results/metrics/historical_vs_current.csv : scores historiques des
  notebooks (split stratifié, sélection sur le test) à côté des scores du
  protocole actuel, avec l'avertissement qu'ils ne sont pas comparables ;
- results/metrics/protocol_replay.csv : rejeu contrôlé de l'effet du split
  stratifié ligne à ligne, sur TRAIN + VALIDATION uniquement ;
- results/metrics/selected_model.json : modèle retenu (meilleur F1 macro en
  validation), figé AVANT toute évaluation sur le test ;
- figures de comparaison et matrices de confusion dans results/figures/.

Usage :
    python scripts/compare_models.py [--skip-replay]
"""

import argparse

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from dpe.benchmarking import load_validation_results, read_json, write_json
from dpe.config import (
    FIGURES_DIR,
    METRICS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
)
from dpe.models import get_spec
from dpe.optimization import fit_pipeline
from dpe.training import split_data
from dpe.visualization import (
    GROUP_COLORS,
    plot_bar_scores,
    plot_confusion,
    plot_grouped_bars,
)

import optimize_models as om

# Scores publiés dans les notebooks (protocole historique : split stratifié
# ligne à ligne 80/20, sélection sur le test). Notebooks 03 à 07.
HISTORICAL_TEST_F1 = {
    "lr_elasticnet": (0.726, "03"),
    "lr_ridge": (0.704, "03"),
    "lr_lasso": (0.708, "03"),
    "svm_rbf": (0.707, "04"),
    "decision_tree": (0.766, "05"),
    "random_forest": (0.788, "05"),
    "adaboost": (0.706, "06"),
    "gradient_boosting": (0.780, "06"),
    "xgboost": (0.777, "06"),
}
HISTORICAL_FINAL_F1 = {  # notebook 07 (50 essais)
    "lr_elasticnet": 0.715,
    "svm_rbf": 0.743,
    "random_forest": 0.790,
    "gradient_boosting": 0.799,
}

REPLAY_MODELS = ("lr_elasticnet", "decision_tree", "random_forest", "xgboost")


def markdown_table(table: pd.DataFrame) -> str:
    cols = {
        "label": "Modèle",
        "origin": "Origine",
        "f1_macro_cv": "F1 macro CV (train)",
        "val_f1_macro": "F1 macro val.",
        "val_balanced_accuracy": "Balanced acc. val.",
        "val_accuracy": "Accuracy val.",
        "Performant_f1": "F1 Performant",
        "Passable_f1": "F1 Passable",
        "Energivore_f1": "F1 Energivore",
    }
    df = table[list(cols)].rename(columns=cols).copy()
    df["Origine"] = df["Origine"].map({"historical": "historique", "new": "nouveau"})
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def protocol_replay(table: pd.DataFrame, n_repeats: int = 5) -> pd.DataFrame:
    """Mesure l'optimisme du split stratifié ligne à ligne.

    Sur DEV = TRAIN + VALIDATION (le test n'est pas utilisé), chaque modèle
    est entraîné avec ses meilleurs paramètres sur 80 % de DEV puis évalué
    sur les 20 % restants, selon deux découpages, répétés `n_repeats` fois :
    - stratifié ligne à ligne (protocole des notebooks) : on mesure aussi le
      F1 séparément sur les lignes dont le profil exact a été vu à
      l'entraînement et sur les autres (effet de mémorisation) ;
    - par profils (protocole actuel).
    """
    X_train, y_train, X_val, y_val, _ = om.load_splits()
    X_dev = pd.concat([X_train, X_val])
    y_dev = pd.concat([y_train, y_val])
    keys = X_dev.astype(str).apply(lambda r: "|".join(r), axis=1)

    rows = []
    available = set(table["model"])
    for name in REPLAY_MODELS:
        if name not in available:
            continue
        spec = get_spec(name)
        params = read_json(METRICS_DIR / f"{name}_best_params.json")["params"]
        for rep in range(n_repeats):
            seed = RANDOM_STATE + rep
            Xa, Xb, ya, yb = train_test_split(
                X_dev, y_dev, test_size=0.2, random_state=seed, stratify=y_dev)
            pred = fit_pipeline(spec, params, Xa, ya).predict(Xb)
            seen = keys.loc[Xb.index].isin(set(keys.loc[Xa.index])).to_numpy()
            Xg, Xh, yg, yh = split_data(X_dev, y_dev, strategy="group",
                                        test_size=0.2, random_state=seed)
            pred_g = fit_pipeline(spec, params, Xg, yg).predict(Xh)
            rows.append({
                "model": name, "label": spec.label, "repetition": rep,
                "f1_stratifie": f1_score(yb, pred, average="macro"),
                "f1_stratifie_profils_vus": f1_score(
                    yb[seen], pred[seen], average="macro"),
                "f1_stratifie_profils_nouveaux": f1_score(
                    yb[~seen], pred[~seen], average="macro"),
                "part_profils_vus": float(seen.mean()),
                "f1_par_profils": f1_score(yh, pred_g, average="macro"),
            })
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["model", "label"], sort=False).agg(
        f1_stratifie=("f1_stratifie", "mean"),
        f1_stratifie_std=("f1_stratifie", "std"),
        f1_par_profils=("f1_par_profils", "mean"),
        f1_par_profils_std=("f1_par_profils", "std"),
        f1_profils_vus=("f1_stratifie_profils_vus", "mean"),
        f1_profils_nouveaux=("f1_stratifie_profils_nouveaux", "mean"),
        part_profils_vus=("part_profils_vus", "mean"),
    ).reset_index()
    summary["optimisme"] = summary["f1_stratifie"] - summary["f1_par_profils"]
    summary["ecart_vus_nouveaux"] = (
        summary["f1_profils_vus"] - summary["f1_profils_nouveaux"])
    detail.to_csv(METRICS_DIR / "protocol_replay_detail.csv", index=False)
    print(summary.drop(columns="model").to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-replay", action="store_true")
    args = parser.parse_args(argv)

    table = load_validation_results()
    if table.empty:
        raise SystemExit("Aucun résultat de validation : lancer optimize_models.py")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(METRICS_DIR / "benchmark_validation.csv", index=False)
    (RESULTS_DIR / "benchmark_validation.md").write_text(
        "# Benchmark sur la validation\n\n"
        "Optimisation Optuna sur le train (CV stratifiée par groupes de "
        "profils), évaluation sur la validation. Le test n'est pas utilisé.\n\n"
        + markdown_table(table) + "\n",
        encoding="utf-8",
    )

    print(table[["label", "origin", "f1_macro_cv", "val_f1_macro",
                 "val_balanced_accuracy", "Performant_f1", "Energivore_f1"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Figures
    plot_bar_scores(table, "label", "val_f1_macro",
                    "F1 macro sur la validation (protocole par profils)",
                    FIGURES_DIR / "benchmark_f1_macro_validation.png",
                    color_col="origin")
    plot_bar_scores(table, "label", "val_balanced_accuracy",
                    "Balanced accuracy sur la validation",
                    FIGURES_DIR / "benchmark_balanced_accuracy_validation.png",
                    color_col="origin", xlabel="Balanced accuracy")
    plot_grouped_bars(
        table, "label",
        {f"{g}_f1": (g, GROUP_COLORS[g]) for g in GROUP_COLORS},
        "F1 par classe sur la validation",
        FIGURES_DIR / "benchmark_f1_per_class_validation.png", ylabel="F1",
    )
    plot_grouped_bars(
        table, "label",
        {f"{g}_recall": (g, GROUP_COLORS[g]) for g in GROUP_COLORS},
        "Rappel par classe sur la validation",
        FIGURES_DIR / "benchmark_recall_per_class_validation.png",
        ylabel="Rappel",
    )
    for name, label in zip(table["model"], table["label"]):
        confusion = pd.read_csv(METRICS_DIR / f"{name}_validation_confusion.csv",
                                index_col=0)
        plot_confusion(confusion, f"{label} - validation",
                       FIGURES_DIR / f"{name}_validation_confusion.png")

    # Historique vs actuel
    hist = pd.DataFrame([
        {
            "model": name,
            "notebook": nb,
            "f1_test_historique_notebook": f1,
            "f1_test_historique_07": HISTORICAL_FINAL_F1.get(name),
        }
        for name, (f1, nb) in HISTORICAL_TEST_F1.items()
    ]).merge(table[["model", "label", "f1_macro_cv", "val_f1_macro"]],
             on="model", how="left")
    hist["avertissement"] = (
        "Historique : split stratifié ligne à ligne et sélection sur le test ; "
        "non comparable au protocole par profils."
    )
    hist.to_csv(METRICS_DIR / "historical_vs_current.csv", index=False)

    # Rejeu du protocole historique
    if not args.skip_replay:
        print("\n=== Rejeu : split stratifié vs split par profils (DEV) ===")
        replay = protocol_replay(table)
        replay.to_csv(METRICS_DIR / "protocol_replay.csv", index=False)

    # Sélection figée avant le test
    best = table.iloc[0]
    runner_up = table.iloc[1] if len(table) > 1 else None
    selection = {
        "selected_model": best["model"],
        "label": best["label"],
        "criterion": "F1 macro maximal sur la validation",
        "validation_f1_macro": best["val_f1_macro"],
        "validation_balanced_accuracy": best["val_balanced_accuracy"],
        "cv_f1_macro_train": best["f1_macro_cv"],
        "runner_up": None if runner_up is None else runner_up["model"],
        "runner_up_validation_f1_macro": (
            None if runner_up is None else runner_up["val_f1_macro"]),
        "test_used_for_selection": False,
        "ranking": table[["model", "val_f1_macro"]].to_dict(orient="records"),
    }
    write_json(METRICS_DIR / "selected_model.json", selection)
    print(f"\nModèle retenu (validation) : {best['label']} "
          f"(F1 macro {best['val_f1_macro']:.4f})")


if __name__ == "__main__":
    main()
