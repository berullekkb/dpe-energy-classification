"""Analyse des erreurs sur la VALIDATION (le test n'est pas utilisé).

Pour le modèle retenu (results/metrics/selected_model.json) ou celui passé en
argument :

- types d'erreurs : confusions entre classes voisines (Performant <->
  Passable, Passable <-> Energivore) et confusions extrêmes (Performant <->
  Energivore) ;
- taux d'erreur et rappel par sous-groupe (type de bâtiment, période de
  construction, isolation de l'enveloppe, énergie de chauffage) ;
- confiance du modèle (probabilité de la classe prédite) pour les bonnes et
  les mauvaises prédictions ;
- accord entre modèles : proportion des lignes de validation mal classées
  par tous les modèles évalués (erreurs « dures »).

Sorties : results/metrics/error_analysis_*.csv|json et figures associées.

Usage :
    python scripts/analyze_errors.py [--model random_forest]
"""

import argparse

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dpe.benchmarking import read_json, write_json  # noqa: E402
from dpe.config import FIGURES_DIR, GROUPS, METRICS_DIR, MODELS_DIR  # noqa: E402
from dpe.models import get_spec  # noqa: E402

import optimize_models as om  # noqa: E402

SUBGROUPS = {
    "type_batiment": "Type de bâtiment",
    "periode_construction_enc": "Période de construction (0 = avant 1975, 4 = après 2021)",
    "qualite_isolation_enveloppe_enc": "Isolation de l'enveloppe (0 = insuffisante, 4 = très bonne)",
    "type_energie_principale_chauffage": "Énergie de chauffage",
}
ORDER = {"Energivore": 0, "Passable": 1, "Performant": 2}


def error_types(y_true, y_pred) -> pd.DataFrame:
    df = pd.DataFrame({"reel": y_true, "predit": y_pred})
    df = df[df["reel"] != df["predit"]]
    gap = (df["reel"].map(ORDER) - df["predit"].map(ORDER)).abs()
    df["type"] = np.where(gap == 2, "extreme (Performant <-> Energivore)",
                          "classes voisines")
    out = (df.groupby(["reel", "predit", "type"]).size()
           .rename("n").reset_index().sort_values("n", ascending=False))
    out["part_des_erreurs"] = out["n"] / len(df)
    return out


def subgroup_table(X, y_true, y_pred, column) -> pd.DataFrame:
    df = pd.DataFrame({"modalite": X[column].to_numpy(), "reel": y_true,
                       "predit": y_pred})
    rows = []
    for value, part in df.groupby("modalite"):
        row = {
            "variable": column,
            "modalite": value,
            "n": len(part),
            "taux_erreur": float((part["reel"] != part["predit"]).mean()),
        }
        for g in GROUPS:
            mask = part["reel"] == g
            row[f"rappel_{g}"] = (
                float((part.loc[mask, "predit"] == g).mean()) if mask.any() else np.nan
            )
            row[f"n_{g}"] = int(mask.sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)

    name = args.model or read_json(METRICS_DIR / "selected_model.json")["selected_model"]
    spec = get_spec(name)
    pipe = joblib.load(MODELS_DIR / f"{name}.joblib")

    _, _, X_val, y_val, _ = om.load_splits()
    y_true = y_val.to_numpy()
    y_pred = pipe.predict(X_val)
    wrong = y_true != y_pred
    print(f"Modèle : {spec.label} | erreurs validation : {wrong.sum()} / "
          f"{len(y_true)} ({wrong.mean():.1%})")

    # 1. Types d'erreurs
    types = error_types(y_true, y_pred)
    types.to_csv(METRICS_DIR / "error_analysis_types.csv", index=False)
    print(types.to_string(index=False))

    # 2. Sous-groupes
    sub = pd.concat([subgroup_table(X_val, y_true, y_pred, c) for c in SUBGROUPS])
    sub.to_csv(METRICS_DIR / "error_analysis_subgroups.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (col, title) in zip(axes.ravel(), SUBGROUPS.items()):
        part = sub[(sub["variable"] == col) & (sub["n"] >= 50)]
        part = part.sort_values("modalite" if col.endswith("_enc") else "taux_erreur")
        ax.barh(part["modalite"].astype(str), part["taux_erreur"], color="#2a6f97")
        for i, (v, n) in enumerate(zip(part["taux_erreur"], part["n"])):
            ax.text(v + 0.005, i, f"{v:.0%} (n={n})", va="center", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Taux d'erreur (validation)")
        ax.set_xlim(0, max(0.6, part["taux_erreur"].max() + 0.12))
    fig.suptitle(f"Taux d'erreur par sous-groupe - {spec.label}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "error_analysis_subgroups.png", dpi=130)
    plt.close(fig)

    # 3. Confiance
    summary = {"model": name, "n_validation": int(len(y_true)),
               "n_errors": int(wrong.sum()), "error_rate": float(wrong.mean())}
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(X_val).max(axis=1)
            summary["confidence_correct_median"] = float(np.median(proba[~wrong]))
            summary["confidence_errors_median"] = float(np.median(proba[wrong]))
            fig, ax = plt.subplots(figsize=(7, 4))
            bins = np.linspace(0.3, 1, 36)
            ax.hist(proba[~wrong], bins=bins, alpha=0.6, label="Correctes",
                    color="#2a6f97", density=True)
            ax.hist(proba[wrong], bins=bins, alpha=0.6, label="Erreurs",
                    color="#e07a5f", density=True)
            ax.set_xlabel("Probabilité de la classe prédite")
            ax.set_ylabel("Densité")
            ax.set_title(f"Confiance du modèle - {spec.label} (validation)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "error_analysis_confidence.png", dpi=130)
            plt.close(fig)
        except AttributeError:
            pass

    # 4. Accord entre modèles
    preds = {}
    for path in sorted(MODELS_DIR.glob("*.joblib")):
        preds[path.stem] = joblib.load(path).predict(X_val)
    wrong_all = np.column_stack([p != y_true for p in preds.values()])
    summary["models_compared"] = sorted(preds)
    summary["rows_wrong_for_all_models"] = int(wrong_all.all(axis=1).sum())
    summary["share_wrong_for_all_models"] = float(wrong_all.all(axis=1).mean())
    summary["share_wrong_for_at_least_one_model"] = float(wrong_all.any(axis=1).mean())
    summary["extreme_confusions"] = int(
        types.loc[types["type"].str.startswith("extreme"), "n"].sum())
    hard = pd.Series(y_true[wrong_all.all(axis=1)]).value_counts().to_dict()
    summary["hard_errors_by_true_class"] = hard
    write_json(METRICS_DIR / "error_analysis_summary.json", summary)
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
