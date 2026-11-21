"""Audit des risques de fuite de cible.

Le DPE est calculé de façon réglementaire (méthode 3CL) à partir des
caractéristiques du logement. Les variables explicatives sont donc, par
construction, liées à l'étiquette. L'objectif de cet audit est de distinguer :

- les variables qui sont des *entrées* légitimes du calcul (isolation,
  système de chauffage, période de construction, ...) ;
- les variables qui *révèleraient directement* l'étiquette (consommation,
  émissions GES, étiquette GES, classe ...), qui ne doivent pas être utilisées.

Contrôles effectués (uniquement sur TRAIN pour les statistiques et sur
VALIDATION pour la performance, jamais sur TEST) :

1. Recherche de noms de colonnes associés à des sorties du calcul DPE.
2. Association de chaque variable avec la cible (V de Cramér, train).
3. Pouvoir prédictif d'un modèle n'utilisant qu'une seule variable
   (F1 macro en validation). Un score proche de 1 signalerait une fuite.
4. Profils identiques portant des étiquettes différentes (bruit
   irréductible) et part des lignes appartenant à des profils dupliqués.

Sorties : results/metrics/leakage_audit.csv et leakage_audit.json
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import f1_score

from dpe.config import METRICS_DIR, TARGET
from dpe.data_loader import load_processed_data
from dpe.preprocessing import split_features_target
from dpe.training import split_train_validation_test

# Fragments de noms typiques des sorties du calcul DPE (ADEME).
SUSPICIOUS_FRAGMENTS = (
    "conso", "ep_", "ef_", "emission", "ges", "etiquette", "classe_dpe",
    "cout", "besoin", "deperdition", "ubat", "kwh",
)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """V de Cramér corrigé du biais (Bergsma, 2013)."""
    table = pd.crosstab(x.astype(str), y.astype(str))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def single_feature_f1(x_train, y_train, x_val, y_val) -> float:
    """F1 macro d'une règle « modalité -> classe la plus probable ».

    Les classes sont repondérées (équivalent class_weight='balanced') pour
    que la règle ne prédise pas uniquement la classe majoritaire.
    """
    weights = 1.0 / y_train.value_counts(normalize=True)
    table = pd.crosstab(x_train.astype(str), y_train).mul(weights, axis=1)
    rule = table.idxmax(axis=1)
    fallback = y_train.value_counts().idxmax()
    pred = x_val.astype(str).map(rule).fillna(fallback)
    return float(f1_score(y_val, pred, average="macro"))


def main():
    df = load_processed_data()
    X, y = split_features_target(df)
    X_train, X_val, _, y_train, y_val, _ = split_train_validation_test(X, y)

    rows = []
    for col in X.columns:
        rows.append({
            "variable": col,
            "nom_suspect": any(f in col.lower() for f in SUSPICIOUS_FRAGMENTS),
            "n_modalites": int(X_train[col].nunique()),
            "cramers_v_train": cramers_v(X_train[col], y_train),
            "f1_macro_validation_seule": single_feature_f1(
                X_train[col], y_train, X_val[col], y_val
            ),
        })

    audit = pd.DataFrame(rows).sort_values(
        "f1_macro_validation_seule", ascending=False
    )

    # Profils dupliqués et étiquettes contradictoires (jeu complet, sans
    # entraînement : simple description des données).
    grouped = df.groupby(list(X.columns), dropna=False)[TARGET]
    stats = grouped.agg(["nunique", "size"])
    conflicting = stats[stats["nunique"] > 1]
    duplicated_rows = int(X.duplicated(keep=False).sum())

    summary = {
        "n_observations": int(len(df)),
        "n_variables": int(X.shape[1]),
        "n_profils_uniques": int(len(stats)),
        "lignes_dans_profils_dupliques": duplicated_rows,
        "part_lignes_profils_dupliques": duplicated_rows / len(df),
        "profils_etiquettes_contradictoires": int(len(conflicting)),
        "lignes_profils_contradictoires": int(conflicting["size"].sum()),
        "variables_nom_suspect": audit.loc[
            audit["nom_suspect"], "variable"
        ].tolist(),
        "max_f1_une_variable": float(audit["f1_macro_validation_seule"].max()),
        "seuil_alerte_f1_une_variable": 0.90,
        "conclusion": None,
    }

    alert = (
        summary["variables_nom_suspect"]
        or summary["max_f1_une_variable"] >= summary["seuil_alerte_f1_une_variable"]
    )
    summary["conclusion"] = (
        "ALERTE : variable potentiellement révélatrice de la cible."
        if alert
        else (
            "Aucune variable ne révèle directement l'étiquette : pas de "
            "sortie du calcul DPE parmi les colonnes et aucune variable seule "
            "ne dépasse le seuil d'alerte. Les variables sont des entrées du "
            "calcul 3CL : le problème revient à approcher la méthode "
            "réglementaire à partir d'une description partielle du logement."
        )
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(METRICS_DIR / "leakage_audit.csv", index=False)
    with (METRICS_DIR / "leakage_audit.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(audit.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
