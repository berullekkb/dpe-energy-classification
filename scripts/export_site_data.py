"""Exporte les données du site (docs/data/*.json) à partir des résultats.

Aucun chiffre du site n'est saisi à la main : tout provient de
results/metrics/ ou des modèles entraînés, via ce script.

Fichiers produits :
- meta.json        : tailles des ensembles, répartition des classes, profils
- benchmark.json   : 11 modèles (validation), matrices de confusion, test final
- replay.json      : rejeu du protocole historique (protocol_replay.csv)
- sweep.json       : effet du poids de la classe Performant
- tsne.json        : 6 000 logements de validation projetés (t-SNE)
- model.json       : LightGBM compact + encodage, pour prédire dans le navigateur
- simulator.json   : description des 19 variables et logements réels de départ
- parity.json      : 500 logements de validation et probabilités Python,
                     pour vérifier que le navigateur calcule exactement pareil

Prérequis : avoir lancé les scripts d'entraînement, compare_models.py,
visualize_projections.py, sweep_class_weights.py et evaluate_final.py.
"""

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from dpe.benchmarking import read_json
from dpe.config import (
    GROUPS,
    METRICS_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
)
from dpe.data_loader import load_processed_data
from dpe.preprocessing import split_features_target
from dpe.training import split_train_validation_test

SITE_DATA = PROJECT_ROOT / "docs" / "data"
SIM_MODEL = "lightgbm"
N_TEMPLATES = 300
N_PARITY = 500

# Libellés des variables. Les codes ordinaux suivent l'ordre du notebook 02 ;
# voir docs/audit_notebooks.md (section 8) pour les codes de l'isolation.
ISOLATION = {0: "Insuffisante", 1: "Moyenne", 2: "Bonne", 4: "Très bonne"}
FIELDS = [
    # (variable, libellé, groupe, type)
    ("type_batiment", "Type de bâtiment", "main", "cat"),
    ("periode_construction_enc", "Période de construction", "main", "ord"),
    ("type_energie_principale_chauffage", "Énergie de chauffage", "main", "cat"),
    ("type_generateur_n1_installation_n1", "Générateur de chauffage", "main", "cat"),
    ("qualite_isolation_enveloppe_enc", "Isolation de l'enveloppe", "main", "ord"),
    ("ventilation_posterieure_2012", "Ventilation installée après 2012", "main", "ord"),
    ("qualite_isolation_murs_enc", "Isolation des murs", "advanced", "ord"),
    ("qualite_isolation_menuiseries_enc", "Isolation des menuiseries", "advanced", "ord"),
    ("qualite_isolation_plancher_bas_enc", "Isolation du plancher bas", "advanced", "ord"),
    ("hauteur_sous_plafond", "Hauteur sous plafond (m)", "advanced", "num"),
    ("type_installation_chauffage_n1", "Installation de chauffage", "advanced", "cat"),
    ("configuration_installation_chauffage_n1", "Configuration du chauffage", "advanced", "cat"),
    ("type_emetteur_installation_chauffage_n1", "Émetteurs de chauffage", "advanced", "cat"),
    ("usage_generateur_n1_installation_n1", "Usage du générateur", "advanced", "cat"),
    ("type_installation_ecs_n1", "Installation d'eau chaude", "advanced", "cat"),
    ("type_generateur_n1_ecs_n1", "Générateur d'eau chaude", "advanced", "cat"),
    ("type_energie_principale_ecs", "Énergie de l'eau chaude", "advanced", "cat"),
    ("volume_stockage_generateur_n1_ecs_n1", "Volume du ballon d'eau chaude (L)", "advanced", "litres"),
    ("classe_inertie_batiment_enc", "Inertie du bâtiment", "fixed", "ord"),
]
ORDINAL_LABELS = {
    "periode_construction_enc": {0: "Avant 1975", 1: "1975 – 1988", 2: "1989 – 2012",
                                 3: "2013 – 2021", 4: "Après 2021"},
    "qualite_isolation_enveloppe_enc": ISOLATION,
    "qualite_isolation_murs_enc": ISOLATION,
    "qualite_isolation_menuiseries_enc": ISOLATION,
    "qualite_isolation_plancher_bas_enc": ISOLATION,
    "ventilation_posterieure_2012": {0: "Non", 1: "Oui"},
    # Seul le code 1 (« Moyenne ») est certain : la variable n'est pas modifiable.
    "classe_inertie_batiment_enc": {1: "Moyenne"},
}


def write(name, payload):
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    path = SITE_DATA / name
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")
    print(f"{path.relative_to(PROJECT_ROOT)} : {len(text) / 1e3:.0f} ko")


def rnd(x, n=6):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


# ------------------------------------------------------------------ modèle
def flatten_tree(node):
    """Arbre LightGBM -> tableaux (feature, seuil, gauche, droite, feuilles).

    Les enfants négatifs désignent une feuille : -(k + 1) pour la feuille k.
    """
    feats, thrs, lefts, rights, leaves = [], [], [], [], []

    def visit(n):
        if "leaf_value" in n:
            leaves.append(n["leaf_value"])
            return -len(leaves)
        if n["decision_type"] != "<=":
            raise ValueError(f"Décision non gérée : {n['decision_type']}")
        idx = len(feats)
        feats.append(n["split_feature"])
        thrs.append(n["threshold"])
        lefts.append(None)
        rights.append(None)
        lefts[idx] = visit(n["left_child"])
        rights[idx] = visit(n["right_child"])
        return idx

    visit(node)
    return {"f": feats, "t": thrs, "l": lefts, "r": rights, "v": leaves}


def export_model(pipe):
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    booster = clf.estimator_.booster_
    dump = booster.dump_model()

    features = []
    for name, transformer, columns in pre.transformers_:
        if name == "remainder":
            continue
        for col in columns:
            spec = {"name": col, "kind": "cat" if name == "cat" else "num"}
            if name == "cat":
                encoder = transformer.named_steps["encoder"]
                j = list(columns).index(col)
                spec["map"] = {str(k): float(v) for k, v in encoder.mappings_[j].items()}
                spec["prior"] = float(encoder.prior_)
            features.append(spec)

    trees = [flatten_tree(t["tree_structure"]) for t in dump["tree_info"]]
    # Seuils mis en table (peu de valeurs distinctes) : fichier plus léger,
    # valeurs strictement identiques.
    thresholds = sorted({t for tree in trees for t in tree["t"]})
    position = {t: i for i, t in enumerate(thresholds)}
    for tree in trees:
        tree["t"] = [position[t] for t in tree["t"]]
    return {
        "model": SIM_MODEL,
        "classes": [str(c) for c in clf.classes_],
        "num_class": dump["num_class"],
        "features": features,
        "thresholds": thresholds,
        "trees": trees,
        "note": "LightGBM optimisé (Optuna, CV par profils) ; encodage cible "
                "historique (category_encoders, smoothing=3, min_samples_leaf=50).",
    }


# ------------------------------------------------------------------ main
def main():
    df = load_processed_data()
    X, y = split_features_target(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_validation_test(X, y)
    leakage = read_json(METRICS_DIR / "leakage_audit.json")

    # meta
    write("meta.json", {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": len(df), "n_features": X.shape[1],
        "class_counts": {g: int((y == g).sum()) for g in GROUPS},
        "split_sizes": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "n_profiles": leakage["n_profils_uniques"],
        "share_rows_duplicated_profiles": rnd(leakage["part_lignes_profils_dupliques"]),
        "contradictory_profiles": leakage["profils_etiquettes_contradictoires"],
        "contradictory_rows": leakage["lignes_profils_contradictoires"],
    })

    # benchmark
    bench = pd.read_csv(METRICS_DIR / "benchmark_validation.csv")
    models = []
    for r in bench.itertuples():
        conf = pd.read_csv(METRICS_DIR / f"{r.model}_validation_confusion.csv", index_col=0)
        conf = conf.loc[list(GROUPS), list(GROUPS)]
        models.append({
            "id": r.model, "label": r.label, "family": r.family, "origin": r.origin,
            "cv_f1_macro": rnd(r.f1_macro_cv), "cv_f1_macro_std": rnd(r.f1_macro_cv_std),
            "f1_macro": rnd(r.val_f1_macro), "balanced_accuracy": rnd(r.val_balanced_accuracy),
            "accuracy": rnd(r.val_accuracy),
            "per_class": {g: {k: rnd(getattr(r, f"{g}_{k}")) for k in ("precision", "recall", "f1")}
                          for g in GROUPS},
            "confusion": conf.to_numpy().tolist(),
        })
    final = read_json(METRICS_DIR / "final_test_evaluation.json")
    tpv = final["test"]["train_plus_validation"]
    conf_test = pd.read_csv(METRICS_DIR / "final_test_confusion.csv", index_col=0)
    conf_test = conf_test.loc[list(GROUPS), list(GROUPS)]
    write("benchmark.json", {
        "labels": list(GROUPS), "models": models,
        "selected": read_json(METRICS_DIR / "selected_model.json")["selected_model"],
        "test": {
            "model": final["model"], "label": final["label"],
            "metrics": {k: rnd(v) for k, v in tpv["metrics"].items()},
            "per_class": {g: {k: rnd(tpv["per_class"][g][k]) for k in ("precision", "recall", "f1")}
                          for g in GROUPS},
            "confusion": conf_test.to_numpy().tolist(),
            "baseline_f1_macro": rnd(final["reference_baseline_test"]["metrics"]["f1_macro"]),
        },
    })

    # rejeu du protocole
    rep = pd.read_csv(METRICS_DIR / "protocol_replay.csv")
    write("replay.json", {
        "share_seen_profiles": rnd(rep["part_profils_vus"].iloc[0]),
        "models": [{
            "id": r.model, "label": r.label, "stratified": rnd(r.f1_stratifie),
            "group": rnd(r.f1_par_profils), "seen": rnd(r.f1_profils_vus),
            "unseen": rnd(r.f1_profils_nouveaux),
        } for r in rep.itertuples()],
    })

    # balayage des poids
    sw = pd.read_csv(METRICS_DIR / "class_weight_sweep.csv")
    chosen = read_json(METRICS_DIR / "random_forest_best_params.json")["params"]["weight_Performant"]
    write("sweep.json", {
        "model": "random_forest", "chosen_weight": int(chosen),
        "points": [{
            "w": int(r.weight_Performant), "f1_macro": rnd(r.f1_macro),
            "recall": rnd(r.Performant_recall), "precision": rnd(r.Performant_precision),
            "f1": rnd(r.Performant_f1),
        } for r in sw.itertuples()],
    })

    # t-SNE
    pts = pd.read_csv(METRICS_DIR / "tsne_validation_points.csv")
    feats = X.loc[pts["row_id"]]
    lo, hi = pts[["tsne_1", "tsne_2"]].min(), pts[["tsne_1", "tsne_2"]].max()
    norm = (pts[["tsne_1", "tsne_2"]] - lo) / (hi - lo)
    hover_cols = ["type_batiment", "periode_construction_enc",
                  "type_energie_principale_chauffage", "qualite_isolation_enveloppe_enc"]
    dicts = {c: sorted(feats[c].astype(str).unique()) for c in hover_cols}
    value_labels = {c: {str(k): v for k, v in ORDINAL_LABELS[c].items()}
                    for c in hover_cols if c in ORDINAL_LABELS}
    write("tsne.json", {
        "model": read_json(METRICS_DIR / "projections_summary.json")["model"],
        "labels": list(GROUPS), "hover_columns": hover_cols, "dictionaries": dicts,
        "value_labels": value_labels,
        "points": [
            [rnd(a, 4), rnd(b, 4), GROUPS.index(t), GROUPS.index(p)]
            + [dicts[c].index(str(v)) for c, v in zip(hover_cols, row)]
            for a, b, t, p, row in zip(norm["tsne_1"], norm["tsne_2"], pts["true"],
                                       pts["predicted"], feats[hover_cols].itertuples(index=False))
        ],
    })

    # modèle du simulateur
    pipe = joblib.load(MODELS_DIR / f"{SIM_MODEL}.joblib")
    write("model.json", export_model(pipe))
    sim_val = read_json(METRICS_DIR / f"{SIM_MODEL}_validation.json")

    # simulateur : variables et logements réels de départ (validation)
    fields = []
    for col, label, group, kind in FIELDS:
        field = {"name": col, "label": label, "group": group, "kind": kind}
        if kind == "cat":
            counts = X_train[col].value_counts()
            field["options"] = [{"value": str(v), "label": str(v), "count": int(n)}
                                for v, n in counts.items()]
        elif kind == "ord":
            counts = X_train[col].value_counts()
            labels = ORDINAL_LABELS[col]
            field["options"] = [{"value": int(code), "label": labels[code],
                                 "count": int(counts.get(code, 0))}
                                for code in sorted(labels)]
        elif kind == "num":
            field.update({"min": float(X_train[col].min()), "max": float(X_train[col].max()),
                          "step": 0.05})
        elif kind == "litres":
            litres = np.expm1(X_train[col])
            field.update({"min": 0, "max": int(np.ceil(litres.quantile(0.995) / 50) * 50),
                          "step": 10, "transform": "log1p"})
        fields.append(field)

    rng = np.random.default_rng(RANDOM_STATE)
    template_idx = rng.choice(len(X_val), size=N_TEMPLATES, replace=False)
    templates_X = X_val.iloc[template_idx]
    templates_pred = pipe.predict(templates_X)

    def to_ui(row):
        out = {}
        for col, _, _, kind in FIELDS:
            v = row[col]
            if kind == "cat":
                out[col] = str(v)
            elif kind == "ord":
                out[col] = int(v)
            elif kind == "litres":
                out[col] = int(round(float(np.expm1(v))))
            else:
                out[col] = round(float(v), 2)
        return out

    write("simulator.json", {
        "model": SIM_MODEL, "model_label": "LightGBM",
        "validation_f1_macro": rnd(sim_val["metrics"]["f1_macro"]),
        "fields": fields,
        "templates": [{"features": to_ui(r), "true": str(t), "predicted": str(p)}
                      for (_, r), t, p in zip(templates_X.iterrows(),
                                              y_val.iloc[template_idx], templates_pred)],
        "fixed_note": "L'inertie du bâtiment n'est pas modifiable : ses codes sont "
                      "ambigus dans les données (voir l'audit). La valeur du logement "
                      "réel de départ est conservée.",
    })

    # parité navigateur / Python
    parity_idx = rng.choice(len(X_val), size=N_PARITY, replace=False)
    Xp = X_val.iloc[parity_idx]
    proba = pipe.predict_proba(Xp)
    write("parity.json", {
        "classes": [str(c) for c in pipe.named_steps["clf"].classes_],
        "rows": [{c: (v.item() if hasattr(v, "item") else v) for c, v in r.items()}
                 for r in Xp.to_dict(orient="records")],
        "expected": proba.round(12).tolist(),
    })


if __name__ == "__main__":
    main()
