"""Entraînement final, évaluation et sauvegarde des résultats par modèle."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import GROUPS, METRICS_DIR
from .evaluation import evaluate_classifier
from .models import class_weights_from_params, get_spec


def per_class_table(report: dict) -> pd.DataFrame:
    """Précision, rappel, F1 et support par classe (ordre de GROUPS)."""
    return pd.DataFrame(
        {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": int(report[label]["support"]),
            }
            for label in GROUPS
        }
    ).T


def evaluate_pipeline(pipeline, X, y):
    """Prédit puis évalue ; renvoie (metrics, report, confusion, y_pred)."""
    y_pred = pipeline.predict(X)
    metrics, report, confusion = evaluate_classifier(y, y_pred)
    return metrics, report, confusion, y_pred


def _to_builtin(value):
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_builtin(payload), f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def save_validation_results(name, params, cv_summary, metrics, report,
                            confusion, split_sizes, extra=None,
                            output_dir: Path = METRICS_DIR) -> dict:
    """Écrit `<name>_best_params.json`, `<name>_validation.json` et la matrice."""
    spec = get_spec(name)
    output_dir = Path(output_dir)

    best_params = {
        "model": name,
        "label": spec.label,
        "family": spec.family,
        "origin": spec.origin,
        "historical_notebook": spec.notebook,
        "encoding": spec.encoding,
        "scaled": spec.scale,
        "params": params,
        "class_weight": class_weights_from_params(params),
        "selection_metric": "f1_macro",
        "tuned_on": "train (StratifiedGroupKFold par profils)",
        **cv_summary,
    }
    write_json(output_dir / f"{name}_best_params.json", best_params)

    payload = {
        "model": name,
        "label": spec.label,
        "family": spec.family,
        "origin": spec.origin,
        "evaluated_on": "validation",
        "test_evaluated": False,
        **split_sizes,
        "metrics": metrics,
        "per_class": per_class_table(report).to_dict(orient="index"),
        "cv": cv_summary,
        **(extra or {}),
    }
    write_json(output_dir / f"{name}_validation.json", payload)
    confusion.to_csv(output_dir / f"{name}_validation_confusion.csv",
                     encoding="utf-8")
    return payload


def load_validation_results(output_dir: Path = METRICS_DIR) -> pd.DataFrame:
    """Tableau comparatif à partir de tous les `*_validation.json`."""
    rows = []
    for path in sorted(Path(output_dir).glob("*_validation.json")):
        res = read_json(path)
        row = {
            "model": res["model"],
            "label": res["label"],
            "family": res["family"],
            "origin": res["origin"],
            "f1_macro_cv": res["cv"].get("f1_macro_cv_mean"),
            "f1_macro_cv_std": res["cv"].get("f1_macro_cv_std"),
            **{f"val_{k}": v for k, v in res["metrics"].items()},
        }
        for label in GROUPS:
            for key in ("precision", "recall", "f1"):
                row[f"{label}_{key}"] = res["per_class"][label][key]
        row["fit_seconds"] = res.get("fit_seconds")
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values("val_f1_macro", ascending=False)
        .reset_index(drop=True)
    )
