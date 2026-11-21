"""Tests des fonctions pures des scripts d'analyse (sans données réelles)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

analyze_errors = pytest.importorskip("analyze_errors")
compare_models = pytest.importorskip("compare_models")


def test_error_types_distinguishes_extreme_confusions():
    y_true = ["Performant", "Performant", "Passable", "Energivore", "Passable"]
    y_pred = ["Energivore", "Passable", "Passable", "Passable", "Energivore"]
    table = analyze_errors.error_types(y_true, y_pred)
    assert table["n"].sum() == 4
    extreme = table[table["type"].str.startswith("extreme")]
    assert extreme["n"].sum() == 1
    assert table["part_des_erreurs"].sum() == pytest.approx(1.0)


def test_subgroup_table_error_rate_and_recall():
    X = pd.DataFrame({"type_batiment": ["maison", "maison", "appartement",
                                        "appartement"]})
    y_true = ["Passable", "Energivore", "Passable", "Performant"]
    y_pred = ["Passable", "Passable", "Passable", "Performant"]
    table = analyze_errors.subgroup_table(X, y_true, y_pred, "type_batiment")
    maison = table[table["modalite"] == "maison"].iloc[0]
    assert maison["taux_erreur"] == 0.5
    assert maison["rappel_Energivore"] == 0.0
    appart = table[table["modalite"] == "appartement"].iloc[0]
    assert appart["taux_erreur"] == 0.0


def test_markdown_table_formats_benchmark():
    table = pd.DataFrame([{
        "label": "Random Forest", "origin": "historical", "f1_macro_cv": 0.75,
        "val_f1_macro": 0.775, "val_balanced_accuracy": 0.79,
        "val_accuracy": 0.81, "Performant_f1": 0.74, "Passable_f1": 0.86,
        "Energivore_f1": 0.72,
    }])
    md = compare_models.markdown_table(table)
    lines = md.splitlines()
    assert lines[0].startswith("| Modèle")
    assert "historique" in lines[2]
    assert "0.775" in lines[2]
