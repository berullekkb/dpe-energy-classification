"""Tests de la sauvegarde et de l'agrégation des résultats de validation."""

import pandas as pd

from dpe.benchmarking import (
    load_validation_results,
    per_class_table,
    read_json,
    save_validation_results,
)
from dpe.evaluation import evaluate_classifier


def fake_results():
    y_true = ["Performant", "Passable", "Energivore", "Passable"]
    y_pred = ["Performant", "Passable", "Passable", "Passable"]
    return evaluate_classifier(y_true, y_pred)


def test_per_class_table_follows_groups_order():
    _, report, _ = fake_results()
    table = per_class_table(report)
    assert list(table.index) == ["Performant", "Passable", "Energivore"]
    assert table.loc["Energivore", "recall"] == 0.0
    assert table.loc["Passable", "support"] == 2


def test_save_and_load_validation_results(tmp_path):
    metrics, report, confusion = fake_results()
    cv = {"f1_macro_cv_mean": 0.7, "f1_macro_cv_std": 0.01}
    sizes = {"train_size": 10, "validation_size": 4, "test_size": 3}
    for name, shift in [("decision_tree", 0.0), ("random_forest", 0.1)]:
        m = dict(metrics, f1_macro=metrics["f1_macro"] + shift)
        save_validation_results(
            name, {"max_depth": 3, "weight_Performant": 4}, cv, m, report,
            confusion, sizes, extra={"fit_seconds": 1.0}, output_dir=tmp_path,
        )

    params = read_json(tmp_path / "decision_tree_best_params.json")
    assert params["class_weight"]["Performant"] == 4.0
    assert params["origin"] == "historical"

    payload = read_json(tmp_path / "random_forest_validation.json")
    assert payload["test_evaluated"] is False
    assert payload["evaluated_on"] == "validation"

    confusion_back = pd.read_csv(
        tmp_path / "decision_tree_validation_confusion.csv", index_col=0)
    assert confusion_back.to_numpy().sum() == 4

    table = load_validation_results(tmp_path)
    assert list(table["model"]) == ["random_forest", "decision_tree"]
    assert "Performant_recall" in table.columns


def test_load_validation_results_empty(tmp_path):
    assert load_validation_results(tmp_path).empty
