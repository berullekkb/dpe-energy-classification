"""Cohérence du site (docs/data) avec les résultats du dépôt.

Ces tests n'utilisent que la bibliothèque standard et ne demandent ni les
données ni les modèles : ils tournent sur un simple clone du dépôt.
"""

import csv
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data"
METRICS = ROOT / "results" / "metrics"
FILES = ("meta", "benchmark", "replay", "sweep", "tsne", "model", "simulator", "parity")


def load(name):
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def rows(name):
    with (METRICS / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("name", FILES)
def test_data_files_exist_and_parse(name):
    assert isinstance(load(name), dict)


def test_benchmark_matches_results():
    site = {m["id"]: m for m in load("benchmark")["models"]}
    reference = rows("benchmark_validation.csv")
    assert set(site) == {r["model"] for r in reference}
    for r in reference:
        m = site[r["model"]]
        assert m["f1_macro"] == pytest.approx(float(r["val_f1_macro"]), abs=1e-6)
        assert m["per_class"]["Performant"]["recall"] == pytest.approx(
            float(r["Performant_recall"]), abs=1e-6)
        assert sum(map(sum, m["confusion"])) == 11130


def test_final_test_matches_results():
    test = load("benchmark")["test"]
    final = json.loads((METRICS / "final_test_evaluation.json").read_text(encoding="utf-8"))
    expected = final["test"]["train_plus_validation"]["metrics"]["f1_macro"]
    assert test["metrics"]["f1_macro"] == pytest.approx(expected, abs=1e-6)
    assert sum(map(sum, test["confusion"])) == final["test_size"]


def test_replay_and_sweep_match_results():
    replay = {m["id"]: m for m in load("replay")["models"]}
    for r in rows("protocol_replay.csv"):
        assert replay[r["model"]]["stratified"] == pytest.approx(float(r["f1_stratifie"]), abs=1e-6)
        assert replay[r["model"]]["group"] == pytest.approx(float(r["f1_par_profils"]), abs=1e-6)
    sweep = load("sweep")["points"]
    reference = rows("class_weight_sweep.csv")
    assert [p["w"] for p in sweep] == list(range(1, 26))
    for p, r in zip(sweep, reference):
        assert p["recall"] == pytest.approx(float(r["Performant_recall"]), abs=1e-6)


def test_simulator_describes_all_19_variables():
    sim = load("simulator")
    groups = [f["group"] for f in sim["fields"]]
    assert len(groups) == 19
    assert groups.count("main") == 6 and groups.count("advanced") == 12
    assert groups.count("fixed") == 1
    model_features = {f["name"] for f in load("model")["features"]}
    assert {f["name"] for f in sim["fields"]} == model_features
    assert len(sim["templates"]) == 300


def test_model_and_tsne_are_well_formed():
    model = load("model")
    n_thr = len(model["thresholds"])
    assert model["num_class"] == 3 and len(model["trees"]) % 3 == 0
    for tree in model["trees"][:50]:
        assert all(0 <= t < n_thr for t in tree["t"])
        assert len(tree["v"]) == len(tree["f"]) + 1
    points = load("tsne")["points"]
    assert len(points) == 6000
    assert all(0 <= p[0] <= 1 and 0 <= p[1] <= 1 for p in points)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js absent")
def test_browser_predictions_match_python():
    result = subprocess.run(["node", str(ROOT / "tests" / "js" / "parity.test.mjs")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
