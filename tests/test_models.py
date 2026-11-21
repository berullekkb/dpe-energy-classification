"""Tests du registre des modèles et des poids de classe."""

import numpy as np
import pandas as pd
import pytest

from dpe.models import (
    FAMILIES,
    MODEL_SPECS,
    LabelEncodedClassifier,
    class_weights_from_params,
    get_spec,
    make_sample_weight,
    resolve_models,
    split_model_params,
)
from dpe.optimization import fit_pipeline


def small_dataset(n=180, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.array(["Performant", "Passable", "Energivore"])
    y = pd.Series(labels[rng.integers(0, 3, n)])
    score = y.map({"Energivore": 0, "Passable": 1, "Performant": 2})
    X = pd.DataFrame({
        "hauteur_sous_plafond": 2.5 + rng.normal(0, 0.1, n),
        "volume_stockage_generateur_n1_ecs_n1": score * 2 + rng.normal(0, 1, n),
        "qualite_isolation_murs_enc": np.clip(score + rng.integers(-1, 2, n), 0, 3),
        "type_batiment": rng.choice(["maison", "appartement"], n),
        "type_energie_principale_chauffage": np.where(
            score > 0, "Gaz naturel", "Fioul domestique"),
    })
    return X, y


class FakeTrial:
    """Trial minimal : renvoie la borne basse de chaque intervalle."""

    def __init__(self):
        self.params = {}

    def suggest_int(self, name, low, high, step=1, log=False):
        self.params[name] = low
        return low

    def suggest_float(self, name, low, high, log=False):
        self.params[name] = low
        return low


def test_registry_covers_historical_scope():
    historical = {n for n, s in MODEL_SPECS.items() if s.origin == "historical"}
    assert historical == {
        "lr_elasticnet", "lr_ridge", "lr_lasso", "svm_rbf", "decision_tree",
        "random_forest", "adaboost", "gradient_boosting", "xgboost",
    }
    assert {"lightgbm", "catboost"} <= set(MODEL_SPECS)
    assert MODEL_SPECS["lightgbm"].origin == "new"
    assert MODEL_SPECS["svm_rbf"].tuning_sample == 8000


def test_every_family_member_is_registered():
    for members in FAMILIES.values():
        for name in members:
            assert name in MODEL_SPECS


def test_resolve_models_deduplicates_and_validates():
    assert resolve_models(["random_forest"], ["tree"]) == [
        "decision_tree", "random_forest"]
    with pytest.raises(ValueError):
        resolve_models(["inconnu"])
    with pytest.raises(ValueError):
        resolve_models(families=["inconnue"])
    with pytest.raises(ValueError):
        get_spec("inconnu")


def test_class_weights_and_sample_weights():
    params = {"weight_Performant": 7, "weight_Energivore": 3, "C": 1.0}
    model_params, weights = split_model_params(params)
    assert model_params == {"C": 1.0}
    assert weights == {"Performant": 7.0, "Passable": 1.0, "Energivore": 3.0}
    assert class_weights_from_params({})["Performant"] == 1.0

    y = ["Performant", "Passable", "Energivore", "Passable"]
    assert make_sample_weight(y, weights).tolist() == [7.0, 1.0, 3.0, 1.0]
    assert make_sample_weight(y, None).tolist() == [1.0] * 4
    balanced = make_sample_weight(y, "balanced")
    assert balanced[0] == pytest.approx(4 / 3)
    assert balanced[1] == pytest.approx(4 / 6)


def test_search_spaces_include_historical_class_weights():
    trial = FakeTrial()
    params = get_spec("random_forest").suggest(trial)
    assert params["weight_Performant"] == 1
    assert params["weight_Energivore"] == 1
    assert {"n_estimators", "max_depth", "max_features",
            "min_samples_leaf"} <= set(params)


@pytest.mark.parametrize("name", sorted(MODEL_SPECS))
def test_each_model_fits_and_predicts_labels(name):
    spec = get_spec(name)
    if not spec.is_available():
        pytest.skip(f"dépendance absente : {spec.requires}")
    X, y = small_dataset()
    trial = FakeTrial()
    params = spec.suggest(trial)
    # Accélère les modèles d'ensemble pour le test.
    for key in ("n_estimators", "iterations"):
        if key in params:
            params[key] = 20
    pipeline = fit_pipeline(spec, params, X, y)
    pred = pipeline.predict(X.head(10))
    assert set(pred) <= {"Performant", "Passable", "Energivore"}
    proba_capable = hasattr(pipeline.named_steps["clf"], "predict_proba")
    if proba_capable and name != "svm_rbf":
        proba = pipeline.predict_proba(X.head(5))
        assert proba.shape == (5, 3)


def test_label_encoded_classifier_round_trip():
    from sklearn.tree import DecisionTreeClassifier

    X, y = small_dataset()
    clf = LabelEncodedClassifier(DecisionTreeClassifier(random_state=0))
    clf.fit(X[["volume_stockage_generateur_n1_ecs_n1"]], y)
    assert list(clf.classes_) == ["Energivore", "Passable", "Performant"]
    pred = clf.predict(X[["volume_stockage_generateur_n1_ecs_n1"]])
    assert pred.dtype.kind in "UO"


def test_svm_kernel_screening_configurations_fit():
    from dpe.models import SVM_KERNEL_SCREENING

    X, y = small_dataset()
    spec = get_spec("svm_rbf")
    assert set(SVM_KERNEL_SCREENING) == {"Linéaire", "Polynomial (degré 3)", "RBF"}
    for params in SVM_KERNEL_SCREENING.values():
        pipeline = fit_pipeline(spec, {**params, "class_weight": "balanced"}, X, y)
        clf = pipeline.named_steps["clf"]
        assert clf.kernel == params["kernel"]
        assert clf.n_support_.sum() > 0
