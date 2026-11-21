"""Tests de l'optimisation par validation croisée stratifiée par groupes."""

import numpy as np
import pandas as pd
import pytest

from dpe.models import get_spec
from dpe.optimization import cross_validate_f1, optimize_model, trials_dataframe
from dpe.training import make_group_cv, make_profile_groups, stratified_subsample


def grouped_dataset(n_profiles=60, copies=3, seed=1):
    """Chaque profil est dupliqué `copies` fois, comme dans les données DPE."""
    rng = np.random.default_rng(seed)
    labels = np.array(["Performant", "Passable", "Energivore"])
    base_y = labels[np.arange(n_profiles) % 3]
    base = pd.DataFrame({
        "hauteur_sous_plafond": np.round(2.4 + rng.random(n_profiles), 2),
        "volume_stockage_generateur_n1_ecs_n1": np.arange(n_profiles, dtype=float),
        "type_batiment": rng.choice(["maison", "appartement"], n_profiles),
    })
    X = pd.concat([base] * copies, ignore_index=True)
    y = pd.Series(np.tile(base_y, copies))
    return X, y


def test_group_cv_never_splits_a_profile():
    X, y = grouped_dataset()
    groups = make_profile_groups(X)
    cv = make_group_cv(n_splits=3)
    for train_idx, val_idx in cv.split(X, y, groups):
        assert not set(groups[train_idx]) & set(groups[val_idx])


def test_cross_validate_returns_one_score_per_fold():
    X, y = grouped_dataset()
    groups = make_profile_groups(X)
    params = {"max_depth": 3, "min_samples_leaf": 1, "min_samples_split": 2,
              "weight_Performant": 2, "weight_Energivore": 1}
    scores = cross_validate_f1(get_spec("decision_tree"), params, X, y, groups,
                               n_splits=3)
    assert scores.shape == (3,)
    assert np.all((scores >= 0) & (scores <= 1))


def test_optimize_model_is_reproducible():
    X, y = grouped_dataset()
    groups = make_profile_groups(X)
    spec = get_spec("decision_tree")
    first = optimize_model(spec, X, y, groups, n_trials=4, n_splits=3,
                           verbose=False)
    second = optimize_model(spec, X, y, groups, n_trials=4, n_splits=3,
                            verbose=False)
    assert first.best_params == second.best_params
    assert first.best_value == pytest.approx(second.best_value)

    table = trials_dataframe(first)
    assert len(table) == 4
    assert "f1_macro_cv" in table.columns
    assert any(col.startswith("params_weight_") for col in table.columns)


def test_stratified_subsample_keeps_proportions():
    X, y = grouped_dataset(n_profiles=300, copies=1)
    Xs, ys = stratified_subsample(X, y, 90)
    assert len(Xs) == 90
    assert ys.value_counts().min() == 30
    Xall, _ = stratified_subsample(X, y, None)
    assert len(Xall) == len(X)
