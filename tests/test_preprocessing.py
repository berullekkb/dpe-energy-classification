import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression

from dpe.preprocessing import (
    split_features_target,
    make_model_pipeline
)


def test_pipeline_accepts_missing_and_unseen_categories():

    df = pd.DataFrame({
        "hauteur_sous_plafond": [
            2.3, 2.6, None, 2.8, 2.4, 2.9
        ],
        "energie": [
            "Gaz", "Bois", "Gaz", None, "Bois", "Gaz"
        ],
        "etiquette_dpe": [
            "Passable",
            "Performant",
            "Passable",
            "Performant",
            "Passable",
            "Performant"
        ]
    })

    X, y = split_features_target(df)

    model = make_model_pipeline(
        X,
        LogisticRegression(max_iter=500)
    )

    model.fit(X, y)

    new = pd.DataFrame({
        "hauteur_sous_plafond": [None],
        "energie": ["Electricite"]
    })

    pred = model.predict(new)

    assert len(pred) == 1
    assert pred[0] in set(y)


def test_target_is_not_a_feature():

    df = pd.DataFrame({
        "feature": [1, 2],
        "etiquette_dpe": ["Passable", "Energivore"]
    })

    X, y = split_features_target(df)

    assert "etiquette_dpe" not in X.columns
    assert len(y) == 2


def test_missing_target_raises():

    with pytest.raises(ValueError):
        split_features_target(
            pd.DataFrame({"feature": [1, 2]})
        )

# ============================================================
# Encodage cible historique (notebooks 03 à 08)
# ============================================================

import numpy as np  # noqa: E402

from dpe.preprocessing import OrdinalTargetEncoder, make_preprocessor  # noqa: E402


def target_dataset(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = pd.Series(rng.choice(["Performant", "Passable", "Energivore"], n,
                             p=[0.1, 0.6, 0.3]))
    cat = np.where(y == "Energivore", "Fioul", rng.choice(["Gaz", "Bois"], n))
    cat[:1] = "Unique"
    X = pd.DataFrame({"energie": cat, "type": rng.choice(["a", "b", "c"], n)})
    return X, y


def test_ordinal_target_encoder_formula():
    X, y = target_dataset()
    enc = OrdinalTargetEncoder(smoothing=3.0, min_samples_leaf=50).fit(X, y)
    codes = y.map({"Energivore": 0, "Passable": 1, "Performant": 2})
    prior = codes.mean()

    out = enc.transform(pd.DataFrame({"energie": ["Fioul", "Unique", "Jamais vu"],
                                      "type": ["a", "a", "a"]}))
    fioul = codes[X["energie"] == "Fioul"]
    lam = 1 / (1 + np.exp(-(len(fioul) - 50) / 3.0))
    assert out[0, 0] == pytest.approx(prior * (1 - lam) + fioul.mean() * lam)
    assert out[1, 0] == pytest.approx(prior)   # modalité vue une seule fois
    assert out[2, 0] == pytest.approx(prior)   # modalité inconnue
    # Fioul n'est associé qu'à Energivore : code proche de 0.
    assert out[0, 0] < 0.1


def test_ordinal_target_encoder_matches_category_encoders():
    ce = pytest.importorskip("category_encoders")
    X, y = target_dataset()
    ours = OrdinalTargetEncoder().fit(X, y).transform(X)
    ref = ce.TargetEncoder(smoothing=3.0, min_samples_leaf=50).fit(X, y)
    np.testing.assert_allclose(ours, ref.transform(X).to_numpy(), atol=1e-9)


def test_ordinal_target_encoder_out_of_fold():
    X, y = target_dataset()
    enc = OrdinalTargetEncoder(cv=5)
    oof = enc.fit_transform(X, y)
    full = enc.transform(X)
    assert oof.shape == full.shape
    assert not np.allclose(oof, full)


def test_target_preprocessor_keeps_19_like_columns():
    df = pd.DataFrame({
        "hauteur_sous_plafond": [2.5, 2.6, 2.4, 2.8],
        "qualite_isolation_murs_enc": [0, 1, 2, 3],
        "energie": ["Gaz", "Bois", "Gaz", "Fioul"],
        "etiquette_dpe": ["Passable", "Performant", "Passable", "Energivore"],
    })
    X, y = split_features_target(df)
    pre = make_preprocessor(X, encoding="target", scale=False)
    Z = pre.fit_transform(X, y)
    assert Z.shape == (4, 3)
    # Variable ordinale passée sans mise à l'échelle.
    assert sorted(Z[:, 1].tolist()) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        make_preprocessor(X, encoding="inconnu")
