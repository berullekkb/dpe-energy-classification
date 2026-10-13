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