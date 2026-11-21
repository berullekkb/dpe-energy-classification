"""Registre des modèles DPE et de leurs espaces de recherche.

Chaque modèle est décrit par une ``ModelSpec`` :

- ``build(params)`` construit l'estimateur scikit-learn (sans prétraitement) ;
- ``suggest(trial)`` tire les hyperparamètres Optuna ;
- ``encoding`` / ``scale`` décrivent le prétraitement associé ;
- ``origin`` distingue les expériences reprises des notebooks
  (``historical``) des nouvelles expériences (``new``) ;
- ``tuning_sample`` limite la taille du train pendant l'optimisation
  (SVM à noyau : 8 000 lignes, comme dans le notebook 04).

Le déséquilibre est géré comme dans les notebooks : deux poids de classe
optimisés par Optuna (``weight_Performant`` ∈ [1, 25],
``weight_Energivore`` ∈ [1, 15], ``Passable`` = 1), appliqués via
``sample_weight`` pour que tous les modèles soient traités de la même façon.
"""

from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Callable

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ..config import GROUPS, RANDOM_STATE

WEIGHT_RANGES = {
    "weight_Performant": (1, 25),
    "weight_Energivore": (1, 15),
}

_SKLEARN_VERSION = tuple(int(p) for p in sklearn.__version__.split(".")[:2])


# ============================================================
# Poids de classe
# ============================================================

def class_weights_from_params(params: dict) -> dict:
    """Extrait le dictionnaire de poids de classe des paramètres Optuna."""
    return {
        "Performant": float(params.get("weight_Performant", 1)),
        "Passable": 1.0,
        "Energivore": float(params.get("weight_Energivore", 1)),
    }


def make_sample_weight(y, class_weight) -> np.ndarray:
    """Convertit un class_weight en vecteur de poids par observation.

    ``class_weight`` peut être un dictionnaire, ``'balanced'`` ou ``None``.
    """
    y = pd.Series(np.asarray(y))
    if class_weight is None:
        return np.ones(len(y))
    if class_weight == "balanced":
        counts = y.value_counts()
        weights = len(y) / (len(counts) * counts)
        return y.map(weights).to_numpy(dtype=float)
    return y.map(class_weight).fillna(1.0).to_numpy(dtype=float)


def split_model_params(params: dict):
    """Sépare les poids de classe des hyperparamètres de l'estimateur."""
    model_params = {
        k: v for k, v in params.items() if k not in WEIGHT_RANGES
    }
    return model_params, class_weights_from_params(params)


def suggest_class_weights(trial) -> None:
    for name, (low, high) in WEIGHT_RANGES.items():
        trial.suggest_int(name, low, high)


# ============================================================
# Adaptateur pour les bibliothèques qui exigent une cible entière
# ============================================================

class LabelEncodedClassifier(ClassifierMixin, BaseEstimator):
    """Encode la cible en entiers pour XGBoost / LightGBM / CatBoost.

    L'ordre des classes est fixé (ordre alphabétique, identique à
    ``LabelEncoder``) et les prédictions sont redécodées en libellés, ce qui
    permet d'utiliser ces modèles exactement comme les modèles scikit-learn.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None):
        self.label_encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder_.classes_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(
            X, self.label_encoder_.transform(y), sample_weight=sample_weight
        )
        return self

    def predict(self, X):
        pred = np.asarray(self.estimator_.predict(X)).reshape(-1).astype(int)
        return self.label_encoder_.inverse_transform(pred)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.estimator_.feature_importances_


# ============================================================
# Constructeurs
# ============================================================

def make_logistic(l1_ratio: float, solver: str, **params) -> LogisticRegression:
    """Régression logistique compatible scikit-learn < 1.8 et >= 1.8.

    Depuis scikit-learn 1.8, la pénalité est définie par ``l1_ratio`` seul
    (0 = Ridge, 1 = Lasso) et ``penalty`` est déprécié.
    """
    if _SKLEARN_VERSION >= (1, 8):
        return LogisticRegression(l1_ratio=l1_ratio, solver=solver, **params)
    if l1_ratio == 0:
        penalty = "l2"
    elif l1_ratio == 1:
        penalty = "l1"
    else:
        penalty = "elasticnet"
    extra = {"l1_ratio": l1_ratio} if penalty == "elasticnet" else {}
    return LogisticRegression(penalty=penalty, solver=solver, **extra, **params)


def _build_elasticnet(p):
    return make_logistic(
        l1_ratio=p["l1_ratio"], solver="saga", C=p["C"],
        max_iter=p.get("max_iter", 1500), tol=p.get("tol", 1e-4),
        random_state=RANDOM_STATE,
    )


def _build_ridge(p):
    return make_logistic(
        l1_ratio=0.0, solver="lbfgs", C=p["C"],
        max_iter=p.get("max_iter", 1500), random_state=RANDOM_STATE,
    )


def _build_lasso(p):
    return make_logistic(
        l1_ratio=1.0, solver="saga", C=p["C"],
        max_iter=p.get("max_iter", 200), tol=p.get("tol", 1e-3),
        random_state=RANDOM_STATE,
    )


def _build_svm(p):
    return SVC(
        kernel=p.get("kernel", "rbf"), C=p["C"], gamma=p.get("gamma", "scale"),
        degree=p.get("degree", 3), random_state=RANDOM_STATE,
        cache_size=1000,
    )


def _build_tree(p):
    return DecisionTreeClassifier(
        max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
        min_samples_split=p["min_samples_split"], random_state=RANDOM_STATE,
    )


def _build_rf(p):
    return RandomForestClassifier(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"],
        max_features=p["max_features"], min_samples_leaf=p["min_samples_leaf"],
        n_jobs=-1, random_state=RANDOM_STATE,
    )


def _build_ada(p):
    return AdaBoostClassifier(
        n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
        random_state=RANDOM_STATE,
    )


def _build_gb(p):
    return GradientBoostingClassifier(
        n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
        max_depth=p["max_depth"], subsample=p["subsample"],
        random_state=RANDOM_STATE,
    )


def _build_xgb(p):
    from xgboost import XGBClassifier

    return LabelEncodedClassifier(XGBClassifier(
        n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
        max_depth=p["max_depth"], subsample=p["subsample"],
        reg_lambda=p["reg_lambda"], objective="multi:softprob",
        eval_metric="mlogloss", tree_method="hist", n_jobs=-1,
        random_state=RANDOM_STATE,
    ))


def _build_lgbm(p):
    from lightgbm import LGBMClassifier

    return LabelEncodedClassifier(LGBMClassifier(
        n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
        num_leaves=p["num_leaves"], max_depth=p["max_depth"],
        min_child_samples=p["min_child_samples"],
        subsample=p["subsample"], subsample_freq=1,
        colsample_bytree=p["colsample_bytree"], reg_lambda=p["reg_lambda"],
        objective="multiclass", n_jobs=-1, verbose=-1,
        random_state=RANDOM_STATE,
    ))


def _build_catboost(p):
    from catboost import CatBoostClassifier

    return LabelEncodedClassifier(CatBoostClassifier(
        iterations=p["iterations"], learning_rate=p["learning_rate"],
        depth=p["depth"], l2_leaf_reg=p["l2_leaf_reg"],
        loss_function="MultiClass", thread_count=-1, verbose=False,
        allow_writing_files=False, random_seed=RANDOM_STATE,
    ))


# ============================================================
# Espaces de recherche (identiques aux notebooks pour `historical`)
# ============================================================

def _space_elasticnet(t):
    t.suggest_float("C", 1e-3, 100.0, log=True)
    t.suggest_float("l1_ratio", 0.0, 1.0)


def _space_ridge(t):
    t.suggest_float("C", 1e-3, 100.0, log=True)


def _space_lasso(t):
    t.suggest_float("C", 1e-2, 10.0, log=True)


def _space_svm(t):
    t.suggest_float("C", 1e-2, 100.0, log=True)
    t.suggest_float("gamma", 1e-4, 10.0, log=True)


def _space_tree(t):
    t.suggest_int("max_depth", 3, 30)
    t.suggest_int("min_samples_leaf", 1, 100)
    t.suggest_int("min_samples_split", 2, 100)


def _space_rf(t):
    t.suggest_int("n_estimators", 100, 500, step=50)
    t.suggest_int("max_depth", 5, 30)
    t.suggest_float("max_features", 0.1, 1.0)
    t.suggest_int("min_samples_leaf", 1, 50)


def _space_ada(t):
    t.suggest_int("n_estimators", 50, 500, step=50)
    t.suggest_float("learning_rate", 0.01, 2.0, log=True)


def _space_gb(t):
    t.suggest_int("n_estimators", 50, 400, step=50)
    t.suggest_float("learning_rate", 0.01, 0.5, log=True)
    t.suggest_int("max_depth", 2, 8)
    t.suggest_float("subsample", 0.5, 1.0)


def _space_xgb(t):
    _space_gb(t)
    t.suggest_float("reg_lambda", 1e-3, 10.0, log=True)


def _space_lgbm(t):
    t.suggest_int("n_estimators", 50, 600, step=50)
    t.suggest_float("learning_rate", 0.01, 0.3, log=True)
    t.suggest_int("num_leaves", 8, 128, log=True)
    t.suggest_int("max_depth", 3, 12)
    t.suggest_int("min_child_samples", 5, 100, log=True)
    t.suggest_float("subsample", 0.5, 1.0)
    t.suggest_float("colsample_bytree", 0.5, 1.0)
    t.suggest_float("reg_lambda", 1e-3, 10.0, log=True)


def _space_catboost(t):
    t.suggest_int("iterations", 100, 600, step=50)
    t.suggest_float("learning_rate", 0.02, 0.3, log=True)
    t.suggest_int("depth", 4, 8)
    t.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True)


# ============================================================
# Registre
# ============================================================

@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    family: str
    origin: str
    notebook: str
    build: Callable[[dict], object]
    space: Callable
    encoding: str = "target"
    scale: bool = False
    tuning_sample: int | None = None
    requires: tuple = field(default_factory=tuple)
    notes: str = ""

    def suggest(self, trial) -> dict:
        suggest_class_weights(trial)
        self.space(trial)
        return dict(trial.params)

    def make_estimator(self, params: dict):
        model_params, _ = split_model_params(params)
        return self.build(model_params)

    def is_available(self) -> bool:
        return all(find_spec(pkg) is not None for pkg in self.requires)


MODEL_SPECS = {
    spec.name: spec
    for spec in [
        ModelSpec("lr_elasticnet", "Logistique Elastic Net", "linear",
                  "historical", "03", _build_elasticnet, _space_elasticnet,
                  scale=True),
        ModelSpec("lr_ridge", "Logistique Ridge", "linear", "historical",
                  "03", _build_ridge, _space_ridge, scale=True),
        ModelSpec("lr_lasso", "Logistique Lasso", "linear", "historical",
                  "03", _build_lasso, _space_lasso, scale=True),
        ModelSpec("svm_rbf", "SVM (noyau RBF)", "svm", "historical", "04",
                  _build_svm, _space_svm, scale=True, tuning_sample=8000,
                  notes="Optimisation sur 8 000 lignes du train (notebook 04)."),
        ModelSpec("decision_tree", "Arbre de décision", "tree", "historical",
                  "05", _build_tree, _space_tree),
        ModelSpec("random_forest", "Random Forest", "tree", "historical",
                  "05", _build_rf, _space_rf),
        ModelSpec("adaboost", "AdaBoost", "boosting", "historical", "06",
                  _build_ada, _space_ada),
        ModelSpec("gradient_boosting", "Gradient Boosting", "boosting",
                  "historical", "06", _build_gb, _space_gb),
        ModelSpec("xgboost", "XGBoost", "boosting", "historical", "06",
                  _build_xgb, _space_xgb, requires=("xgboost",)),
        ModelSpec("lightgbm", "LightGBM", "boosting", "new", "-",
                  _build_lgbm, _space_lgbm, requires=("lightgbm",)),
        ModelSpec("catboost", "CatBoost", "boosting", "new", "-",
                  _build_catboost, _space_catboost, requires=("catboost",)),
    ]
}

FAMILIES = {
    "linear": ["lr_elasticnet", "lr_ridge", "lr_lasso"],
    "svm": ["svm_rbf"],
    "tree": ["decision_tree", "random_forest"],
    "boosting": ["adaboost", "gradient_boosting", "xgboost", "lightgbm",
                 "catboost"],
}

# Criblage des noyaux du notebook 04 (C = 1, poids 'balanced').
SVM_KERNEL_SCREENING = {
    "Linéaire": {"kernel": "linear", "C": 1.0},
    "Polynomial (degré 3)": {"kernel": "poly", "C": 1.0, "degree": 3},
    "RBF": {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
}


def get_spec(name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[name]
    except KeyError as exc:
        raise ValueError(
            f"Modèle inconnu : {name}. Choix : {sorted(MODEL_SPECS)}"
        ) from exc


def resolve_models(names=None, families=None) -> list[str]:
    """Liste ordonnée de modèles à partir de noms et/ou de familles."""
    selected = []
    for family in families or []:
        if family not in FAMILIES:
            raise ValueError(f"Famille inconnue : {family}. Choix : {sorted(FAMILIES)}")
        selected.extend(FAMILIES[family])
    for name in names or []:
        get_spec(name)
        selected.append(name)
    return list(dict.fromkeys(selected)) or list(MODEL_SPECS)


__all__ = [
    "GROUPS", "MODEL_SPECS", "FAMILIES", "SVM_KERNEL_SCREENING", "ModelSpec",
    "LabelEncodedClassifier", "class_weights_from_params", "get_spec",
    "make_logistic", "make_sample_weight", "resolve_models",
    "split_model_params",
]
