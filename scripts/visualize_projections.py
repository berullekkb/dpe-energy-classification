"""Visualisations ACP et t-SNE (équivalent corrigé de 08_tsne.ipynb).

Corrections par rapport au notebook :
- le prétraitement (encodage cible + mise à l'échelle) est ajusté sur TRAIN
  uniquement ; les points projetés proviennent de la VALIDATION, que
  l'encodeur n'a jamais vue (pas de fuite de la cible dans la représentation) ;
- les erreurs affichées sont des erreurs de généralisation (validation), et
  non des erreurs sur des données d'entraînement ;
- l'ACP est ajustée sur TRAIN puis appliquée à la validation ; le t-SNE
  (sans fonction de projection) est calculé sur l'échantillon de validation.

Sorties (results/figures/) : pca_*.png, tsne_*.png, tsne_pairs_*.png
et results/metrics/projections_summary.json.

Usage :
    python scripts/visualize_projections.py [--n-sample 6000] [--perplexity 30]
"""

import argparse
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from dpe.benchmarking import read_json, write_json
from dpe.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, RANDOM_STATE
from dpe.models import get_spec
from dpe.preprocessing import make_preprocessor
from dpe.training import stratified_subsample
from dpe.visualization import plot_embedding

import optimize_models as om

PAIRS = [("Performant", "Passable"), ("Passable", "Energivore"),
         ("Performant", "Energivore")]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--n-sample", type=int, default=6000)
    parser.add_argument("--perplexity", type=float, default=30.0)
    args = parser.parse_args(argv)

    name = args.model or read_json(METRICS_DIR / "selected_model.json")["selected_model"]
    spec = get_spec(name)
    model = joblib.load(MODELS_DIR / f"{name}.joblib")

    X_train, y_train, X_val, y_val, _ = om.load_splits()

    # Représentation commune : encodage cible historique + RobustScaler,
    # ajustés sur TRAIN uniquement.
    pre = make_preprocessor(X_train, encoding="target", scale=True)
    Z_train = pre.fit_transform(X_train, y_train)
    # Standardisation globale pour que chaque variable pèse autant dans les
    # distances (ACP / t-SNE), paramètres estimés sur TRAIN.
    mu, sd = Z_train.mean(axis=0), Z_train.std(axis=0)
    sd[sd == 0] = 1.0
    Z_train = (Z_train - mu) / sd

    X_s, y_s = stratified_subsample(X_val, y_val, args.n_sample, RANDOM_STATE)
    Z_s = (pre.transform(X_s) - mu) / sd
    y_true = y_s.to_numpy()
    y_pred = model.predict(X_s)
    errors = y_true != y_pred

    summary = {"model": name, "n_points": int(len(X_s)),
               "points_from": "validation", "error_rate_sample": float(errors.mean())}

    # ACP
    pca = PCA(n_components=10, random_state=RANDOM_STATE).fit(Z_train)
    P = pca.transform(Z_s)
    var = pca.explained_variance_ratio_
    summary["pca_explained_variance_ratio"] = var.round(4).tolist()
    names = (f"CP1 ({var[0]:.1%})", f"CP2 ({var[1]:.1%})")
    plot_embedding(P[:, :2], y_true, "ACP (ajustée sur train) - validation, vraie classe",
                   FIGURES_DIR / "pca_validation_true.png", axis_names=names)
    plot_embedding(P[:, :2], y_pred, f"ACP - validation, prédiction {spec.label}",
                   FIGURES_DIR / "pca_validation_predicted.png", axis_names=names)
    plot_embedding(P[:, :2], y_true, f"ACP - erreurs de {spec.label} (validation)",
                   FIGURES_DIR / "pca_validation_errors.png", highlight=errors,
                   axis_names=names)
    loadings = dict(zip(pre.get_feature_names_out(), pca.components_[0].round(3)))
    summary["pca_cp1_top_loadings"] = dict(
        sorted(loadings.items(), key=lambda kv: -abs(kv[1]))[:6])

    # t-SNE
    t0 = time.time()
    tsne = TSNE(n_components=2, perplexity=args.perplexity, init="pca",
                random_state=RANDOM_STATE)
    T = tsne.fit_transform(Z_s)
    summary["tsne_seconds"] = round(time.time() - t0, 1)
    summary["tsne_perplexity"] = args.perplexity
    summary["tsne_kl_divergence"] = float(tsne.kl_divergence_)
    names = ("t-SNE 1", "t-SNE 2")
    pd.DataFrame({
        "row_id": X_s.index, "tsne_1": T[:, 0].round(4), "tsne_2": T[:, 1].round(4),
        "true": y_true, "predicted": y_pred,
    }).to_csv(METRICS_DIR / "tsne_validation_points.csv", index=False)
    plot_embedding(T, y_true, "t-SNE - validation, vraie classe",
                   FIGURES_DIR / "tsne_validation_true.png", axis_names=names)
    plot_embedding(T, y_pred, f"t-SNE - validation, prédiction {spec.label}",
                   FIGURES_DIR / "tsne_validation_predicted.png", axis_names=names)
    plot_embedding(T, y_true, f"t-SNE - erreurs de {spec.label} (validation)",
                   FIGURES_DIR / "tsne_validation_errors.png", highlight=errors,
                   axis_names=names)

    for a, b in PAIRS:
        mask = np.isin(y_true, [a, b])
        slug = f"{a}_{b}".lower()
        plot_embedding(T[mask], y_true[mask], f"t-SNE - {a} vs {b} (validation)",
                       FIGURES_DIR / f"tsne_pairs_{slug}.png",
                       highlight=errors[mask], axis_names=names)

    # Séparabilité locale : part des 10 plus proches voisins (dans l'espace
    # standardisé) qui partagent la classe du point.
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=11).fit(Z_s)
    idx = nn.kneighbors(Z_s, return_distance=False)[:, 1:]
    same = (y_true[idx] == y_true[:, None]).mean(axis=1)
    summary["knn10_same_class_by_class"] = {
        g: float(same[y_true == g].mean()) for g in np.unique(y_true)
    }

    write_json(METRICS_DIR / "projections_summary.json", summary)
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
