"""Graphiques communs (matplotlib, backend sans affichage)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import GROUPS  # noqa: E402

GROUP_COLORS = {
    "Performant": "#00923F",
    "Passable": "#F2C12B",
    "Energivore": "#E6442E",
}
ORIGIN_COLORS = {"historical": "#2a6f97", "new": "#e07a5f"}


def _save(fig, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion(confusion: pd.DataFrame, title: str, path: Path):
    """Matrice de confusion en effectifs et normalisée par ligne (rappel)."""
    confusion = confusion.loc[list(GROUPS), list(GROUPS)]
    normalized = confusion.div(confusion.sum(axis=1), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, data, fmt, sub in [
        (axes[0], confusion, "{:d}", "effectifs"),
        (axes[1], normalized, "{:.2f}", "normalisée par ligne"),
    ]:
        ax.imshow(normalized.to_numpy(), cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                value = data.iat[i, j]
                text = fmt.format(int(value) if fmt == "{:d}" else value)
                color = "white" if normalized.iat[i, j] > 0.6 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color)
        ax.set_xticks(range(3), GROUPS)
        ax.set_yticks(range(3), GROUPS)
        ax.set_xlabel("Classe prédite")
        ax.set_ylabel("Classe réelle")
        ax.set_title(sub)
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, path)


def plot_heatmap(values: np.ndarray, x_labels, y_labels, xlabel, ylabel,
                 title, path: Path, fmt="{:.3f}"):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(values, cmap="viridis", aspect="auto", origin="lower")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, fmt.format(values[i, j]), ha="center", va="center",
                    color="white" if values[i, j] < np.nanmean(values) else "black",
                    fontsize=8)
    ax.set_xticks(range(len(x_labels)), x_labels)
    ax.set_yticks(range(len(y_labels)), y_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="F1 macro (CV par groupes)")
    fig.tight_layout()
    return _save(fig, path)


def plot_bar_scores(df: pd.DataFrame, label_col: str, value_col: str,
                    title: str, path: Path, err_col=None, color_col=None,
                    xlabel="F1 macro"):
    """Barres horizontales triées (meilleur en haut)."""
    df = df.sort_values(value_col)
    colors = (
        df[color_col].map(ORIGIN_COLORS).fillna("#888").tolist()
        if color_col else "#2a6f97"
    )
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(df) + 1.5))
    ax.barh(df[label_col], df[value_col],
            xerr=df[err_col] if err_col else None, color=colors, capsize=3)
    for i, v in enumerate(df[value_col]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)
    lo = max(0.0, df[value_col].min() - 0.08)
    ax.set_xlim(lo, min(1.0, df[value_col].max() + 0.06))
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if color_col:
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ORIGIN_COLORS.values()]
        ax.legend(handles, ["historique", "nouveau"], loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def plot_grouped_bars(df: pd.DataFrame, label_col: str, value_cols: dict,
                      title: str, path: Path, ylabel="Score"):
    """Barres groupées (par ex. F1 par classe pour chaque modèle)."""
    labels = df[label_col].tolist()
    x = np.arange(len(labels))
    width = 0.8 / len(value_cols)
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(labels)), 4.8))
    for k, (col, (name, color)) in enumerate(value_cols.items()):
        ax.bar(x + (k - (len(value_cols) - 1) / 2) * width, df[col], width,
               label=name, color=color)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper center", ncol=len(value_cols), frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def plot_embedding(coords: np.ndarray, labels, title: str, path: Path,
                   highlight=None, highlight_label="Erreurs", axis_names=None):
    """Nuage 2D coloré par classe, avec mise en évidence optionnelle."""
    labels = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for grp in ["Passable", "Energivore", "Performant"]:
        mask = labels == grp
        ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.55,
                   color=GROUP_COLORS[grp], label=f"{grp} ({mask.sum()})")
    if highlight is not None:
        highlight = np.asarray(highlight, dtype=bool)
        ax.scatter(coords[highlight, 0], coords[highlight, 1], s=14,
                   facecolors="none", edgecolors="black", linewidths=0.5,
                   label=f"{highlight_label} ({highlight.sum()})")
    names = axis_names or ("Dimension 1", "Dimension 2")
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, markerscale=2)
    fig.tight_layout()
    return _save(fig, path)


def plot_decision_regions(coords, labels, predict_grid, title, path,
                          support=None, axis_names=None, resolution=300):
    """Frontière de décision d'un modèle entraîné dans un plan 2D."""
    x_min, x_max = coords[:, 0].min() - 0.5, coords[:, 0].max() + 0.5
    y_min, y_max = coords[:, 1].min() - 0.5, coords[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid_pred = np.asarray(predict_grid(np.c_[xx.ravel(), yy.ravel()]))
    codes = pd.Categorical(grid_pred, categories=list(GROUPS)).codes
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.contourf(xx, yy, codes.reshape(xx.shape), levels=[-0.5, 0.5, 1.5, 2.5],
                colors=[GROUP_COLORS[g] for g in GROUPS], alpha=0.22)
    labels = np.asarray(labels)
    for grp in ["Passable", "Energivore", "Performant"]:
        mask = labels == grp
        ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.6,
                   color=GROUP_COLORS[grp], label=grp)
    if support is not None:
        ax.scatter(support[:, 0], support[:, 1], s=18, facecolors="none",
                   edgecolors="black", linewidths=0.4,
                   label=f"Vecteurs de support ({len(support)})")
    names = axis_names or ("Dimension 1", "Dimension 2")
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, markerscale=2)
    fig.tight_layout()
    return _save(fig, path)
