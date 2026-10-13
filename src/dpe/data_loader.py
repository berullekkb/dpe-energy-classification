"""Lecture et validation du jeu de données DPE préparé."""

from pathlib import Path
import pandas as pd

from .config import PROCESSED_DATA, TARGET, GROUPS


def load_processed_data(
    path: str | Path = PROCESSED_DATA
) -> pd.DataFrame:

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Données introuvables : {path}"
        )

    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)

    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)

    else:
        raise ValueError("Format attendu : .xlsx ou .csv")

    if TARGET not in df.columns:
        raise ValueError(
            f"Colonne cible absente : {TARGET}"
        )

    if df.empty:
        raise ValueError("Le jeu de données est vide.")

    invalid = (
        set(df[TARGET].dropna().unique())
        - set(GROUPS)
    )

    if invalid:
        raise ValueError(
            f"Classes inattendues : {sorted(map(str, invalid))}"
        )

    if df[TARGET].isna().any():
        raise ValueError(
            "La cible contient des valeurs manquantes."
        )

    if df[TARGET].nunique() < 2:
        raise ValueError(
            "Au moins deux classes sont nécessaires."
        )

    return df