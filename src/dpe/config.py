"""Configuration partagée du projet DPE."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "data_for_processing_final.xlsx"
)

TARGET = "etiquette_dpe"
GROUPS = ("Performant", "Passable", "Energivore")

RANDOM_STATE = 42
TEST_SIZE = 0.20