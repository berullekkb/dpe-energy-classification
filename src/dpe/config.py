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
VALIDATION_SIZE = 0.20

# Variables numériques continues (mises à l'échelle pour les modèles
# sensibles aux distances, comme dans les notebooks historiques).
CONTINUOUS_FEATURES = (
    "hauteur_sous_plafond",
    "volume_stockage_generateur_n1_ecs_n1",
)

# Codage ordinal de la cible utilisé par le TargetEncoder historique
# (category_encoders trie les classes par ordre alphabétique, ce qui
# correspond ici à l'ordre de performance énergétique croissante).
TARGET_ORDINAL_CODES = {
    "Energivore": 0,
    "Passable": 1,
    "Performant": 2,
}

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
