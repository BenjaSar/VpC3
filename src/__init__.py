"""
CubiCasa5K Room Detection & Area Estimation
Sistema avanzado de visión por computadora para detección de habitaciones
"""

__version__ = "1.0.0"
__author__ = "Vision por Computadora 3 - Maestría"
__description__ = "Room detection and area estimation using Swin Transformer"

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# Crear directorios si no existen
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, MLRUNS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
