"""Rutas compartidas por los notebooks finales del proyecto CSE–Qiwen."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

NATURE_ROOT = OUTPUT_ROOT / "01_nature_run"
NATURE_DATA_DIR = NATURE_ROOT / "data"
NATURE_FIGURES_DIR = NATURE_ROOT / "figures"

OBSERVATIONS_ROOT = OUTPUT_ROOT / "02_observations"
OBSERVATIONS_DATA_DIR = OBSERVATIONS_ROOT / "data"
OBSERVATIONS_FIGURES_DIR = OBSERVATIONS_ROOT / "figures"

ANALYSIS_ROOT = OUTPUT_ROOT / "03_analysis_initialization"
ANALYSIS_DATA_DIR = ANALYSIS_ROOT / "data"
ANALYSIS_FIGURES_DIR = ANALYSIS_ROOT / "figures"

OSSE_ROOT = OUTPUT_ROOT / "04_osse"
OSSE_DATA_DIR = OSSE_ROOT / "data"
OSSE_RESULTS_DIR = OSSE_ROOT / "results"
OSSE_FIGURES_DIR = OSSE_ROOT / "figures"

CSE_ROOT = OUTPUT_ROOT / "05_cse_qiwen"
CSE_EXPERIMENT_NAMES = (
    "xyz_control_nature_qiwenevents",
    "z_control_nature_qiwenevents",
)

CLE_ROOT = OUTPUT_ROOT / "06_cle"
CLE_NATURE_ROOT = CLE_ROOT / "nature"
CLE_NATURE_DATA_DIR = CLE_NATURE_ROOT / "data"
CLE_NATURE_FIGURES_DIR = CLE_NATURE_ROOT / "figures"
CLE_TARGET_ROOT = CLE_ROOT / "target"


def cse_experiment_paths(name):
    """Devuelve (raíz, resultados, figuras) para un experimento CSE."""
    root = CSE_ROOT / name
    return root, root / "results", root / "figures"


def cle_target_experiment_paths(name):
    """Devuelve (raíz, datos, figuras) para el CLE de un TARGET CSE."""
    root = CLE_TARGET_ROOT / name
    return root, root / "data", root / "figures"


def ensure_output_directories():
    """Crea toda la estructura de salidas sin escribir datos científicos."""
    directories = [
        NATURE_DATA_DIR,
        NATURE_FIGURES_DIR,
        OBSERVATIONS_DATA_DIR,
        OBSERVATIONS_FIGURES_DIR,
        ANALYSIS_DATA_DIR,
        ANALYSIS_FIGURES_DIR,
        OSSE_DATA_DIR,
        OSSE_RESULTS_DIR,
        OSSE_FIGURES_DIR,
        CSE_ROOT,
        CLE_NATURE_DATA_DIR,
        CLE_NATURE_FIGURES_DIR,
        CLE_TARGET_ROOT,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    for experiment_name in CSE_EXPERIMENT_NAMES:
        _, results_dir, figures_dir = cse_experiment_paths(experiment_name)
        results_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)


ensure_output_directories()
