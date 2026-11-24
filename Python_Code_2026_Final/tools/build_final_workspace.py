#!/usr/bin/env python3
"""Reconstruye los notebooks finales a partir de Python_Code_2026_New.

El script copia únicamente las versiones seleccionadas, limpia las salidas
embebidas y sustituye las rutas antiguas por la estructura definida en
project_paths.py. No copia pickles, CSV ni figuras generadas.
"""

from __future__ import annotations

import json
import re
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path


FINAL_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = FINAL_DIR.parent / "Python_Code_2026_New"


@dataclass(frozen=True)
class NotebookSpec:
    filename: str
    role: str
    experiment_name: str | None = None
    final_filename: str | None = None


NOTEBOOKS = (
    NotebookSpec("gen_nature_run.ipynb", "nature", final_filename="1_gen_nature_run.ipynb"),
    NotebookSpec("gen_obs.ipynb", "observations", final_filename="2_gen_obs.ipynb"),
    NotebookSpec("ana_ini.ipynb", "analysis", final_filename="3_ana_ini.ipynb"),
    NotebookSpec("CLE_NATURE.ipynb", "cle_nature", final_filename="4_CLE_NATURE.ipynb"),
    NotebookSpec("run_osse.ipynb", "osse", final_filename="5_run_osse.ipynb"),
    NotebookSpec(
        "run_cse_qiwen_xyz_control_github.ipynb",
        "cse",
        "xyz_control_nature_qiwenevents",
        "6_run_cse_qiwen_xyz_control_nature_qiwenevents.ipynb",
    ),
    NotebookSpec(
        "run_cse_qiwen_z_control_nature_qiwenevents.ipynb",
        "cse",
        "z_control_nature_qiwenevents",
        "6_run_cse_qiwen_z_control_nature_qiwenevents.ipynb",
    ),
)

CLASS_FILES = (
    "class_Lorenz63.py",
    "class_state_vec.py",
    "class_obs.py",
    "class_da_sys.py",
    "class_cse.py",
    "class_osse.py",
    "class_cse_qiwen.py",
)

OLD_FIGURES_PATH = str(SOURCE_DIR / "Python_Figures") + "/"


def setup_cells(spec: NotebookSpec) -> list[dict]:
    notice = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Versión final organizada\n",
            "\n",
            "Esta copia escribe sus salidas bajo `outputs/`. Las rutas comunes "
            "se definen en `project_paths.py`.\n",
        ],
    }

    imports_by_role = {
        "nature": (
            "NATURE_DATA_DIR, NATURE_FIGURES_DIR, ensure_output_directories"
        ),
        "observations": (
            "NATURE_DATA_DIR, OBSERVATIONS_DATA_DIR, "
            "OBSERVATIONS_FIGURES_DIR, ensure_output_directories"
        ),
        "analysis": (
            "NATURE_DATA_DIR, OBSERVATIONS_DATA_DIR, ANALYSIS_DATA_DIR, "
            "ANALYSIS_FIGURES_DIR, ensure_output_directories"
        ),
        "osse": (
            "PROJECT_ROOT, ANALYSIS_DATA_DIR, OSSE_DATA_DIR, OSSE_RESULTS_DIR, "
            "OSSE_FIGURES_DIR, ensure_output_directories"
        ),
        "cle_nature": (
            "ANALYSIS_DATA_DIR, CLE_NATURE_DATA_DIR, "
            "CLE_NATURE_FIGURES_DIR, ensure_output_directories"
        ),
    }

    if spec.role == "cse":
        source = (
            "from project_paths import (\n"
            "    PROJECT_ROOT, ANALYSIS_DATA_DIR, OSSE_DATA_DIR,\n"
            "    cse_experiment_paths, ensure_output_directories,\n"
            ")\n"
            "ensure_output_directories()\n"
            f"EXPERIMENT_NAME = {spec.experiment_name!r}\n"
            "EXPERIMENT_ROOT, RESULTS_DIR, EXPERIMENT_FIGURES_DIR = "
            "cse_experiment_paths(EXPERIMENT_NAME)\n"
            "RESULTS_DIR.mkdir(parents=True, exist_ok=True)\n"
            "EXPERIMENT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)\n"
        )
    else:
        source = (
            f"from project_paths import {imports_by_role[spec.role]}\n"
            "ensure_output_directories()\n"
        )

    setup = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }
    return [notice, setup]


def transform_nature(text: str) -> str:
    text = text.replace(OLD_FIGURES_PATH, "{NATURE_FIGURES_DIR}/")
    text = text.replace(
        "PathFigures = '{NATURE_FIGURES_DIR}/'",
        "PathFigures = str(NATURE_FIGURES_DIR) + '/'",
    )
    text = text.replace(
        "outfile = name+'.pkl'",
        "outfile = str(NATURE_DATA_DIR / (name + '.pkl'))",
    )
    return text


def transform_observations(text: str) -> str:
    text = text.replace("gen_nature_run.ipynb", "1_gen_nature_run.ipynb")
    text = text.replace(OLD_FIGURES_PATH, "{OBSERVATIONS_FIGURES_DIR}/")
    text = text.replace(
        "PathFigures = '{OBSERVATIONS_FIGURES_DIR}/'",
        "PathFigures = str(OBSERVATIONS_FIGURES_DIR) + '/'",
    )
    text = text.replace(
        "infile      = 'lorenz63_run.pkl'",
        "infile = str(NATURE_DATA_DIR / 'lorenz63_run.pkl')",
    )
    text = text.replace(
        "sv.setFigurePath('Python_Figures/')",
        "sv.setFigurePath(str(OBSERVATIONS_FIGURES_DIR) + '/')",
    )
    text = text.replace(
        "outfile = 'y_obs.pkl'",
        "outfile = str(OBSERVATIONS_DATA_DIR / 'y_obs.pkl')",
    )
    text = text.replace(
        "infile = 'y_obs.pkl'",
        "infile = str(OBSERVATIONS_DATA_DIR / 'y_obs.pkl')",
    )
    return text


def transform_analysis(text: str) -> str:
    text = text.replace(OLD_FIGURES_PATH, "{ANALYSIS_FIGURES_DIR}/")
    text = text.replace(
        "PathFigures = '{ANALYSIS_FIGURES_DIR}/'",
        "PathFigures = str(ANALYSIS_FIGURES_DIR) + '/'",
    )
    text = text.replace(
        "infile      = 'lorenz63_run.pkl'",
        "infile = str(NATURE_DATA_DIR / 'lorenz63_run.pkl')",
    )
    text = text.replace(
        "infile = 'y_obs.pkl'",
        "infile = str(OBSERVATIONS_DATA_DIR / 'y_obs.pkl')",
    )
    text = text.replace(
        "outfile = name+'.pkl'",
        "outfile = str(ANALYSIS_DATA_DIR / (name + '.pkl'))",
    )
    return text


def transform_osse(text: str) -> str:
    text = text.replace(OLD_FIGURES_PATH + "OSSE/", "{OSSE_FIGURES_DIR}/")
    text = text.replace(OLD_FIGURES_PATH, "{OSSE_FIGURES_DIR}/")
    text = text.replace(
        "PathFigures = '{OSSE_FIGURES_DIR}/'",
        "PathFigures = str(OSSE_FIGURES_DIR) + '/'",
    )
    text = text.replace(
        "infile = name + '_init.pkl'",
        "infile = str(ANALYSIS_DATA_DIR / (name + '_init.pkl'))",
    )

    output_names = {
        "analysis_file": "osse_Xa_history.pkl",
        "prior_file": "osse_Xf_history.pkl",
        "summary_file": "osse_summary.pkl",
        "checkpoint_file": "osse_Xa_history_checkpoint.npy",
        "prior_checkpoint_file": "osse_Xf_history_checkpoint.npy",
        "checkpoint_summary_file": "osse_summary_checkpoint.pkl",
    }
    for key, filename in output_names.items():
        text = text.replace(
            f"'{key}': '{filename}'",
            f"'{key}': str(OSSE_DATA_DIR / '{filename}')",
        )

    text = text.replace(
        "summary_file='osse_summary.pkl'",
        "summary_file=str(OSSE_DATA_DIR / 'osse_summary.pkl')",
    )
    text = text.replace(
        "INFLATION_SWEEP_CSV_FILE = 'osse_inflation_sweep_table.csv'",
        "INFLATION_SWEEP_CSV_FILE = str(OSSE_RESULTS_DIR / 'osse_inflation_sweep_table.csv')",
    )
    text = text.replace(
        "INFLATION_SWEEP_CSV_FILE = 'osse_inflation_sweep_table_stable.csv'",
        "INFLATION_SWEEP_CSV_FILE = str(OSSE_RESULTS_DIR / 'osse_inflation_sweep_table_stable.csv')",
    )

    candidates_pattern = re.compile(
        r"BASE_PATH_CANDIDATES = \[.*?"
        r"if str\(BASE_PATH\) not in sys\.path:\n"
        r"    sys\.path\.insert\(0, str\(BASE_PATH\)\)",
        flags=re.DOTALL,
    )
    text = candidates_pattern.sub(
        "BASE_PATH = ANALYSIS_DATA_DIR\n"
        "if str(PROJECT_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT))",
        text,
    )
    text = text.replace(
        "PathFigures = globals().get('PathFigures', str(BASE_PATH / 'Python_Figures' / 'OSSE') + '/')",
        "PathFigures = globals().get('PathFigures', str(OSSE_FIGURES_DIR) + '/')",
    )

    text = text.replace(
        "        diagnostics=diagnostics,\n",
        "        diagnostics=diagnostics,\n"
        "        truth_file=str(OSSE_DATA_DIR / 'osse_truth_full.pkl'),\n"
        "        truth_da_file=str(OSSE_DATA_DIR / 'osse_truth_da_times.pkl'),\n"
        "        obs_file=str(OSSE_DATA_DIR / 'osse_yobs.pkl'),\n"
        "        summary_file=str(OSSE_DATA_DIR / 'osse_summary.pkl'),\n"
        "        results_file=str(OSSE_DATA_DIR / 'osse_results.pkl'),\n"
        "        analysis_file=str(OSSE_DATA_DIR / 'osse_analysis_history.pkl'),\n"
        "        xa_history_file=str(OSSE_DATA_DIR / 'osse_Xa_history.pkl'),\n"
        "        prior_file=str(OSSE_DATA_DIR / 'osse_Xf_history.pkl'),\n",
    )
    return text


def transform_cle_nature(text: str) -> str:
    text = text.replace(
        OLD_FIGURES_PATH,
        "{CLE_NATURE_FIGURES_DIR}/",
    )
    text = text.replace(
        "PathFigures = '{CLE_NATURE_FIGURES_DIR}/'",
        "PathFigures = str(CLE_NATURE_FIGURES_DIR)",
    )
    text = text.replace(
        "infile   = name + '_init.pkl'",
        "infile = str(ANALYSIS_DATA_DIR / (name + '_init.pkl'))",
    )
    text = text.replace(
        "cache_filename = f'distances_ensemble_nature_",
        "cache_filename = CLE_NATURE_DATA_DIR / f'distances_ensemble_nature_",
    )
    text = text.replace(
        "cache_filename_renorm = (",
        "cache_filename_renorm = CLE_NATURE_DATA_DIR / (",
    )
    return text


def transform_cse(text: str, experiment_name: str) -> str:
    old_figure_names = {
        "xyz_control_nature_qiwenevents": "CSE_Qiwen_xyz_control_nature_qiwenevents",
        "z_control_nature_qiwenevents": "CSE_Qiwen_z_control_nature_qiwenevents",
    }
    old_figure_name = old_figure_names[experiment_name]
    text = text.replace(
        f"BASE_PATH = Path.cwd()\nPathFigures = BASE_PATH / '{old_figure_name}'\nPathFigures.mkdir(parents=True, exist_ok=True)",
        "BASE_PATH = PROJECT_ROOT\n"
        "PathFigures = EXPERIMENT_FIGURES_DIR\n"
        "PathFigures.mkdir(parents=True, exist_ok=True)",
    )
    text = text.replace(
        f"PathFigures = globals().get('PathFigures', BASE_PATH / '{old_figure_name}')",
        "PathFigures = globals().get('PathFigures', EXPERIMENT_FIGURES_DIR)",
    )
    text = text.replace(
        "da_system().load(BASE_PATH / 'x_analysis_init.pkl')",
        "da_system().load(ANALYSIS_DATA_DIR / 'x_analysis_init.pkl')",
    )
    text = text.replace(
        "with open(BASE_PATH / 'osse_Xa_history.pkl', 'rb')",
        "with open(OSSE_DATA_DIR / 'osse_Xa_history.pkl', 'rb')",
    )
    text = text.replace(
        "with open(BASE_PATH / 'osse_summary.pkl', 'rb')",
        "with open(OSSE_DATA_DIR / 'osse_summary.pkl', 'rb')",
    )
    text = text.replace(
        "    x_nature,\n    BASE_PATH,\n    position=position,",
        "    x_nature,\n    RESULTS_DIR,\n    position=position,",
    )

    # Pickles, CSV y tablas del experimento dejan de escribirse en la raíz.
    output_prefixes = (
        "cse_qiwen",
        "change_region_time_qiwen",
        "qiwen_fig3",
    )
    for prefix in output_prefixes:
        text = text.replace(
            f"BASE_PATH / f'{prefix}",
            f"RESULTS_DIR / f'{prefix}",
        )
        text = text.replace(
            f"BASE_PATH / '{prefix}",
            f"RESULTS_DIR / '{prefix}",
        )
    text = text.replace(
        "my_success_by_D = _load_my_success_by_D(BASE_PATH, comparison_D_values)",
        "my_success_by_D = _load_my_success_by_D(RESULTS_DIR, comparison_D_values)",
    )
    return text


def transform_source(text: str, spec: NotebookSpec) -> str:
    if spec.role == "nature":
        return transform_nature(text)
    if spec.role == "observations":
        return transform_observations(text)
    if spec.role == "analysis":
        return transform_analysis(text)
    if spec.role == "osse":
        return transform_osse(text)
    if spec.role == "cle_nature":
        return transform_cle_nature(text)
    if spec.role == "cse" and spec.experiment_name:
        return transform_cse(text, spec.experiment_name)
    raise ValueError(f"Unknown notebook role: {spec.role}")


def build_notebook(spec: NotebookSpec) -> None:
    source_path = SOURCE_DIR / spec.filename
    destination_path = FINAL_DIR / (spec.final_filename or spec.filename)
    with source_path.open("r", encoding="utf-8") as stream:
        notebook = json.load(stream)

    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        transformed = transform_source(source, spec)
        cell["source"] = transformed.splitlines(keepends=True)
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    notebook["cells"] = setup_cells(spec) + notebook.get("cells", [])
    notebook.get("metadata", {}).pop("widgets", None)
    with destination_path.open("w", encoding="utf-8") as stream:
        json.dump(notebook, stream, ensure_ascii=False, indent=1)
        stream.write("\n")


def split_xyz_control_batch_notebook() -> None:
    """Separate the xyz single-event diagnostics from its batch experiments."""
    single_path = FINAL_DIR / "6_run_cse_qiwen_xyz_control_nature_qiwenevents.ipynb"
    batch_path = FINAL_DIR / "7_run_cse_qiwen_xyz_control_nature_qiwenevents_batch.ipynb"
    with single_path.open("r", encoding="utf-8") as stream:
        notebook = json.load(stream)

    cells = notebook["cells"]
    by_id = {cell.get("id"): cell for cell in cells if cell.get("id")}
    required_ids = (
        "md-imports-load-das", "imports-load-das",
        "md-load-das-setup", "load-das-setup",
        "md-load-osse-history", "load-osse-history",
        "md-detect-regime-changes", "detect-regime-changes",
        "cse-parameters",
        "md-batch-run-40-events", "batch-run-40-events",
        "md-batch-run-fig3g-grid", "batch-run-fig3g-grid",
        "md-fig3g-grid-summary", "fig3g-grid-summary",
        "plot-3d-controlled-vs-free-nr",
    )
    missing = [cell_id for cell_id in required_ids if cell_id not in by_id]
    if missing:
        raise RuntimeError(f"Cannot split xyz-control notebook; missing cells: {missing}")

    def markdown(cell_id: str, source: str) -> dict:
        return {
            "cell_type": "markdown",
            "id": cell_id,
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }

    def code(cell_id: str, source: str) -> dict:
        return {
            "cell_type": "code",
            "execution_count": None,
            "id": cell_id,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        }

    wrapper_source = """# Thin wrapper around the single authoritative implementation.
# The complete result dictionary is returned unchanged so the batch files retain
# both summary metrics and all diagnostic histories produced by the core runner.
def run_single_cse(event_index, D_run, seed, regime_reference='nature', candidate_pool='full'):
    return run_single_cse_core(
        event_index=event_index,
        D_run=D_run,
        seed=seed,
        das=das,
        x_nature=x_nature,
        t_start_series=t_start_series,
        osse_Xa_history=osse_Xa_history,
        osse_summary=osse_summary,
        experiment_step=experiment_step,
        position=position,
        Ta=Ta,
        fcst_dt=fcst_dt,
        path_figures=PathFigures,
        total_events=40,
        regime_reference=regime_reference,
        candidate_pool=candidate_pool,
        control_indices=CONTROL_INDICES,
    )
"""

    # The first two cells are the final-workspace notice and generated path setup.
    batch_cells = [
        markdown(
            "batch-organized-version",
            "## Versión final organizada\n\n"
            "Este notebook contiene los experimentos CSE por lotes y reutiliza "
            "las rutas reanudables de `outputs/`.\n",
        ),
        deepcopy(cells[1]),
        markdown(
            "title-run-cse-qiwen-batch",
            "# Qiwen CSE xyz control — Batch experiments\n",
        ),
        markdown(
            "batch-purpose",
            "Ejecuta el batch de 40 eventos, la rejilla T×D de la Figura 3(a) "
            "y su resumen. La implementación científica permanece centralizada "
            "en `class_cse_qiwen.py`.\n",
        ),
        *(
            deepcopy(by_id[cell_id])
            for cell_id in (
                "md-imports-load-das", "imports-load-das",
                "md-load-das-setup", "load-das-setup",
                "md-load-osse-history", "load-osse-history",
                "md-detect-regime-changes", "detect-regime-changes",
            )
        ),
        markdown(
            "md-batch-parameters",
            "## Batch Parameters\n\n"
            "Se conservan los mismos parámetros del experimento individual.\n",
        ),
        deepcopy(by_id["cse-parameters"]),
        markdown(
            "md-batch-single-wrapper",
            "## Reusable Single-Event Runner\n\n"
            "Esta envoltura no modifica ni reduce el resultado devuelto.\n",
        ),
        code("batch-single-wrapper", wrapper_source),
        markdown(
            "md-batch-diagnostics",
            "## Diagnostics Preserved\n\n"
            "Cada nueva realización se guarda completa en su archivo pickle, "
            "incluidos los historiales diagnósticos disponibles.\n",
        ),
        *(
            deepcopy(by_id[cell_id])
            for cell_id in (
                "md-batch-run-40-events", "batch-run-40-events",
                "md-batch-run-fig3g-grid", "batch-run-fig3g-grid",
                "md-fig3g-grid-summary", "fig3g-grid-summary",
            )
        ),
    ]
    batch_notebook = {
        "cells": batch_cells,
        "metadata": deepcopy(notebook["metadata"]),
        "nbformat": notebook["nbformat"],
        "nbformat_minor": notebook["nbformat_minor"],
    }

    final_single_index = next(
        index
        for index, cell in enumerate(cells)
        if cell.get("id") == "plot-3d-controlled-vs-free-nr"
    )
    notebook["cells"] = cells[:final_single_index + 1]

    for path, output_notebook in (
        (single_path, notebook),
        (batch_path, batch_notebook),
    ):
        with path.open("w", encoding="utf-8") as stream:
            json.dump(output_notebook, stream, ensure_ascii=False, indent=1)
            stream.write("\n")


def main() -> None:
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR}")

    for filename in CLASS_FILES:
        shutil.copy2(SOURCE_DIR / filename, FINAL_DIR / filename)

    for spec in NOTEBOOKS:
        build_notebook(spec)

    split_xyz_control_batch_notebook()

    print(f"Final workspace rebuilt at: {FINAL_DIR}")
    print(
        f"Copied {len(CLASS_FILES)} class modules and built "
        f"{len(NOTEBOOKS) + 1} notebooks, including the derived xyz batch notebook."
    )


if __name__ == "__main__":
    main()
