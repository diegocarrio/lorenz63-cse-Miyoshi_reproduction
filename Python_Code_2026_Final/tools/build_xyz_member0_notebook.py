"""Build the isolated full-state xyz CSE experiment with member0 reference."""

import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "6_run_cse_qiwen_xyz_control_nature_qiwenevents.ipynb"
BATCH_SOURCE = ROOT / "7_run_cse_qiwen_xyz_control_nature_qiwenevents_batch.ipynb"
TARGET = ROOT / "20_run_cse_qiwen_xyz_control_member0_qiwenevents.ipynb"


with SOURCE.open("r", encoding="utf-8") as stream:
    notebook = deepcopy(json.load(stream))
with BATCH_SOURCE.open("r", encoding="utf-8") as stream:
    batch_notebook = json.load(stream)

notebook["cells"][0]["source"] = [
    "## Isolated full-state xyz CSE with member0 reference\n",
    "\n",
    "This notebook combines the single-event workflow from notebook 6 with the "
    "40-event batch and Figure 3(a) T-D grid from notebook 7. It assimilates "
    "and controls all three Lorenz-63 variables and uses `member0` throughout. "
    "Outputs are isolated under "
    "`outputs/05_cse_qiwen/xyz_control_member0_qiwenevents/`.\n",
]
notebook["cells"][2]["source"] = [
    "# Qiwen CSE xyz control with member0 regime reference\n"
]

replacements = {
    "xyz_control_nature_qiwenevents": "xyz_control_member0_qiwenevents",
    "REGIME_REFERENCE_MODE = 'nature'": "REGIME_REFERENCE_MODE = 'member0'",
    "regime_reference='nature'": "regime_reference='member0'",
}

for cell in notebook["cells"]:
    source = "".join(cell.get("source", []))
    for old, new in replacements.items():
        source = source.replace(old, new)
    cell["source"] = source.splitlines(keepends=True)

    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

    cell_text = "".join(cell.get("source", []))
    if "REGIME_REFERENCE_MODE = 'member0'" in cell_text:
        cell_text = cell_text.replace(
            "REGIME_REFERENCE_MODE = 'member0'",
            "# Match the original Qiwen fallback: use forecast member 0's x sign.\n"
            "REGIME_REFERENCE_MODE = 'member0'",
        )
        cell["source"] = cell_text.splitlines(keepends=True)

# An unsuccessful member0 pilot must remain inspectable and must not prevent
# the batch/grid cells from running when the notebook is executed in order.
single_event_cell = notebook["cells"][22]
single_event_source = "".join(single_event_cell["source"])
single_event_source = single_event_source.replace(
    "assert single_cse_result['success'], 'run_single_cse did not achieve control.'\n",
    "# A failed pilot is a valid scientific result; keep the batch runnable.\n",
)
single_event_cell["source"] = single_event_source.splitlines(keepends=True)

notebook["cells"].append({
    "cell_type": "markdown",
    "id": "member0-batch-intro",
    "metadata": {},
    "source": [
        "## Batch experiments and Figure 3(a) comparison\n",
        "\n",
        "The following cells use the same full-state OSSE and `member0` reference "
        "as the single-event run. They write to this notebook's isolated output "
        "directory. The 40-event sweep and the 5 × 9 × 40 grid are expensive; "
        "run them only when needed. The comparison is with Qiwen's xyz-control "
        "Figure 3(a), not the z-control Figure 3(g).\n",
    ],
})

# Define the runner independently of the single-event cell, so users can skip
# the pilot plots and still execute the batch cells from the setup above.
batch_runner = deepcopy(batch_notebook["cells"][15])
batch_runner["id"] = "member0-batch-runner"
runner_source = "".join(batch_runner["source"])
runner_source = (
    "import importlib\n"
    "import class_cse_qiwen as _cse_qiwen_module\n"
    "_cse_qiwen_module = importlib.reload(_cse_qiwen_module)\n"
    "run_single_cse_core = _cse_qiwen_module.run_single_cse\n\n"
    + runner_source.replace("regime_reference='nature'", "regime_reference='member0'")
)
batch_runner["source"] = runner_source.splitlines(keepends=True)
batch_runner["execution_count"] = None
batch_runner["outputs"] = []
notebook["cells"].append(batch_runner)

for source_index in range(16, 23):
    cell = deepcopy(batch_notebook["cells"][source_index])
    cell["id"] = f"member0-batch-{source_index}"
    source = "".join(cell["source"])
    source = source.replace("REGIME_REFERENCE = 'nature'", "REGIME_REFERENCE = 'member0'")
    source = source.replace("REGIME_REFERENCE', 'nature'", "REGIME_REFERENCE', 'member0'")
    source = source.replace("_nature_qiwenevents", "_member0_qiwenevents")
    source = source.replace("FIG3G", "FIG3A").replace("fig3g", "fig3a")
    if source_index == 22:
        source = source.replace(
            "return RESULTS_DIR / f'cse_qiwen_xyz_control_40_results_D{D_run}{OUTPUT_TAG}.pkl'",
            "output_suffix = (\n"
            "        '' if (FIG3A_REGIME_REFERENCE == 'nature' and FIG3A_CANDIDATE_POOL == 'full')\n"
            "        else f'_{FIG3A_REGIME_REFERENCE}_{FIG3A_CANDIDATE_POOL}'\n"
            "    )\n"
            "    return RESULTS_DIR / f'cse_qiwen_xyz_control_40_results_D{D_run}{output_suffix}{OUTPUT_TAG}.pkl'",
        )
        source = source.replace(
            "if not result_file.exists() and T_label == '[4T0]':",
            "if (not result_file.exists() and T_label == '[4T0]'\n"
            "        and int(np.ceil(T_0 * multiple_number)) == position_run):",
        )
    cell["source"] = source.splitlines(keepends=True)
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    notebook["cells"].append(cell)

assert "xyz_control_member0_qiwenevents" in "".join(notebook["cells"][1]["source"])
assert "REGIME_REFERENCE_MODE = 'member0'" in "".join(notebook["cells"][6]["source"])
assert "control_indices=None" in "".join(notebook["cells"][6]["source"])
assert "regime_reference='member0'" in "".join(notebook["cells"][22]["source"])
assert len(notebook["cells"]) == 36
assert "REGIME_REFERENCE = 'member0'" in "".join(notebook["cells"][31]["source"])
assert "FIG3A_REGIME_REFERENCE = 'member0'" in "".join(notebook["cells"][33]["source"])

with TARGET.open("w", encoding="utf-8") as stream:
    json.dump(notebook, stream, ensure_ascii=False, indent=1)
    stream.write("\n")

print(f"Built {TARGET}")
