"""Move the free-versus-controlled figure legend outside its data axes."""

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "15_run_cse_qiwen_z_control_z_obs_nature_qiwenevents.ipynb"
)
SECTION_TITLE = "## Free versus Controlled Nature Trajectories"

OLD_LAYOUT = """axes[0].legend(frameon=False, ncol=2)
fig.suptitle('z-only observations and z-only interventions: single-event outcome')
fig.tight_layout()"""

NEW_LAYOUT = """legend_handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(
    legend_handles,
    legend_labels,
    frameon=False,
    ncol=2,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.945),
)
fig.suptitle(
    'z-only observations and z-only interventions: single-event outcome',
    y=0.995,
)
fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))"""


notebook = nbf.read(NOTEBOOK_PATH, as_version=4)
section_index = next(
    index
    for index, cell in enumerate(notebook.cells)
    if cell.cell_type == "markdown" and SECTION_TITLE in cell.source
)
figure_cell = notebook.cells[section_index + 1]
if figure_cell.cell_type != "code":
    raise TypeError("The free-versus-controlled section is not followed by code.")

if OLD_LAYOUT in figure_cell.source:
    figure_cell.source = figure_cell.source.replace(OLD_LAYOUT, NEW_LAYOUT)
elif NEW_LAYOUT not in figure_cell.source:
    raise ValueError("The expected figure-layout code was not found.")

figure_cell.execution_count = None
figure_cell.outputs = []
nbf.write(notebook, NOTEBOOK_PATH)
print(f"Updated {NOTEBOOK_PATH}")
