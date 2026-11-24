"""Insert or refresh the z-only ensemble-symmetry figure in notebook 15."""

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "15_run_cse_qiwen_z_control_z_obs_nature_qiwenevents.ipynb"
)
SECTION_TITLE = "## z-Only Analysis Ensemble and Lorenz Symmetry"
NEXT_SECTION_TITLE = "## Free versus Controlled Nature Trajectories"


markdown_source = r"""
## z-Only Analysis Ensemble and Lorenz Symmetry

The preceding trajectory figures show the controlled truth, not the DA ensemble. This diagnostic compares the analysis ensemble at the beginning of every CSE cycle with both the controlled state and its Lorenz-symmetric counterpart,

\[
(x,y,z) \longrightarrow (-x,-y,z).
\]

If observations of $z$ cannot identify the active lobe, the ensemble may remain close to the symmetry-related state even while the controlled truth stays in the desired regime.
""".strip() + "\n"


code_source = r"""
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np

from project_paths import CSE_ROOT

symmetry_experiment_root = CSE_ROOT / 'z_control_z_obs_nature_qiwenevents'
FIGURES_DIR = symmetry_experiment_root / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

if 'single_cse_result' not in globals():
    event_for_plot = int(globals().get('EVENT_INDEX', 1))
    D_for_plot = float(globals().get('D', 0.05))
    D_tag_for_plot = f'{D_for_plot:g}'.replace('.', 'p')
    saved_result_path = symmetry_experiment_root / 'results' / (
        f'cse_z_control_z_obs_event_{event_for_plot}_D{D_tag_for_plot}_qiwenevents.pkl'
    )
    with saved_result_path.open('rb') as stream:
        single_cse_result = pickle.load(stream)

Ta = int(single_cse_result['cse_summary']['Ta'])
analysis_ensembles = np.asarray(
    single_cse_result['Xa_cycle_initial_history'],
    dtype=float,
)
n_analysis_cycles = analysis_ensembles.shape[0]
analysis_steps = np.arange(n_analysis_cycles) * Ta
controlled_at_analysis = np.asarray(
    single_cse_result['evo_history'],
    dtype=float,
)[:, analysis_steps].T
analysis_mean = analysis_ensembles.mean(axis=2)

symmetry_sign = np.array([-1.0, -1.0, 1.0])
symmetry_counterpart = controlled_at_analysis * symmetry_sign

rmse_to_controlled = np.sqrt(
    np.mean((analysis_mean - controlled_at_analysis) ** 2, axis=1)
)
rmse_to_symmetry = np.sqrt(
    np.mean((analysis_mean - symmetry_counterpart) ** 2, axis=1)
)
x_sign_agreement = np.mean(
    np.sign(analysis_ensembles[:, 0, :])
    == np.sign(controlled_at_analysis[:, 0, None])
)

fig, axes = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True)
state_labels = ('x', 'y', 'z')

for state_index, ax in enumerate(axes):
    ax.plot(
        analysis_steps,
        controlled_at_analysis[:, state_index],
        color='tab:blue',
        linewidth=1.5,
        label='controlled state',
        zorder=4,
    )
    if state_index < 2:
        ax.plot(
            analysis_steps,
            symmetry_counterpart[:, state_index],
            color='0.35',
            linestyle=':',
            linewidth=1.3,
            label='symmetry counterpart',
            zorder=2,
        )
        ax.axhline(0.0, color='0.75', linewidth=0.8)
    for member in range(analysis_ensembles.shape[2]):
        ax.plot(
            analysis_steps,
            analysis_ensembles[:, state_index, member],
            color='lightcoral',
            linewidth=0.65,
            alpha=0.35,
            label='analysis members' if member == 0 else None,
            zorder=1,
        )
    ax.plot(
        analysis_steps,
        analysis_mean[:, state_index],
        color='darkred',
        linestyle='--',
        linewidth=1.2,
        label='analysis mean',
        zorder=3,
    )
    ax.set_ylabel(state_labels[state_index])
    ax.grid(alpha=0.2)

legend_handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(
    legend_handles,
    legend_labels,
    frameon=False,
    ncol=4,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.945),
)
axes[0].text(
    0.01,
    0.03,
    'x-sign agreement = {:.1f}%\nMean RMSE to controlled state = {:.2f}\nMean RMSE to symmetry counterpart = {:.2f}'.format(
        100.0 * x_sign_agreement,
        np.mean(rmse_to_controlled),
        np.mean(rmse_to_symmetry),
    ),
    transform=axes[0].transAxes,
    ha='left',
    va='bottom',
    fontsize=9,
    bbox={'facecolor': 'white', 'edgecolor': '0.75', 'alpha': 0.9},
)
axes[-1].set_xlabel('Steps after CSE initialization')
fig.suptitle(
    'z-only DA follows the symmetry-related Lorenz state',
    y=0.995,
)
fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

ensemble_symmetry_path = FIGURES_DIR / 'z_only_analysis_ensemble_symmetry_comparison.png'
fig.savefig(ensemble_symmetry_path, dpi=300, bbox_inches='tight')
plt.show()

print('x-sign agreement across all members and cycles:', x_sign_agreement)
print('Mean RMSE to controlled state:', float(np.mean(rmse_to_controlled)))
print('Mean RMSE to symmetry counterpart:', float(np.mean(rmse_to_symmetry)))
print('Figure:', ensemble_symmetry_path)
""".strip() + "\n"


notebook = nbf.read(NOTEBOOK_PATH, as_version=4)
new_cells = [
    nbf.v4.new_markdown_cell(markdown_source),
    nbf.v4.new_code_cell(code_source),
]

existing_index = next(
    (
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "markdown" and SECTION_TITLE in cell.source
    ),
    None,
)
if existing_index is not None:
    replacement_end = existing_index + 1
    if (
        replacement_end < len(notebook.cells)
        and notebook.cells[replacement_end].cell_type == "code"
    ):
        replacement_end += 1
    notebook.cells[existing_index:replacement_end] = new_cells
else:
    insertion_index = next(
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "markdown" and NEXT_SECTION_TITLE in cell.source
    )
    notebook.cells[insertion_index:insertion_index] = new_cells

nbf.write(notebook, NOTEBOOK_PATH)
print(f"Updated {NOTEBOOK_PATH}")
