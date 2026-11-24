"""Insert or refresh the z-only analysis-ensemble 3D figure in notebook 15."""

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "15_run_cse_qiwen_z_control_z_obs_nature_qiwenevents.ipynb"
)
SECTION_TITLE = "## Three-Dimensional Free Trajectory versus z-Only Analysis Ensemble"
NEXT_SECTION_TITLE = "## Interpretation Boundary"


markdown_source = r"""
## Three-Dimensional Free Trajectory versus z-Only Analysis Ensemble

This companion figure replaces the controlled trajectory in the right-hand panel with the analysis ensemble at every DA-cycle initialization. It directly tests the Lorenz symmetry ambiguity: the free trajectory visits both lobes, whereas the z-only analyses are expected to populate the symmetry-related negative lobe.
""".strip() + "\n"


code_source = r"""
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np

from class_da_sys import da_system
from project_paths import ANALYSIS_DATA_DIR, CSE_ROOT

analysis_3d_experiment_root = CSE_ROOT / 'z_control_z_obs_nature_qiwenevents'
analysis_3d_figures_dir = analysis_3d_experiment_root / 'figures'
analysis_3d_figures_dir.mkdir(parents=True, exist_ok=True)

if 'single_cse_result' not in globals():
    event_for_plot = int(globals().get('EVENT_INDEX', 1))
    D_for_plot = float(globals().get('D', 0.05))
    D_tag_for_plot = f'{D_for_plot:g}'.replace('.', 'p')
    saved_result_path = analysis_3d_experiment_root / 'results' / (
        f'cse_z_control_z_obs_event_{event_for_plot}_D{D_tag_for_plot}_qiwenevents.pkl'
    )
    with saved_result_path.open('rb') as stream:
        single_cse_result = pickle.load(stream)

if 'x_nature' not in globals():
    z_only_analysis_file = (
        ANALYSIS_DATA_DIR.parent / 'z_only_obs' / 'data'
        / 'x_analysis_init_z_only_obs.pkl'
    )
    das_3d = da_system().load(z_only_analysis_file)
    x_nature = np.asarray(das_3d.getStateVector().getTrajectory(), dtype=float)

Ta = int(single_cse_result['cse_summary']['Ta'])
n_completed_3d = int(single_cse_result['n_completed'])
n_steps_3d = n_completed_3d * Ta
t0_3d = int(single_cse_result['t0'])
free_trajectory_3d = np.asarray(
    x_nature[:, t0_3d:t0_3d + n_steps_3d + 1],
    dtype=float,
)
analysis_ensembles_3d = np.asarray(
    single_cse_result['Xa_cycle_initial_history'],
    dtype=float,
)
analysis_mean_3d = analysis_ensembles_3d.mean(axis=2)

fig = plt.figure(figsize=(14, 7.2), facecolor='white')
ax_free = fig.add_subplot(121, projection='3d')
ax_analysis = fig.add_subplot(122, projection='3d')

free_artist = ax_free.scatter(
    free_trajectory_3d[0],
    free_trajectory_3d[1],
    free_trajectory_3d[2],
    c=np.linspace(0.0, 1.0, free_trajectory_3d.shape[1]),
    cmap='turbo',
    s=3,
    alpha=0.8,
    label='free nature trajectory',
)

for member in range(analysis_ensembles_3d.shape[2]):
    analysis_artist = ax_analysis.scatter(
        analysis_ensembles_3d[:, 0, member],
        analysis_ensembles_3d[:, 1, member],
        analysis_ensembles_3d[:, 2, member],
        color='lightcoral',
        s=5,
        alpha=0.35,
        label='analysis members' if member == 0 else None,
    )

mean_artist, = ax_analysis.plot(
    analysis_mean_3d[:, 0],
    analysis_mean_3d[:, 1],
    analysis_mean_3d[:, 2],
    color='darkred',
    linewidth=1.0,
    alpha=0.9,
    label='analysis mean',
)

for ax in (ax_free, ax_analysis):
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-25, 25)
    ax.set_ylim(-20, 20)
    ax.set_zlim(0, 45)
    ax.grid(alpha=0.25)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

ax_free.set_title('Free nature trajectory', pad=8)
ax_analysis.set_title('z-only analysis ensemble', pad=8)
ax_analysis.text2D(
    0.03,
    0.04,
    'Analysis x range: [{:.2f}, {:.2f}]'.format(
        np.min(analysis_ensembles_3d[:, 0, :]),
        np.max(analysis_ensembles_3d[:, 0, :]),
    ),
    transform=ax_analysis.transAxes,
    fontsize=9,
    bbox={'facecolor': 'white', 'edgecolor': '0.75', 'alpha': 0.9},
)

fig.legend(
    [free_artist, analysis_artist, mean_artist],
    ['free nature trajectory', 'analysis members', 'analysis mean'],
    frameon=False,
    ncol=3,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.93),
)
fig.suptitle('Free trajectory versus z-only analysis ensemble', y=0.995)
fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))

analysis_3d_path = (
    analysis_3d_figures_dir
    / 'free_nature_vs_z_only_analysis_ensemble_3d.png'
)
fig.savefig(analysis_3d_path, dpi=300, bbox_inches='tight')
plt.show()

print('Analysis x range:', float(np.min(analysis_ensembles_3d[:, 0, :])),
      float(np.max(analysis_ensembles_3d[:, 0, :])))
print('3D analysis-ensemble figure:', analysis_3d_path)
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
