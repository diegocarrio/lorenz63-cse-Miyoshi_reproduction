"""Build the isolated xyz-control experiment with ensemble-consensus regime logic."""

from copy import deepcopy
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '6_run_cse_qiwen_xyz_control_nature_qiwenevents.ipynb'
TARGET = ROOT / '17_run_cse_qiwen_xyz_control_ensemble_consensus_qiwenevents.ipynb'


def code_cell(source):
    return nbformat.v4.new_code_cell(source.strip() + '\n')


def markdown_cell(source):
    return nbformat.v4.new_markdown_cell(source.strip() + '\n')


with SOURCE.open('r', encoding='utf-8') as handle:
    source_notebook = nbformat.read(handle, as_version=4)

notebook = deepcopy(source_notebook)
for cell in notebook.cells:
    if cell.cell_type == 'code':
        cell.outputs = []
        cell.execution_count = None

notebook.cells[0].source = """## Isolated ensemble-consensus experiment

This notebook writes all results below `outputs/` and does not overwrite the
validated `nature`-reference experiment. Shared paths are defined in
`project_paths.py`."""

notebook.cells[1].source = """import importlib
import project_paths as _project_paths
_project_paths = importlib.reload(_project_paths)
from project_paths import (
    PROJECT_ROOT, ANALYSIS_DATA_DIR, OSSE_DATA_DIR,
    cle_target_experiment_paths, cse_experiment_paths, ensure_output_directories,
)
ensure_output_directories()
EXPERIMENT_NAME = 'xyz_control_ensemble_consensus_qiwenevents'
EXPERIMENT_ROOT, RESULTS_DIR, EXPERIMENT_FIGURES_DIR = cse_experiment_paths(EXPERIMENT_NAME)
CLE_TARGET_EXPERIMENT_ROOT, CLE_TARGET_DATA_DIR, CLE_TARGET_FIGURES_DIR = cle_target_experiment_paths(EXPERIMENT_NAME)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CLE_TARGET_DATA_DIR.mkdir(parents=True, exist_ok=True)
CLE_TARGET_FIGURES_DIR.mkdir(parents=True, exist_ok=True)"""

notebook.cells[2].source = """# Qiwen CSE xyz Control with an Ensemble-Consensus Regime Reference

This experiment changes only the source used to define the current Lorenz
regime. The sign is inferred from the complete analysis ensemble, following
Takemasa's proposal. No truth information is used by the controller."""

notebook.cells[6].source = """BASE_PATH = PROJECT_ROOT
PathFigures = EXPERIMENT_FIGURES_DIR
PathFigures.mkdir(parents=True, exist_ok=True)

# Full-state observations and full xyz actuation are retained from notebook 6.
# The only experimental change is the current-regime definition.
CONTROL_INDICES = None
CONTROL_LABEL = 'xyz'
REGIME_REFERENCE_MODE = 'ensemble_consensus'
AMBIGUOUS_REGIME_POLICY = 'stop'
CANDIDATE_POOL = 'full'
EVENT_SELECTION = 'qiwen'
OUTPUT_TAG = '_ensemble_consensus_qiwenevents'
multiple_number = 4
T_0 = 75.1
position = int(np.ceil(T_0 * multiple_number))

das = da_system().load(ANALYSIS_DATA_DIR / 'x_analysis_init.pkl')

sv = das.getStateVector()
x_nature = sv.getTrajectory()

obs = das.getObsData()
y_obs = obs.getVal()

Ta = das.acyc_step
fcst_dt = das.fcst_dt

das.multiplicative_inflation = 1.04

print('Regime reference:', REGIME_REFERENCE_MODE)
print('Ambiguous-regime policy:', AMBIGUOUS_REGIME_POLICY)
print('Candidate pool:', CANDIDATE_POOL)"""

notebook.cells[10].source = notebook.cells[10].source.replace(
    "cse_qiwen_xyz_control_change_region_time_qiwen_qiwenevents.pkl",
    "cse_qiwen_xyz_control_change_region_time_ensemble_consensus_qiwenevents.pkl",
)
notebook.cells[20].source = notebook.cells[20].source.replace(
    "cse_initialization_qiwen_qiwenevents.png",
    "cse_initialization_ensemble_consensus_qiwenevents.png",
)

notebook.cells[21].source = """## Run One CSE Event with Ensemble Consensus

At the beginning of every DA cycle, the regime is inferred from the current
analysis ensemble:

- all member values satisfy `x > 0`: positive-regime consensus;
- all member values satisfy `x < 0`: negative-regime consensus;
- otherwise: ambiguous regime.

The usual CSE trigger is retained: control is considered when at least one
member predicts a regime transition. The consensus sign replaces `nature` or
`member0` only where a signed current-regime reference is required. If the
analysis ensemble is ambiguous, this pilot stops and records an information
failure rather than selecting an arbitrary sign. `candidate_pool='full'` is
retained so that the regime-reference rule is the only change relative to
notebook 6."""

notebook.cells[22].source = """# Reload the module so this cell sees the ensemble-consensus implementation.
import importlib
import class_cse_qiwen as _cse_qiwen_module
_cse_qiwen_module = importlib.reload(_cse_qiwen_module)
run_single_cse_core = _cse_qiwen_module.run_single_cse


def run_single_cse(
    event_index,
    D_run,
    seed,
    regime_reference=REGIME_REFERENCE_MODE,
    candidate_pool=CANDIDATE_POOL,
    ambiguous_regime_policy=AMBIGUOUS_REGIME_POLICY,
):
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
        ambiguous_regime_policy=ambiguous_regime_policy,
        control_indices=CONTROL_INDICES,
        T_0=T_0,
        multiple_number=multiple_number,
        cle_target_output_dir=None,
    )


single_cse_result = run_single_cse(
    event_index=EVENT_INDEX,
    D_run=D,
    seed=1,
)

# Export a CLE target only for a completed successful experiment.
if single_cse_result['success']:
    single_cse_result['cle_target_files'] = _cse_qiwen_module.export_cle_target_inputs(
        single_cse_result,
        CLE_TARGET_DATA_DIR,
    )
else:
    single_cse_result['cle_target_files'] = {}

result_file = RESULTS_DIR / (
    f'cse_qiwen_xyz_control_ensemble_consensus_event_{EVENT_INDEX}_D{D}.pkl'
)
temporary_result_file = result_file.with_suffix(result_file.suffix + '.tmp')
with temporary_result_file.open('wb') as handle:
    pickle.dump(single_cse_result, handle)
temporary_result_file.replace(result_file)

print('Event index:', single_cse_result['event_index'])
print('Success:', single_cse_result['success'])
print('Failure cycle:', single_cse_result['failure_cycle'])
print('Failure reason:', single_cse_result['failure_reason'])
print('Completed cycles:', single_cse_result['n_completed'])
print('Control ratio:', single_cse_result['control_ratio'])
print('False alarm ratio:', single_cse_result['false_alarm_ratio'])
print('Success control ratio:', single_cse_result['success_control_ratio'])
print('Branch counts:', single_cse_result['branch_counts'])
print('Safe searches:', single_cse_result['n_safe_searches'])
print('Ambiguous regime cycles:', single_cse_result['ambiguous_regime_cycles'].tolist())
print('Saved result:', result_file)
if single_cse_result['cle_target_files']:
    print('CLE_TARGET input files:')
    for cle_input_name, cle_input_path in single_cse_result['cle_target_files'].items():
        print(f'  {cle_input_name}: {cle_input_path}')"""

consensus_markdown = markdown_cell("""## Analysis-Ensemble Regime Consensus

This diagnostic displays the complete set of current analysis-member values of
`x`. A cycle is unambiguous only when the full ensemble lies strictly on one
side of zero. The diagnostic uses no nature-run values.""")

consensus_code = code_cell("""consensus_history = single_cse_result['regime_consensus_history']
consensus_cycles = np.asarray([row['cycle_index'] for row in consensus_history], dtype=int)
consensus_x = np.asarray([row['x_values'] for row in consensus_history], dtype=float)
consensus_sign = np.asarray([row['regime_sign'] for row in consensus_history], dtype=float)

n_positive_consensus = int(np.sum(consensus_sign > 0))
n_negative_consensus = int(np.sum(consensus_sign < 0))
n_ambiguous = int(np.sum(consensus_sign == 0))

fig_consensus, ax_consensus = plt.subplots(figsize=(14, 5.5), constrained_layout=True)
for member in range(consensus_x.shape[1]):
    ax_consensus.plot(
        consensus_cycles,
        consensus_x[:, member],
        marker='o',
        markersize=2.2,
        linewidth=0.9,
        alpha=0.65,
        label=f'analysis member {member}',
    )
ax_consensus.fill_between(
    consensus_cycles,
    consensus_x.min(axis=1),
    consensus_x.max(axis=1),
    color='gray',
    alpha=0.15,
    label='ensemble x range',
)
ambiguous_mask = consensus_sign == 0
if np.any(ambiguous_mask):
    ax_consensus.scatter(
        consensus_cycles[ambiguous_mask],
        np.zeros(np.sum(ambiguous_mask)),
        color='black',
        marker='x',
        s=70,
        linewidths=2,
        label='ambiguous regime',
        zorder=6,
    )
ax_consensus.axhline(0.0, color='black', linestyle='--', linewidth=1.0)
ax_consensus.set_xlabel('DA cycle')
ax_consensus.set_ylabel('analysis-member x')
ax_consensus.set_title('Current Regime Inferred from Whole-Ensemble Agreement')
ax_consensus.grid(alpha=0.25)
ax_consensus.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4)

consensus_figure_path = PathFigures / 'analysis_ensemble_regime_consensus.png'
fig_consensus.savefig(consensus_figure_path, dpi=200, bbox_inches='tight')
plt.show()

print('Positive-consensus cycles:', n_positive_consensus)
print('Negative-consensus cycles:', n_negative_consensus)
print('Ambiguous cycles:', n_ambiguous)
print('Saved consensus diagnostic:', consensus_figure_path)""")

# Insert the consensus diagnostic immediately after the run cell.
notebook.cells[23:23] = [consensus_markdown, consensus_code]

# The original plot cells shifted by two positions after insertion.
notebook.cells[26].source = notebook.cells[26].source.replace(
    "nr_controlled_vs_nr_qiwenevents.png",
    "nr_controlled_vs_nr_ensemble_consensus_qiwenevents.png",
)
notebook.cells[26].source = """# Plot the controlled and free trajectories only when at least one cycle completed.
if single_cse_result['n_completed'] > 0:
    import importlib
    import class_cse as _class_cse_module
    _class_cse_module = importlib.reload(_class_cse_module)
    plot_controlled_vs_free_nature_run = _class_cse_module.plot_controlled_vs_free_nature_run

    fig, axes, controlled_vs_free_path = plot_controlled_vs_free_nature_run(
        evo_history=single_cse_result['evo_history'],
        x_nature=x_nature,
        t0=single_cse_result['t0'],
        n_completed=single_cse_result['n_completed'],
        Ta=Ta,
        y_obs=y_obs,
        yo_history=single_cse_result['yo_history'],
        branch_history=single_cse_result['branch_history'],
        perturbed_time=single_cse_result['perturbed_time'],
        xdim=das.xdim,
        path_figures=PathFigures,
        output_filename='nr_controlled_vs_nr_ensemble_consensus_qiwenevents.png',
    )
else:
    print('Controlled-versus-free plot skipped: no CSE cycle was completed.')"""

notebook.cells[28].source = """# Plot the 3D comparison only when at least one cycle completed.
if single_cse_result['n_completed'] > 0:
    import importlib
    import class_cse_qiwen as cse_qiwen
    cse_qiwen = importlib.reload(cse_qiwen)

    plot_3d_data = cse_qiwen.plot_3d_nature_run_comparison_with_fallback(
        current_result=single_cse_result,
        x_nature=x_nature,
        Ta=Ta,
        path_figures=PathFigures,
        analysis_data_path=ANALYSIS_DATA_DIR / 'x_analysis_init.pkl',
        results_dir=RESULTS_DIR,
        output_tag=OUTPUT_TAG,
        D=D,
        event_index=EVENT_INDEX,
        experiment_step=experiment_step,
        output_filename='NR_3D_comparison_after_control_ensemble_consensus_qiwenevents.png',
    )

    plot_result = plot_3d_data['plot_result']
    plot_evo_history = plot_3d_data['plot_evo_history']
    plot_t0 = plot_3d_data['plot_t0']
    plot_num_cycles = plot_3d_data['plot_num_cycles']
    x_nature_nocontrol_3d = plot_3d_data['nature_nocontrol']
    x_nature_control_3d = plot_3d_data['nature_control']
    fig_3d_nr = plot_3d_data['fig']
    axes_3d_nr = plot_3d_data['axes']
    nr_3d_path = plot_3d_data['output_path']
else:
    print('Three-dimensional comparison skipped: no CSE cycle was completed.')"""

notebook.metadata.setdefault('kernelspec', {})
notebook.metadata['kernelspec'].update({
    'display_name': 'Control',
    'language': 'python',
    'name': 'python3',
})

with TARGET.open('w', encoding='utf-8') as handle:
    nbformat.write(notebook, handle)

print(TARGET)
