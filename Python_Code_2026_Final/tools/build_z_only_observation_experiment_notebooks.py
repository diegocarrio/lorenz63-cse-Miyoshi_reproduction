"""Build the isolated z-observation / z-control experiment notebooks."""

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def markdown(source):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def write_notebook(filename, cells):
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    output_path = PROJECT_ROOT / filename
    nbf.write(notebook, output_path)
    print(f"Wrote {output_path}")


notebook_12 = [
    markdown(
        r"""
# Prepare z-Only Synthetic Observations

This notebook prepares the observation dataset for the first information-channel experiment proposed by Takemasa: **observe only $z$ while keeping the later CSE interventions restricted to $z$**.

To make the comparison with the existing full-state experiment as controlled as possible, no new noise realization is generated. The $z$ row is extracted from the already validated full-state observation file produced by `2_gen_obs.ipynb`. Therefore, the observation times, truth trajectory, error variance, and actual noise samples in $z$ are unchanged; only $x$ and $y$ are withheld from data assimilation.
"""
    ),
    code(
        """
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from class_obs import obs_da
from class_state_vec import state_vector
from project_paths import NATURE_DATA_DIR, OBSERVATIONS_DATA_DIR, OBSERVATIONS_FIGURES_DIR

OBSERVED_INDICES = (2,)
OBSERVATION_LABEL = 'z_only_obs'

OUTPUT_ROOT = OBSERVATIONS_DATA_DIR.parent / OBSERVATION_LABEL
DATA_DIR = OUTPUT_ROOT / 'data'
FIGURES_DIR = OUTPUT_ROOT / 'figures'
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

NATURE_FILE = NATURE_DATA_DIR / 'lorenz63_run.pkl'
FULL_OBSERVATION_FILE = OBSERVATIONS_DATA_DIR / 'y_obs.pkl'
OUTPUT_FILE = DATA_DIR / 'y_obs_z_only.pkl'

print('Full-state input:', FULL_OBSERVATION_FILE)
print('z-only output:', OUTPUT_FILE)
"""
    ),
    markdown(
        r"""
## Extract the Existing z Observation Channel

The selector is

\[
H_z = \begin{bmatrix}0 & 0 & 1\end{bmatrix}.
\]

The saved observation object has shape $(1,N)$ and position index 2 at every time. This is a new artifact in an isolated directory; `y_obs.pkl` is never modified.
"""
    ),
    code(
        """
sv = state_vector().load(NATURE_FILE)
x_nature = np.asarray(sv.getTrajectory(), dtype=float)

full_obs = obs_da().load(FULL_OBSERVATION_FILE)
full_values = np.asarray(full_obs.getVal(), dtype=float)
full_positions = np.asarray(full_obs.getPos(), dtype=int)

if full_values.shape != x_nature.shape:
    raise ValueError(
        f'Full observations {full_values.shape} and nature run {x_nature.shape} are not aligned.'
    )
if full_positions.shape != full_values.shape:
    raise ValueError('Full observation positions do not match the observation array.')

z_values = full_values[np.asarray(OBSERVED_INDICES), :].copy()
z_positions = np.full(z_values.shape, OBSERVED_INDICES[0], dtype=int)

z_obs = obs_da(name='observe_z_only')
z_obs.setVal(z_values)
z_obs.setPos(z_positions)
z_obs.setErr(np.array([1.5], dtype=float))
z_obs.observed_indices = list(OBSERVED_INDICES)
z_obs.source_observation_file = str(FULL_OBSERVATION_FILE)
z_obs.noise_variance = 1.5
z_obs.common_random_numbers = True
z_obs.save(OUTPUT_FILE)

assert z_values.shape == (1, x_nature.shape[1])
assert np.array_equal(z_values[0], full_values[2])
assert np.all(z_positions == 2)

print('Nature-run shape:', x_nature.shape)
print('z-only observation shape:', z_values.shape)
print('The saved z values exactly match row 2 of the full-state observations.')
"""
    ),
    markdown("## Visual Verification"),
    code(
        """
window = 500
steps = np.arange(window)

fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(steps, x_nature[2, :window], color='black', linewidth=1.3, label='nature z')
ax.scatter(steps, z_values[0, :window], s=9, color='tab:purple', alpha=0.55, label='observed z')
ax.set_xlabel('Time step')
ax.set_ylabel('z')
ax.set_title('z-only synthetic observations: unchanged z noise realization')
ax.grid(alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()
figure_path = FIGURES_DIR / 'z_only_observations_first_500_steps.png'
fig.savefig(figure_path, dpi=300, bbox_inches='tight')
plt.show()

reloaded = obs_da().load(OUTPUT_FILE)
assert np.array_equal(reloaded.getVal(), z_values)
assert np.array_equal(reloaded.getPos(), z_positions)
print('Reload validation passed:', OUTPUT_FILE)
print('Figure:', figure_path)
"""
    ),
]


notebook_13 = [
    markdown(
        r"""
# Initialize Data Assimilation with z-Only Observations

This notebook creates a new DA-system object for the z-only observation experiment. The dynamical state remains $(x,y,z)$, but the observation space has dimension one:

\[
y_k = H_z x_k + \varepsilon_k,\qquad
H_z=\begin{bmatrix}0&0&1\end{bmatrix},\qquad
R=[2].
\]

All ensemble, timing, model-state, and error settings are kept equal to the validated full-observation initialization. The output is written to a separate directory and does not replace `x_analysis_init.pkl`.
"""
    ),
    code(
        """
from pathlib import Path

import numpy as np

from class_da_sys import da_system
from class_obs import obs_da
from class_state_vec import state_vector
from project_paths import ANALYSIS_DATA_DIR, NATURE_DATA_DIR, OBSERVATIONS_DATA_DIR

OBSERVATION_LABEL = 'z_only_obs'
OBSERVED_INDICES = (2,)

OBSERVATION_FILE = (
    OBSERVATIONS_DATA_DIR.parent / OBSERVATION_LABEL / 'data' / 'y_obs_z_only.pkl'
)
OUTPUT_ROOT = ANALYSIS_DATA_DIR.parent / OBSERVATION_LABEL
DATA_DIR = OUTPUT_ROOT / 'data'
FIGURES_DIR = OUTPUT_ROOT / 'figures'
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = DATA_DIR / 'x_analysis_init_z_only_obs.pkl'

print('Observation input:', OBSERVATION_FILE)
print('DA-system output:', OUTPUT_FILE)
"""
    ),
    code(
        """
sv = state_vector().load(NATURE_DATA_DIR / 'lorenz63_run.pkl')
obs = obs_da().load(OBSERVATION_FILE)

x_nature = np.asarray(sv.getTrajectory(), dtype=float)
y_obs = np.asarray(obs.getVal(), dtype=float)
y_pos = np.asarray(obs.getPos(), dtype=int)
xdim, maxit = x_nature.shape
ydim = y_obs.shape[0]

assert xdim == 3
assert ydim == 1
assert y_obs.shape[1] == maxit
assert np.all(y_pos == 2)

das = da_system()
das.setStateVector(sv)
das.setObsData(obs)
das.xdim = xdim
das.ydim = ydim
das.x0 = x_nature[:, 0]

das.edim = 3
das.ens_bias_init = 0
das.ens_sigma_init = np.sqrt(2)

das.setB(np.zeros((xdim, xdim)))
das.setR(2.0 * np.eye(ydim))
das.setH(np.array([[0.0, 0.0, 1.0]]))

t_nature = np.asarray(sv.getTimes())
das.acyc_step = 8
das.dtau = t_nature[das.acyc_step] - t_nature[0]
das.fcst_step = das.acyc_step
das.fcst_dt = das.dtau / das.fcst_step
das.dt = t_nature[1] - t_nature[0]
das.maxit = maxit
das.setMethod('EnKF')

das.observed_indices = list(OBSERVED_INDICES)
das.observation_label = OBSERVATION_LABEL
das.source_observation_file = str(OBSERVATION_FILE)
das.save(OUTPUT_FILE)
"""
    ),
    markdown("## Configuration Checks"),
    code(
        """
reloaded = da_system().load(OUTPUT_FILE)

expected_H = np.array([[0.0, 0.0, 1.0]])
assert reloaded.xdim == 3
assert reloaded.ydim == 1
assert reloaded.edim == 3
assert np.array_equal(np.asarray(reloaded.H), expected_H)
assert np.array_equal(np.asarray(reloaded.R), np.array([[2.0]]))
assert reloaded.acyc_step == 8
assert reloaded.getObsData().getVal().shape == (1, maxit)

print('State dimension:', reloaded.xdim)
print('Observation dimension:', reloaded.ydim)
print('Observed state indices:', reloaded.observed_indices)
print('H =\\n', np.asarray(reloaded.H))
print('R =\\n', np.asarray(reloaded.R))
print('Ensemble size:', reloaded.edim)
print('Analysis interval:', reloaded.acyc_step, 'model steps')
print('Saved and reloaded successfully:', OUTPUT_FILE)
"""
    ),
]


notebook_14 = [
    markdown(
        r"""
# Run the OSSE with z-Only Observations

This notebook reruns the offline OSSE using the new one-channel DA configuration. Only $z$ is assimilated; the ensemble still evolves in the full three-dimensional Lorenz-63 state space. The resulting analysis history is therefore independent of the full-state OSSE and is stored under `outputs/04_osse/z_only_obs/`.

The long run is intentionally identical to the validated OSSE in timing, ensemble size, model, initial-ensemble seed, and multiplicative inflation. This notebook does **not** perform an inflation sweep, because the first experiment is designed to change only the observation operator.
"""
    ),
    code(
        """
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np

from class_da_sys import da_system
from class_osse import osse
from project_paths import ANALYSIS_DATA_DIR, OSSE_DATA_DIR

OBSERVATION_LABEL = 'z_only_obs'
ANALYSIS_FILE = (
    ANALYSIS_DATA_DIR.parent / OBSERVATION_LABEL / 'data'
    / 'x_analysis_init_z_only_obs.pkl'
)
OUTPUT_ROOT = OSSE_DATA_DIR.parent / OBSERVATION_LABEL
DATA_DIR = OUTPUT_ROOT / 'data'
RESULTS_DIR = OUTPUT_ROOT / 'results'
FIGURES_DIR = OUTPUT_ROOT / 'figures'
for directory in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

OSSE_OUTPUT_FILES = {
    'analysis_file': str(DATA_DIR / 'osse_Xa_history.pkl'),
    'prior_file': str(DATA_DIR / 'osse_Xf_history.pkl'),
    'summary_file': str(DATA_DIR / 'osse_summary.pkl'),
    'checkpoint_file': str(DATA_DIR / 'osse_Xa_history_checkpoint.npy'),
    'prior_checkpoint_file': str(DATA_DIR / 'osse_Xf_history_checkpoint.npy'),
    'checkpoint_summary_file': str(DATA_DIR / 'osse_summary_checkpoint.pkl'),
}

print('DA-system input:', ANALYSIS_FILE)
print('Isolated OSSE directory:', OUTPUT_ROOT)
"""
    ),
    markdown("## Load and Validate the z-Only DA System"),
    code(
        """
das = da_system().load(ANALYSIS_FILE)
sv = das.getStateVector()
x_nature = np.asarray(sv.getTrajectory(), dtype=float)
y_obs = np.asarray(das.getObsData().getVal(), dtype=float)

assert das.xdim == 3 and das.ydim == 1
assert np.array_equal(np.asarray(das.H), np.array([[0.0, 0.0, 1.0]]))
assert np.asarray(das.R).shape == (1, 1)
assert y_obs.shape == (1, x_nature.shape[1])

sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0
lorenz63_model = osse.make_lorenz63_model(sigma=sigma, rho=rho, beta=beta)

Ta = 8
t0 = 0
fcst_dt = das.fcst_dt
multiple_number = 4
T_0 = 75.1
MULTIPLICATIVE_INFLATION = 1.04

osse_exp = osse(
    das=das,
    x_nature=x_nature,
    y_obs=y_obs,
    model_func=lorenz63_model,
    sigma=sigma,
    beta=beta,
    rho=rho,
    fcst_dt=fcst_dt,
    path_figures=str(FIGURES_DIR) + '/',
)

print('H =', np.asarray(das.H))
print('Observation array shape:', y_obs.shape)
print('Nature-run shape:', x_nature.shape)
"""
    ),
    markdown(
        r"""
## Run the Full-Period Minimal OSSE

This is the computationally expensive cell. It covers the available nature run and writes checkpoints plus the final analysis/prior histories. Existing baseline OSSE files are not touched.
"""
    ),
    code(
        """
RUN_OSSE_FULL_NATURE_PERIOD = True
RUN_OSSE_MINIMAL_OUTPUT = True
DEFAULT_OSSE_NUM_CYCLES = 1000
OSSE_PROGRESS_EVERY = 10000
OSSE_CHECKPOINT_EVERY = 50000

osse_run = osse_exp.run_standard_osse_experiment(
    t0=t0,
    Ta=Ta,
    default_num_cycles=DEFAULT_OSSE_NUM_CYCLES,
    run_full_nature_period=RUN_OSSE_FULL_NATURE_PERIOD,
    run_minimal_output=RUN_OSSE_MINIMAL_OUTPUT,
    progress_every=OSSE_PROGRESS_EVERY,
    checkpoint_every=OSSE_CHECKPOINT_EVERY,
    multiple_number=multiple_number,
    T_0=T_0,
    multiplicative_inflation=MULTIPLICATIVE_INFLATION,
    **OSSE_OUTPUT_FILES,
    extra_summary={
        'input_das_file': str(ANALYSIS_FILE),
        'observation_mode': 'z_only',
        'observed_indices': [2],
        'observation_operator': [[0.0, 0.0, 1.0]],
        'common_random_numbers_with_full_observation_osse': True,
        'inflation_sweep_performed': False,
        'initEns_seed': 2,
    },
)

osse.publish_run_outputs(osse_run, globals(), include_full_diagnostics=False)
"""
    ),
    markdown("## Short-Window Diagnostic"),
    code(
        """
with open(DATA_DIR / 'osse_Xa_history.pkl', 'rb') as stream:
    saved_analysis = np.asarray(pickle.load(stream), dtype=float)
with open(DATA_DIR / 'osse_Xf_history.pkl', 'rb') as stream:
    saved_prior = np.asarray(pickle.load(stream), dtype=float)
with open(DATA_DIR / 'osse_summary.pkl', 'rb') as stream:
    saved_summary = pickle.load(stream)

assert saved_analysis.shape == saved_prior.shape
assert saved_analysis.shape[1:] == (das.xdim, das.edim)
assert saved_summary['observed_indices'] == [2]

n_plot = min(45, saved_analysis.shape[0])
times = saved_summary['t0'] + np.arange(n_plot) * saved_summary['Ta']
labels = ('x', 'y', 'z')
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for state_index, ax in enumerate(axes):
    ax.plot(times, x_nature[state_index, times], color='black', linewidth=1.5, label='nature')
    for member in range(das.edim):
        ax.plot(times, saved_prior[:n_plot, state_index, member], color='tab:orange', alpha=0.35, linewidth=0.8)
        ax.plot(times, saved_analysis[:n_plot, state_index, member], color='tab:blue', alpha=0.45, linewidth=0.8)
    if state_index == 2:
        ax.scatter(times, y_obs[0, times], color='tab:purple', s=14, alpha=0.7, label='observed z')
    ax.set_ylabel(labels[state_index])
    ax.grid(alpha=0.2)

axes[0].plot([], [], color='tab:orange', label='prior members')
axes[0].plot([], [], color='tab:blue', label='analysis members')
axes[0].legend(frameon=False, ncol=3)
axes[2].legend(frameon=False)
axes[2].set_xlabel('Absolute time step')
fig.suptitle('OSSE analysis with z-only observations')
fig.tight_layout()
diagnostic_path = FIGURES_DIR / 'osse_z_only_obs_analysis_members.png'
fig.savefig(diagnostic_path, dpi=300, bbox_inches='tight')
plt.show()

print('Analysis history:', saved_analysis.shape)
print('Completed cycles:', saved_summary['completed_cycles'])
print('Diagnostic figure:', diagnostic_path)
"""
    ),
]


notebook_15 = [
    markdown(
        r"""
# z-Only CSE Control with z-Only Observations

This notebook performs the first direct information-channel test proposed by Takemasa:

- **observation:** $z$ only;
- **intervention:** $u_z$ only;
- **control objective:** retain the sign of $x$;
- **CSE decision rule:** unchanged.

The experiment uses the independently generated z-only OSSE analysis history from notebook 14. The CSE core in `class_cse_qiwen.py` is not modified. A thin adapter passes only the generated $z$ observation to the one-channel DA update while preserving the original three random draws, so the observation-noise stream remains paired with the existing full-observation experiment.

As in the validated nature-reference configuration, the fallback branch retains one additional binary piece of objective information: the **current sign of the controlled $x$ state** is used to select a past safe trajectory. This is not a full $x$ observation in the DA update, and no future uncontrolled nature state is used. Stating this boundary explicitly is essential: this notebook tests z-only *assimilation* while retaining the sign information that Takemasa identified as necessary for judging the control objective.
"""
    ),
    code(
        """
from pathlib import Path
import importlib
import pickle

import matplotlib.pyplot as plt
import numpy as np

from class_da_sys import da_system
import class_cse_qiwen as cse_qiwen
import class_cse_observation_experiment as observation_experiment
from project_paths import ANALYSIS_DATA_DIR, CSE_ROOT, OSSE_DATA_DIR

cse_qiwen = importlib.reload(cse_qiwen)
observation_experiment = importlib.reload(observation_experiment)

OBSERVATION_LABEL = 'z_only_obs'
EXPERIMENT_NAME = 'z_control_z_obs_nature_qiwenevents'
OBSERVED_INDICES = (2,)
CONTROL_INDICES = [2]

ANALYSIS_FILE = (
    ANALYSIS_DATA_DIR.parent / OBSERVATION_LABEL / 'data'
    / 'x_analysis_init_z_only_obs.pkl'
)
OSSE_EXPERIMENT_ROOT = OSSE_DATA_DIR.parent / OBSERVATION_LABEL
OSSE_EXPERIMENT_DATA_DIR = OSSE_EXPERIMENT_ROOT / 'data'
EXPERIMENT_ROOT = CSE_ROOT / EXPERIMENT_NAME
RESULTS_DIR = EXPERIMENT_ROOT / 'results'
FIGURES_DIR = EXPERIMENT_ROOT / 'figures'
for directory in (RESULTS_DIR, FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

print('DA-system input:', ANALYSIS_FILE)
print('OSSE input:', OSSE_EXPERIMENT_DATA_DIR)
print('CSE output:', EXPERIMENT_ROOT)
"""
    ),
    markdown("## Load the z-Only OSSE and Validate the Information Boundary"),
    code(
        """
das = da_system().load(ANALYSIS_FILE)
das.multiplicative_inflation = 1.04
x_nature = np.asarray(das.getStateVector().getTrajectory(), dtype=float)

with open(OSSE_EXPERIMENT_DATA_DIR / 'osse_Xa_history.pkl', 'rb') as stream:
    osse_Xa_history = np.asarray(pickle.load(stream), dtype=float)
with open(OSSE_EXPERIMENT_DATA_DIR / 'osse_summary.pkl', 'rb') as stream:
    osse_summary = pickle.load(stream)

assert das.xdim == 3 and das.ydim == 1
assert np.array_equal(np.asarray(das.H), np.array([[0.0, 0.0, 1.0]]))
assert np.asarray(das.R).shape == (1, 1)
assert osse_summary['observed_indices'] == [2]
assert osse_Xa_history.shape[1:] == (das.xdim, das.edim)

Ta = das.acyc_step
fcst_dt = das.fcst_dt
multiple_number = 4
T_0 = 75.1
position = int(np.ceil(T_0 * multiple_number))

print('H =', np.asarray(das.H))
print('Observed indices:', OBSERVED_INDICES)
print('Controlled indices:', CONTROL_INDICES)
print("Fallback regime-sign source: current controlled x (regime_reference='nature')")
print('OSSE analysis history:', osse_Xa_history.shape)
"""
    ),
    markdown(
        r"""
## Select the Same Qiwen Event Family

Nature-run crossings define a reproducible evaluation set and aligned initial times. They do not enter the cycle-by-cycle intervention rule. Once the experiment begins, DA receives only $z$. The existing fallback rule may additionally read the current sign of controlled $x$ because `regime_reference='nature'`; it never reads the future uncontrolled transition.
"""
    ),
    code(
        """
regime_change_info = cse_qiwen.detect_regime_changes(
    x_nature,
    RESULTS_DIR,
    position=position,
    output_filename='change_region_time_z_obs_qiwenevents.pkl',
    event_selection='qiwen',
)
t_start_series = regime_change_info['t_start_series']

D = 0.05
experiment_step = 1000
EVENT_INDEX = 1

if EVENT_INDEX >= len(t_start_series):
    raise IndexError(f'EVENT_INDEX={EVENT_INDEX} but only {len(t_start_series)} events are available.')

t_start = int(t_start_series[EVENT_INDEX])
t0 = (t_start - position) - (t_start - position) % Ta
osse_start_index = (t0 - osse_summary['t0']) // Ta
if osse_summary['t0'] + osse_start_index * Ta != t0:
    raise ValueError('t0 is not aligned with the z-only OSSE grid.')

print('Event index:', EVENT_INDEX)
print('Crossing used to define the test window:', t_start)
print('Aligned CSE start:', t0)
print('z-only OSSE start index:', osse_start_index)
"""
    ),
    markdown("## Run One z-Observation / z-Control Event"),
    code(
        """
single_cse_result = observation_experiment.run_single_cse_with_observation_subset(
    das=das,
    observed_indices=OBSERVED_INDICES,
    run_single_cse_core=cse_qiwen.run_single_cse,
    event_index=EVENT_INDEX,
    D_run=D,
    seed=1,
    x_nature=x_nature,
    t_start_series=t_start_series,
    osse_Xa_history=osse_Xa_history,
    osse_summary=osse_summary,
    experiment_step=experiment_step,
    position=position,
    Ta=Ta,
    fcst_dt=fcst_dt,
    path_figures=FIGURES_DIR,
    total_events=len(t_start_series),
    regime_reference='nature',
    candidate_pool='full',
    control_indices=CONTROL_INDICES,
    T_0=T_0,
    multiple_number=multiple_number,
)

D_TAG = f'{D:g}'.replace('.', 'p')
FULL_RESULT_PATH = RESULTS_DIR / (
    f'cse_z_control_z_obs_event_{EVENT_INDEX}_D{D_TAG}_qiwenevents.pkl'
)
temporary_path = FULL_RESULT_PATH.with_suffix(FULL_RESULT_PATH.suffix + '.tmp')
with temporary_path.open('wb') as stream:
    pickle.dump(single_cse_result, stream, pickle.HIGHEST_PROTOCOL)
temporary_path.replace(FULL_RESULT_PATH)

COMPARISON_PATH = RESULTS_DIR / (
    f'cse_z_control_z_obs_nature_run_event_{EVENT_INDEX}_D{D_TAG}_qiwenevents.pkl'
)
cse_qiwen.export_nature_run_comparison(
    single_cse_result,
    x_nature,
    COMPARISON_PATH,
)

assert single_cse_result['yo_history'].shape[1] == 1
assert single_cse_result['yo_history_full_generated'].shape[1] == 3
assert np.allclose(
    single_cse_result['yo_history'][:, 0],
    single_cse_result['yo_history_full_generated'][:, 2],
)

print('Success:', single_cse_result['success'])
print('Failure cycle:', single_cse_result['failure_cycle'])
print('Completed cycles:', single_cse_result['n_completed'])
print('Control ratio:', single_cse_result['control_ratio'])
print('Branch counts:', single_cse_result['branch_counts'])
print('Assimilated observation shape:', single_cse_result['yo_history'].shape)
print('Full generated observation shape:', single_cse_result['yo_history_full_generated'].shape)
print('Full result:', FULL_RESULT_PATH)
print('Nature-run comparison:', COMPARISON_PATH)
"""
    ),
    markdown(
        r"""
## z-Only Analysis Ensemble and Lorenz Symmetry

The preceding trajectory figures show the controlled truth, not the DA ensemble. This diagnostic compares the analysis ensemble at the beginning of every CSE cycle with both the controlled state and its Lorenz-symmetric counterpart,

\[
(x,y,z) \longrightarrow (-x,-y,z).
\]

If observations of $z$ cannot identify the active lobe, the ensemble may remain close to the symmetry-related state even while the controlled truth stays in the desired regime.
"""
    ),
    code(
        """
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
    'x-sign agreement = {:.1f}%\\nMean RMSE to controlled state = {:.2f}\\nMean RMSE to symmetry counterpart = {:.2f}'.format(
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
"""
    ),
    markdown("## Free versus Controlled Nature Trajectories"),
    code(
        """
n_completed = int(single_cse_result['n_completed'])
n_steps = n_completed * Ta
controlled = np.asarray(single_cse_result['evo_history'], dtype=float)[:, :n_steps + 1]
free = x_nature[:, single_cse_result['t0']:single_cse_result['t0'] + n_steps + 1]
relative_steps = np.arange(n_steps + 1)

interventions = np.zeros_like(controlled)
for cycle, item in enumerate(single_cse_result['intervention_history']):
    start = cycle * Ta
    interventions[:, start:start + Ta] = np.asarray(item['perturbations']).T

fig, axes = plt.subplots(4, 1, figsize=(11, 9.5), sharex=True)
labels = ('x', 'y', 'z')
for state_index, ax in enumerate(axes[:3]):
    ax.plot(relative_steps, free[state_index], color='0.35', linewidth=1.15, label='free nature run')
    ax.plot(relative_steps, controlled[state_index], color='tab:blue', linestyle='--', linewidth=1.35, label='controlled nature run')
    if state_index < 2:
        ax.axhline(0.0, color='0.75', linewidth=0.8)
    ax.set_ylabel(labels[state_index])
    ax.grid(alpha=0.2)

nonzero = np.flatnonzero(np.abs(interventions[2]) > 0)
axes[3].vlines(nonzero, 0.0, interventions[2, nonzero], color='tab:red', linewidth=0.8)
axes[3].axhline(0.0, color='0.35', linewidth=0.8)
axes[3].set_ylabel(r'$u_z$')
axes[3].set_xlabel('Steps after CSE initialization')
axes[3].grid(alpha=0.2)
legend_handles, legend_labels = axes[0].get_legend_handles_labels()
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
fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
comparison_figure_path = FIGURES_DIR / 'z_obs_z_control_free_vs_controlled.png'
fig.savefig(comparison_figure_path, dpi=300, bbox_inches='tight')
plt.show()

print('Figure:', comparison_figure_path)
"""
    ),
    markdown("## Three-Dimensional Attractor Comparison"),
    code(
        """
plot_3d_data = cse_qiwen.plot_3d_nature_run_comparison_with_fallback(
    current_result=single_cse_result,
    x_nature=x_nature,
    Ta=Ta,
    path_figures=FIGURES_DIR,
    analysis_data_path=ANALYSIS_FILE,
    results_dir=RESULTS_DIR,
    output_tag='_z_obs_qiwenevents',
    D=D,
    event_index=EVENT_INDEX,
    experiment_step=experiment_step,
    control_label='z_obs_z_control',
    output_filename='NR_3D_comparison_z_obs_z_control.png',
)

print('3D figure:', plot_3d_data['output_path'])
"""
    ),
    markdown(
        r"""
## Three-Dimensional Free Trajectory versus z-Only Analysis Ensemble

This companion figure replaces the controlled trajectory in the right-hand panel with the analysis ensemble at every DA-cycle initialization. It directly tests the Lorenz symmetry ambiguity: the free trajectory visits both lobes, whereas the z-only analyses are expected to populate the symmetry-related negative lobe.
"""
    ),
    code(
        """
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
"""
    ),
    markdown(
        r"""
## Interpretation Boundary

This single event answers whether the current CSE procedure can still achieve the desired trajectory when its **analysis updates** receive no direct information from $x$ or $y$, while its validated fallback retains the current sign of controlled $x$. It does not by itself establish a success rate. A later batch notebook should repeat the same paired comparison over the full event set; that notebook is deliberately deferred until this pilot has been run and checked.
"""
    ),
]


write_notebook("12_gen_obs_z_only.ipynb", notebook_12)
write_notebook("13_ana_ini_z_only_obs.ipynb", notebook_13)
write_notebook("14_run_osse_z_only_obs.ipynb", notebook_14)
write_notebook(
    "15_run_cse_qiwen_z_control_z_obs_nature_qiwenevents.ipynb",
    notebook_15,
)
