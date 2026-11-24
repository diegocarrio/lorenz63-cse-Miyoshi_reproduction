"""Build the z-only experiment with sign-invariant past-safe control."""

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "19_run_cse_qiwen_z_control_z_obs_sign_invariant_qiwenevents.ipynb"


def markdown(source):
    return nbformat.v4.new_markdown_cell(source.strip() + "\n")


def code(source):
    return nbformat.v4.new_code_cell(source.strip() + "\n")


cells = [
    markdown(
        r"""
# z-Only CSE Control with z-Only Observations and Sign-Invariant Past-Safe Search

This notebook performs the information-channel experiment proposed by Takemasa
without using the true state, a selected member, or a consensus regime label
to define which lobe a safe past trajectory must occupy:

- **observation:** $z$ only;
- **intervention:** $u_z$ only;
- **control objective:** retain the initial sign of $x$;
- **past-safe criterion:** $x$ stays strictly on either side of zero throughout
  the aligned forecast window;
- **mixed-sign analysis ensembles:** allowed; regime ambiguity does not stop
  the event.

The notebook uses the independently generated z-only OSSE analysis history from
notebook 14. It writes to a new experiment directory and does not overwrite the
validated nature-reference results from notebook 15.
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
EXPERIMENT_NAME = 'z_control_z_obs_sign_invariant_qiwenevents'
OBSERVED_INDICES = (2,)
CONTROL_INDICES = [2]
REGIME_REFERENCE_MODE = 'sign_invariant'
AMBIGUOUS_REGIME_POLICY = 'continue'
CANDIDATE_POOL = 'full'

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
print('Past-safe mode:', REGIME_REFERENCE_MODE)
print('Mixed-sign analysis handling:', AMBIGUOUS_REGIME_POLICY)
print('Past-safe candidate pool:', CANDIDATE_POOL)
"""
    ),
    markdown(
        """
## Load the z-Only OSSE and Validate the Information Boundary

The DA update receives only the noisy $z$ observation. The CSE core still
generates three observation-noise draws so that this experiment remains paired
with the existing full-observation calculation, but the adapter passes only the
$z$ component to the analysis update.
"""
    ),
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
print('Ensemble size:', das.edim)
print('Observed indices:', OBSERVED_INDICES)
print('Controlled indices:', CONTROL_INDICES)
print('DA-cycle length:', Ta)
print('Forecast horizon in model steps:', position)
print('OSSE analysis history:', osse_Xa_history.shape)
"""
    ),
    markdown(
        """
## Select the Same Qiwen Event Family

Nature-run crossings are used only to define a reproducible evaluation set and
aligned initial times. `INITIAL_NATURE_LOBE` can select events whose nature
state at `t0` is positive, negative, or unrestricted; this only selects the
evaluation window and does not enter the cycle-by-cycle intervention rule.
`EVENT_FAMILY='qiwen'` preserves the residence-filtered Qiwen events;
`EVENT_FAMILY='all_crossings'` allows any detected nature crossing and can
provide lobe-specific starts that are absent from the shorter Qiwen subset.
The selected `t0` must also exist in the saved z-only OSSE history; if it does
not, the cell reports the additional OSSE coverage required.
After initialization, the controller obtains its regime information only from
the current z-only analysis ensemble.
"""
    ),
    code(
        """
regime_change_info = cse_qiwen.detect_regime_changes(
    x_nature,
    RESULTS_DIR,
    position=position,
    output_filename='change_region_time_z_obs_sign_invariant_qiwenevents.pkl',
event_selection='qiwen',
)

D = 0.05
experiment_step = 1000
# 'qiwen' preserves Qiwen's residence-filtered event family; 'all_crossings'
# uses all detected nature-run crossings (and may select a different sample).
EVENT_FAMILY = 'qiwen'
# Choose the nature-run lobe at the CSE initialization time t0:
# 'positive', 'negative', or 'any'. EVENT_INDEX is the zero-based position
# within the events matching EVENT_FAMILY and this lobe selection.
INITIAL_NATURE_LOBE = 'positive'
EVENT_INDEX = 1

if EVENT_FAMILY not in {'qiwen', 'all_crossings'}:
    raise ValueError("EVENT_FAMILY must be 'qiwen' or 'all_crossings'.")
if INITIAL_NATURE_LOBE not in {'positive', 'negative', 'any'}:
    raise ValueError(
        "INITIAL_NATURE_LOBE must be 'positive', 'negative', or 'any'."
    )

source_event_times = (
    regime_change_info['t_start_series']
    if EVENT_FAMILY == 'qiwen'
    else regime_change_info['change_region_time']
)
eligible_events = []
for source_event_index, event_time in enumerate(source_event_times):
    candidate_t_start = int(event_time)
    candidate_t0 = (candidate_t_start - position) - (candidate_t_start - position) % Ta
    candidate_osse_index = (candidate_t0 - osse_summary['t0']) // Ta
    if not (0 <= candidate_t0 < x_nature.shape[1]):
        continue
    if candidate_osse_index < 0:
        continue
    if osse_summary['t0'] + candidate_osse_index * Ta != candidate_t0:
        continue
    initial_x = float(x_nature[0, candidate_t0])
    candidate_lobe = (
        'positive' if initial_x > 0.0
        else 'negative' if initial_x < 0.0
        else 'zero'
    )
    if INITIAL_NATURE_LOBE == 'any' or candidate_lobe == INITIAL_NATURE_LOBE:
        eligible_events.append((
            source_event_index,
            candidate_t_start,
            candidate_t0,
            int(candidate_osse_index),
            initial_x,
            candidate_lobe,
        ))

if EVENT_INDEX < 0 or EVENT_INDEX >= len(eligible_events):
    raise IndexError(
        f'EVENT_INDEX={EVENT_INDEX} is outside the {len(eligible_events)} events '
        f'matching INITIAL_NATURE_LOBE={INITIAL_NATURE_LOBE!r}. '
        'Choose another EVENT_INDEX or lobe selection.'
    )

(
    SOURCE_EVENT_INDEX,
    t_start,
    t0,
    osse_start_index,
    initial_nature_x,
    selected_nature_lobe,
) = eligible_events[EVENT_INDEX]
t_start_series = [int(row[1]) for row in eligible_events]

if osse_start_index >= len(osse_Xa_history):
    available_last_index = len(osse_Xa_history) - 1
    available_last_time = osse_summary['t0'] + available_last_index * Ta
    required_history_length = osse_start_index + 1
    raise IndexError(
        f"The selected {selected_nature_lobe}-lobe event (Qiwen event "
        f"{SOURCE_EVENT_INDEX}) needs the z-only OSSE analysis at t0={t0} "
        f"(history index {osse_start_index}), but the saved OSSE contains "
        f"{len(osse_Xa_history)} ensembles and ends at t={available_last_time}. "
        f"Extend the z-only OSSE to at least {required_history_length} "
        'analysis ensembles, or choose another event.'
    )

initial_consensus = cse_qiwen.infer_analysis_ensemble_regime(
    osse_Xa_history[osse_start_index]
)

print('Requested initial nature lobe:', INITIAL_NATURE_LOBE)
print('Event family:', EVENT_FAMILY)
print('Selected lobe at t0:', selected_nature_lobe)
print('Selected nature x at t0:', initial_nature_x)
print('Index within selected-lobe events:', EVENT_INDEX)
print('Original Qiwen event index:', SOURCE_EVENT_INDEX)
print('Number of matching events:', len(eligible_events))
print('Crossing used to define the test window:', t_start)
print('Aligned CSE start:', t0)
print('z-only OSSE start index:', osse_start_index)
print('Initial analysis x values:', initial_consensus['x_values'])
print('Initial ensemble-regime status:', initial_consensus['status'])
"""
    ),
    markdown(
        r"""
## Visual Check of the CSE Initialization

This pre-run plot checks that the aligned start `t0` precedes the selected
nature-run event and shows the inherited z-only OSSE analysis ensemble at that
time. It is a diagnostic only; the nature state is not passed to the
controller's branch or intervention calculations.
"""
    ),
    code(
        """
# Pre-run visual check of the CSE initialization.
# This confirms that t0 is correctly placed before the selected event and
# shows where the inherited OSSE analysis ensemble sits relative to nature.
Xa_initial = np.asarray(osse_Xa_history[osse_start_index], dtype=float)
x_original = np.asarray(x_nature[:, t0], dtype=float).ravel().copy()
rmse_init = np.linalg.norm(Xa_initial.mean(axis=1) - x_original) / np.sqrt(das.xdim)

fig_cse_init, ax_cse_init, cse_init_path = cse_qiwen.plot_cse_initialization(
    x_nature=x_nature,
    Xa_initial=Xa_initial,
    t0=t0,
    t_start=t_start,
    osse_start_index=osse_start_index,
    rmse_init=rmse_init,
    path_figures=FIGURES_DIR,
    output_filename='cse_initialization_z_obs_sign_invariant_qiwenevents.png',
)
print('Initial analysis RMSE to nature (visual diagnostic):', rmse_init)
print('Initialization figure:', cse_init_path)
"""
    ),
    markdown(
        r"""
## Run One z-Observation / z-Control Event with Sign-Invariant Past-Safe Search

At the beginning of every DA cycle, each forecast member is classified only by
whether its own $x$ trajectory crosses zero in the forecast window. Analysis
members may occupy either lobe or both; a mixed-sign ensemble is recorded as a
diagnostic but does not terminate the event. The CSE branching is:

- no member predicts a crossing: free evolution;
- some members predict a crossing: `current_safe_control`;
- all members predict a crossing: `past_safe_control`, searching earlier
  analysis ensembles for a trajectory that does not cross zero, on either
  lobe. The intervention remains the difference between that safe trajectory
  and a current crossing forecast; only the $z$ component is applied.

Only $z$ increments are retained when the intervention packet is constructed.
`candidate_pool='full'` tests all members of each past ensemble, preserving the
existing candidate-selection policy.
"""
    ),
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
    regime_reference=REGIME_REFERENCE_MODE,
    ambiguous_regime_policy=AMBIGUOUS_REGIME_POLICY,
    candidate_pool=CANDIDATE_POOL,
    control_indices=CONTROL_INDICES,
    T_0=T_0,
    multiple_number=multiple_number,
)

D_TAG = f'{D:g}'.replace('.', 'p')
FULL_RESULT_PATH = RESULTS_DIR / (
    f'cse_z_control_z_obs_sign_invariant_event_{EVENT_INDEX}_D{D_TAG}_qiwenevents.pkl'
)
temporary_path = FULL_RESULT_PATH.with_suffix(FULL_RESULT_PATH.suffix + '.tmp')
with temporary_path.open('wb') as stream:
    pickle.dump(single_cse_result, stream, pickle.HIGHEST_PROTOCOL)
temporary_path.replace(FULL_RESULT_PATH)

n_completed = int(single_cse_result['n_completed'])
COMPARISON_PATH = None
if n_completed > 0:
    COMPARISON_PATH = RESULTS_DIR / (
        'cse_z_control_z_obs_sign_invariant_nature_run_'
        f'event_{EVENT_INDEX}_D{D_TAG}_qiwenevents.pkl'
    )
    cse_qiwen.export_nature_run_comparison(
        single_cse_result,
        x_nature,
        COMPARISON_PATH,
    )

if n_completed > 0:
    assert single_cse_result['yo_history'].shape[1] == 1
    assert single_cse_result['yo_history_full_generated'].shape[1] == 3
    assert np.allclose(
        single_cse_result['yo_history'][:, 0],
        single_cse_result['yo_history_full_generated'][:, 2],
    )

print('Success:', single_cse_result['success'])
print('Failure cycle:', single_cse_result['failure_cycle'])
print('Failure reason:', single_cse_result['failure_reason'])
print('Completed cycles:', n_completed)
print('Control ratio:', single_cse_result['control_ratio'])
print('Branch counts:', single_cse_result['branch_counts'])
print('Ambiguous regime cycles:', single_cse_result['ambiguous_regime_cycles'].tolist())
print('Assimilated observation shape:', single_cse_result['yo_history'].shape)
print('Full result:', FULL_RESULT_PATH)
print('Nature-run comparison:', COMPARISON_PATH)
"""
    ),
    markdown(
        """
## Analysis-Ensemble Sign Diagnostic (Not a Control Gate)

This diagnostic displays every current analysis-member value of $x$ and marks
mixed-sign cycles. These labels are descriptive only: neither the current
lobe's sign nor ensemble consensus is used by `past_safe_control`.
"""
    ),
    code(
        """
consensus_history = single_cse_result['regime_consensus_history']
if not consensus_history:
    print('No regime-consensus diagnostic was recorded.')
else:
    consensus_cycles = np.asarray(
        [row['cycle_index'] for row in consensus_history], dtype=int
    )
    consensus_x = np.asarray(
        [row['x_values'] for row in consensus_history], dtype=float
    )
    consensus_sign = np.asarray(
        [row['regime_sign'] for row in consensus_history], dtype=float
    )

    n_positive_consensus = int(np.sum(consensus_sign > 0))
    n_negative_consensus = int(np.sum(consensus_sign < 0))
    n_ambiguous = int(np.sum(consensus_sign == 0))

    fig_consensus, ax_consensus = plt.subplots(
        figsize=(11, 5.2), constrained_layout=True
    )
    for member in range(consensus_x.shape[1]):
        ax_consensus.plot(
            consensus_cycles,
            consensus_x[:, member],
            marker='o',
            markersize=3.0,
            linewidth=0.9,
            alpha=0.7,
            label=f'analysis member {member}',
        )
    ax_consensus.fill_between(
        consensus_cycles,
        consensus_x.min(axis=1),
        consensus_x.max(axis=1),
        color='0.5',
        alpha=0.15,
        label='ensemble x range',
    )
    ambiguous_mask = consensus_sign == 0
    if np.any(ambiguous_mask):
        ax_consensus.scatter(
            consensus_cycles[ambiguous_mask],
            np.zeros(np.sum(ambiguous_mask)),
            marker='x',
            s=65,
            linewidths=1.5,
            color='black',
            label='ambiguous regime',
            zorder=5,
        )
    ax_consensus.axhline(0.0, color='black', linestyle='--', linewidth=0.9)
    ax_consensus.set_xlabel('DA cycle')
    ax_consensus.set_ylabel('Current analysis-member x')
    ax_consensus.set_title(
        'Current regime inferred from unanimous analysis-ensemble signs'
    )
    ax_consensus.grid(alpha=0.2)
    ax_consensus.legend(
        frameon=False,
        ncol=2,
        loc='upper left',
        bbox_to_anchor=(1.01, 1.0),
    )

    consensus_figure_path = (
        FIGURES_DIR / 'z_obs_sign_invariant_regime_diagnostic.png'
    )
    fig_consensus.savefig(
        consensus_figure_path, dpi=300, bbox_inches='tight'
    )
    plt.show()

    print('Positive-consensus cycles:', n_positive_consensus)
    print('Negative-consensus cycles:', n_negative_consensus)
    print('Ambiguous cycles:', n_ambiguous)
    print('Figure:', consensus_figure_path)
"""
    ),
    markdown(
        r"""
## z-Only Analysis Ensemble and Lorenz Symmetry

This figure compares the analysis ensemble with the controlled state and its
Lorenz-symmetric counterpart,

\[
(x,y,z) \longrightarrow (-x,-y,z).
\]

It is generated only when the experiment completes at least one cycle. If the
initial ensemble is already ambiguous, the preceding consensus figure remains
the relevant diagnostic.
"""
    ),
    code(
        """
analysis_ensembles = np.asarray(
    single_cse_result['Xa_cycle_initial_history'], dtype=float
)

if analysis_ensembles.size == 0:
    print('No completed cycle is available for the symmetry comparison.')
else:
    n_analysis_cycles = analysis_ensembles.shape[0]
    analysis_steps = np.arange(n_analysis_cycles) * Ta
    controlled_at_analysis = np.asarray(
        single_cse_result['evo_history'], dtype=float
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
        'x-sign agreement = {:.1f}%\\n'
        'Mean RMSE to controlled state = {:.2f}\\n'
        'Mean RMSE to symmetry counterpart = {:.2f}'.format(
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
        'z-only analysis ensemble and its Lorenz-symmetric counterpart', y=0.995
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

    ensemble_symmetry_path = (
        FIGURES_DIR
        / 'z_only_analysis_ensemble_symmetry_sign_invariant.png'
    )
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
if n_completed == 0:
    print('No completed cycle is available for the free/control comparison.')
else:
    n_steps = n_completed * Ta
    controlled = np.asarray(
        single_cse_result['evo_history'], dtype=float
    )[:, :n_steps + 1]
    free = x_nature[
        :, single_cse_result['t0']:single_cse_result['t0'] + n_steps + 1
    ]
    relative_steps = np.arange(n_steps + 1)

    interventions = np.zeros_like(controlled)
    for cycle, item in enumerate(single_cse_result['intervention_history']):
        start = cycle * Ta
        interventions[:, start:start + Ta] = np.asarray(item['perturbations']).T

    fig, axes = plt.subplots(4, 1, figsize=(11, 9.5), sharex=True)
    labels = ('x', 'y', 'z')
    for state_index, ax in enumerate(axes[:3]):
        ax.plot(
            relative_steps,
            free[state_index],
            color='0.35',
            linewidth=1.15,
            label='free nature run',
        )
        ax.plot(
            relative_steps,
            controlled[state_index],
            color='tab:blue',
            linestyle='--',
            linewidth=1.35,
            label='controlled nature run',
        )
        if state_index < 2:
            ax.axhline(0.0, color='0.75', linewidth=0.8)
        ax.set_ylabel(labels[state_index])
        ax.grid(alpha=0.2)

    nonzero = np.flatnonzero(np.abs(interventions[2]) > 0)
    axes[3].vlines(
        nonzero,
        0.0,
        interventions[2, nonzero],
        color='tab:red',
        linewidth=0.8,
    )
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
        'z-only observations and z-only interventions: sign-invariant past-safe control',
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    comparison_figure_path = (
        FIGURES_DIR
        / 'z_obs_z_control_sign_invariant_free_vs_controlled.png'
    )
    fig.savefig(comparison_figure_path, dpi=300, bbox_inches='tight')
    plt.show()

    print('Figure:', comparison_figure_path)
"""
    ),
    markdown("## Three-Dimensional Attractor Comparison"),
    code(
        """
if n_completed == 0:
    print('No completed cycle is available for the 3D comparison.')
else:
    plot_3d_data = cse_qiwen.plot_3d_nature_run_comparison_with_fallback(
        current_result=single_cse_result,
        x_nature=x_nature,
        Ta=Ta,
        path_figures=FIGURES_DIR,
        analysis_data_path=ANALYSIS_FILE,
        results_dir=RESULTS_DIR,
        output_tag='_z_obs_sign_invariant_qiwenevents',
        D=D,
        event_index=EVENT_INDEX,
        experiment_step=experiment_step,
        control_label='z_obs_z_control_sign_invariant',
        output_filename=(
            'NR_3D_comparison_z_obs_z_control_sign_invariant.png'
        ),
    )
    print('3D figure:', plot_3d_data['output_path'])
"""
    ),
    markdown(
        """
## Three-Dimensional Free Trajectory versus z-Only Analysis Ensemble

This companion figure compares the free trajectory with the analysis ensemble
at every completed DA-cycle initialization. It shows the symmetry-related
lobe structure; the control rule itself does not assign a current-lobe label.
"""
    ),
    code(
        """
if analysis_ensembles.size == 0:
    print('No completed cycle is available for the 3D analysis-ensemble figure.')
else:
    n_steps_3d = n_completed * Ta
    t0_3d = int(single_cse_result['t0'])
    free_trajectory_3d = np.asarray(
        x_nature[:, t0_3d:t0_3d + n_steps_3d + 1], dtype=float
    )
    analysis_mean_3d = analysis_ensembles.mean(axis=2)

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

    for member in range(analysis_ensembles.shape[2]):
        analysis_artist = ax_analysis.scatter(
            analysis_ensembles[:, 0, member],
            analysis_ensembles[:, 1, member],
            analysis_ensembles[:, 2, member],
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
            np.min(analysis_ensembles[:, 0, :]),
            np.max(analysis_ensembles[:, 0, :]),
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
    fig.suptitle(
        'Free trajectory versus z-only analysis ensemble: sign-invariant experiment',
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))

    analysis_3d_path = (
        FIGURES_DIR
        / 'free_nature_vs_z_only_analysis_ensemble_sign_invariant_3d.png'
    )
    fig.savefig(analysis_3d_path, dpi=300, bbox_inches='tight')
    plt.show()

    print('Analysis x range:', float(np.min(analysis_ensembles[:, 0, :])),
          float(np.max(analysis_ensembles[:, 0, :])))
    print('3D analysis-ensemble figure:', analysis_3d_path)
"""
    ),
    markdown(
        """
## Interpretation Boundary

This experiment tests whether the existing CSE intervention construction can
use a past trajectory that avoids zero-crossing without knowing which lobe the
analysis ensemble or nature run occupies. The nature run is used to initialize
and evaluate the experiment, but its current sign and future trajectory do not
enter the branch selection or the past-safe search. Mixed-sign analysis
ensembles are allowed and are retained as diagnostics, not treated as failures.
"""
    ),
]


notebook = nbformat.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Control",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.9",
        },
    },
)

nbformat.validate(notebook)
with TARGET.open("w", encoding="utf-8") as handle:
    nbformat.write(notebook, handle)

print(TARGET)
