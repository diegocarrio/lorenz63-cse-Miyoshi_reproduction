"""Create notebook 11 and append its plotting section to notebook 10."""

from pathlib import Path
import sys

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NOTEBOOK = PROJECT_ROOT / "11_cse_qiwen_z_only_ensemble_counterfactuals.ipynb"
MECHANISM_NOTEBOOK = PROJECT_ROOT / "10_cse_qiwen_z_only_control_mechanism.ipynb"
SECTION_MARKER = "## Methodological Boundary: CSE Decisions versus Posterior Verification"
BUILD_EXPERIMENT_NOTEBOOK = "--mechanism-only" not in sys.argv


def markdown(source):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source):
    return nbf.v4.new_code_cell(source.strip() + "\n")


experiment_cells = [
    markdown(
        r"""
# Ensemble Counterfactual Diagnostics for z-Only CSE

This notebook investigates **why the interventions selected by the existing CSE algorithm work when only $z$ is directly perturbed**. It does not modify the CSE implementation or use the future uncontrolled nature-run transition to decide when to intervene.

The original CSE run is first reproduced with the same configuration and random seed. Counterfactual forecasts are then launched only at cycles that the original ensemble forecast already caused CSE to control. Each analysis ensemble is propagated under three diagnostic alternatives: no intervention, the stored CSE-selected $u_z$, and the opposite $-u_z$. These branches do not feed back into CSE.
"""
    ),
    code(
        """
import importlib
import pickle

import numpy as np

import project_paths as project_paths
import class_cse_qiwen as cse_qiwen
import class_cse_counterfactual_diagnostics as counterfactual

project_paths = importlib.reload(project_paths)
cse_qiwen = importlib.reload(cse_qiwen)
counterfactual = importlib.reload(counterfactual)

from class_da_sys import da_system

EXPERIMENT_NAME = 'z_control_nature_qiwenevents'
EXPERIMENT_ROOT, RESULTS_DIR, FIGURES_DIR = project_paths.cse_experiment_paths(
    EXPERIMENT_NAME
)
DIAGNOSTICS_DIR = EXPERIMENT_ROOT / 'diagnostics'
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_PATH = (
    RESULTS_DIR / 'cse_z_only_nature_run_event_1_D0p05_qiwenevents.pkl'
)
OUTPUT_PATH = (
    DIAGNOSTICS_DIR
    / 'cse_z_only_ensemble_counterfactuals_event_1_D0p05.pkl'
)

print('Reference result:', REFERENCE_PATH)
print('Counterfactual output:', OUTPUT_PATH)
"""
    ),
    markdown(
        r"""
## Reproduce the Original CSE Run

The compact reference file contains the controlled and uncontrolled nature trajectories and the applied interventions, but not the complete forecast-ensemble history. Therefore, the unchanged authoritative `run_single_cse` function is executed again with the saved configuration. The resulting controlled trajectory, branch history, and intervention array must match the reference before any counterfactual is accepted.

The saved list of regime crossings is used only to reconstruct the same event index that initialized the original experiment. No crossing from the uncontrolled trajectory is used to select the diagnostic cycles or counterfactual interventions.
"""
    ),
    code(
        """
with open(REFERENCE_PATH, 'rb') as stream:
    reference = pickle.load(stream)

summary = dict(reference['summary'])
das = da_system().load(project_paths.ANALYSIS_DATA_DIR / 'x_analysis_init.pkl')
das.multiplicative_inflation = 1.04
x_nature = das.getStateVector().getTrajectory()

with open(project_paths.OSSE_DATA_DIR / 'osse_Xa_history.pkl', 'rb') as stream:
    osse_Xa_history = pickle.load(stream)
with open(project_paths.OSSE_DATA_DIR / 'osse_summary.pkl', 'rb') as stream:
    osse_summary = pickle.load(stream)
with open(RESULTS_DIR / 'change_region_time_qiwen_qiwenevents.pkl', 'rb') as stream:
    change_region_time = pickle.load(stream)

# Exact Qiwen event filtering used by the original notebook.
t_start_series = [
    crossing
    for index, crossing in enumerate(change_region_time)
    if crossing >= 500 and crossing - change_region_time[index - 1] > 385
][:100]

print('Event index:', summary['run_cse_event_index'])
print('Selected event time:', t_start_series[summary['run_cse_event_index']])
print('Saved t_start:', summary['t_start'])
assert t_start_series[summary['run_cse_event_index']] == summary['t_start']
"""
    ),
    code(
        """
single_cse_result = cse_qiwen.run_single_cse(
    event_index=summary['run_cse_event_index'],
    D_run=summary['D'],
    seed=summary['seed'],
    das=das,
    x_nature=x_nature,
    t_start_series=t_start_series,
    osse_Xa_history=osse_Xa_history,
    osse_summary=osse_summary,
    experiment_step=summary['requested_num_cycles'],
    position=summary['position'],
    Ta=summary['Ta'],
    fcst_dt=summary['fcst_dt'],
    path_figures=None,
    total_events=len(t_start_series),
    regime_reference=summary['regime_reference'],
    candidate_pool=summary['candidate_pool'],
    control_indices=summary['control_indices'],
    T_0=summary['T_0'],
    multiple_number=summary['multiple_number'],
    cle_target_output_dir=None,
)
"""
    ),
    code(
        """
Ta = int(summary['Ta'])
n_steps = int(single_cse_result['n_completed']) * Ta
reproduced_control = np.asarray(single_cse_result['evo_history'])[:, :n_steps + 1]
reproduced_interventions = np.zeros_like(reproduced_control)
for cycle, item in enumerate(single_cse_result['intervention_history']):
    start = cycle * Ta
    reproduced_interventions[:, start:start + Ta] = np.asarray(
        item['perturbations']
    ).T

control_error = float(np.max(np.abs(
    reproduced_control - np.asarray(reference['nature_control'])
)))
intervention_error = float(np.max(np.abs(
    reproduced_interventions - np.asarray(reference['interventions'])
)))
branches_match = (
    list(single_cse_result['branch_history']) == list(reference['branch_history'])
)

assert control_error <= 1e-12
assert intervention_error <= 1e-12
assert branches_match

print('Maximum controlled-trajectory error:', control_error)
print('Maximum intervention error:', intervention_error)
print('Branch histories match:', branches_match)
print('Completed cycles:', single_cse_result['n_completed'])
print('Branch counts:', single_cse_result['branch_counts'])
"""
    ),
    markdown(
        r"""
## Counterfactual Ensemble Forecasts

For every cycle already controlled by CSE, all ensemble members start from the same saved analysis ensemble. The diagnostic branches differ only in the intervention packet:

\[
u_z^{\mathrm{free}}=0,\qquad
u_z^{\mathrm{CSE}}=u_z,\qquad
u_z^{\mathrm{opposite}}=-u_z.
\]

The increments retain the original discrete alignment: row zero is zero and rows $1, ..., T_a-1$ are applied after their corresponding free RK4 advances. Risk metrics use the original 301-step CSE forecast horizon. For Figure 6 only, the same post-decision branches are continued freely to step 400 so that their longer-term separation remains visible.
"""
    ),
    code(
        """
diagnostics = counterfactual.build_counterfactual_diagnostics(
    single_cse_result,
    position=summary['position'],
    Ta=summary['Ta'],
    dt=summary['fcst_dt'],
    representative_position=400,
)
counterfactual.save_counterfactual_diagnostics(diagnostics, OUTPUT_PATH)

print('Controlled cycles analysed:', len(diagnostics['controlled_cycles']))
print('Risk-metric horizon:', diagnostics['position'])
print('Representative-figure horizon:', diagnostics['representative_position'])
print('Representative cycles:')
for cycle, record in diagnostics['representative_cycles'].items():
    print(f"  {cycle}: {record['selection_label']}")
print('Maximum free-forecast replay error:',
      diagnostics['validation']['maximum_free_forecast_abs_error'])
print('All interventions are z-only:',
      diagnostics['validation']['all_interventions_z_only'])
print('All crossing counts reproduce CSE:',
      diagnostics['validation']['all_free_crossing_counts_match_cse'])
print('Saved:', OUTPUT_PATH)
"""
    ),
    code(
        """
aggregate = diagnostics['aggregate']
print('Mean crossing fraction by branch:')
for branch, value in aggregate['mean_crossing_fraction'].items():
    print(f'  {branch:>8}: {value:.6f}')

print('Fraction of cycles where CSE has lower risk than no intervention:',
      aggregate['cse_lower_risk_than_free_fraction'])
print('Fraction of cycles where CSE has higher risk than no intervention:',
      aggregate['cse_higher_risk_than_free_fraction'])
print('Fraction of cycles where CSE has a higher margin than no intervention:',
      aggregate['cse_higher_margin_than_free_fraction'])
print('Fraction of cycles where CSE has lower risk than the opposite action:',
      aggregate['cse_lower_risk_than_opposite_fraction'])
print('Fraction of cycles where CSE has a higher margin than the opposite action:',
      aggregate['cse_higher_margin_than_opposite_fraction'])
"""
    ),
    markdown(
        r"""
## Interpretation Boundary

These are **post-decision causal diagnostics**, not a replacement controller. The uncontrolled nature trajectory is not used to choose a cycle, intervention sign, or intervention onset. It remains available only for the separate, posterior verification performed in the mechanism notebook.

Because each diagnostic branch isolates one intervention packet and does not alter later CSE decisions, the results measure the local forecast effect of the selected $u_z$. The complete controlled nature run still represents the cumulative closed-loop result of repeatedly forecasting, intervening, observing, and assimilating.
"""
    ),
]


experiment_notebook = nbf.v4.new_notebook(
    cells=experiment_cells,
    metadata={
        "kernelspec": {
            "display_name": "Control",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.9.18"},
    },
)
if BUILD_EXPERIMENT_NOTEBOOK:
    nbf.write(experiment_notebook, EXPERIMENT_NOTEBOOK)


mechanism_notebook = nbf.read(MECHANISM_NOTEBOOK, as_version=4)
for marker_index, cell in enumerate(mechanism_notebook.cells):
    if cell.cell_type == "markdown" and SECTION_MARKER in cell.source:
        mechanism_notebook.cells = mechanism_notebook.cells[:marker_index]
        break
if not any(
    cell.cell_type == "markdown" and SECTION_MARKER in cell.source
    for cell in mechanism_notebook.cells
):
    mechanism_notebook.cells.extend(
        [
            markdown(
                r"""
## Methodological Boundary: CSE Decisions versus Posterior Verification

The free nature-run crossing at step 306 is useful only for verifying the realized outcome after the experiment. It is not available to CSE in real time and is not used below to determine when an intervention should occur.

The new diagnostic starts exclusively from cycles that the original CSE algorithm already selected from its ensemble forecasts. At each such cycle, the stored $z$-only intervention is replayed on copies of the same analysis ensemble and compared with zero and opposite interventions. These counterfactual branches never feed back into the working CSE run.
"""
            ),
            code(
                """
import importlib
import class_cse_counterfactual_diagnostics as counterfactual

counterfactual = importlib.reload(counterfactual)

COUNTERFACTUAL_PATH = (
    RESULTS_DIR.parent
    / 'diagnostics'
    / 'cse_z_only_ensemble_counterfactuals_event_1_D0p05.pkl'
)
counterfactual_data = counterfactual.load_counterfactual_diagnostics(
    COUNTERFACTUAL_PATH
)

print('Counterfactual data:', COUNTERFACTUAL_PATH)
print('Cycle-selection policy:',
      counterfactual_data['information_policy']['cycle_selection'])
print('Future free nature used for cycle selection:',
      counterfactual_data['information_policy'][
          'uncontrolled_nature_future_used_to_select_cycles'
      ])
print('Controlled cycles analysed:',
      len(counterfactual_data['controlled_cycles']))
"""
            ),
            markdown(
                r"""
## Figure 5 — Ensemble Counterfactuals at CSE-Selected Cycles

This figure compares the forecast obtained with no intervention, the $u_z$ packet actually selected by CSE, and the same packet with the opposite sign. All branches start from the same analysis ensemble at the same CSE-selected cycles.

The upper panel shows the ensemble fraction predicted to change regime. The middle panel subtracts the no-intervention risk, so negative values indicate fewer predicted crossings. The lower panel compares the median minimum signed-$x$ margin over the forecast horizon; positive values indicate that the ensemble remains farther from $x=0$. A rolling mean is used only to make the cycle-to-cycle pattern readable and does not affect any calculation.
"""
            ),
            code(
                """
(
    fig_counterfactual_risk,
    axes_counterfactual_risk,
    counterfactual_risk_png,
    counterfactual_risk_pdf,
    counterfactual_aggregate,
) = counterfactual.plot_counterfactual_risk_summary(
    counterfactual_data,
    FIGURES_DIR,
    rolling_window=25,
)
plt.show()

print('Mean crossing fractions:',
      counterfactual_aggregate['mean_crossing_fraction'])
print('CSE lower risk than no intervention:',
      counterfactual_aggregate['cse_lower_risk_than_free_fraction'])
print('CSE higher margin than no intervention:',
      counterfactual_aggregate['cse_higher_margin_than_free_fraction'])
print('Saved PNG:', counterfactual_risk_png)
print('Saved PDF:', counterfactual_risk_pdf)
"""
            ),
            markdown(
                r"""
### Aggregate Result

Across the 690 CSE-selected cycles, the mean predicted crossing fraction is 0.804 with no intervention, 0.752 with the CSE-selected intervention, and 0.836 with the opposite intervention. Thus, the selected sign of $u_z$ produces a net shift toward safer ensemble forecasts, whereas reversing its sign shifts the ensemble in the unfavorable direction.

The single-packet crossing classification is unchanged in most cycles because the ensemble contains only three members and the risk can therefore change only in increments of $1/3$. Even with this coarse measure, CSE lowers the crossing fraction in 18.8% of controlled cycles and raises it in only 4.6%. The more continuous minimum signed-$x$ margin improves relative to no intervention in 55.7% of cycles and relative to the opposite intervention in 58.4%.

This is evidence of a systematic directional effect, not a claim that every individual packet immediately removes all predicted crossings. The realized control is cumulative and closed-loop: CSE repeats the forecast–intervention–assimilation cycle 1,000 times.
"""
            ),
            markdown(
                r"""
## Figure 6 — Forecast Geometry at an Ensemble-Selected Cycle

The representative cycle is the first cycle selected for intervention by the original CSE ensemble forecast—not a cycle chosen from the hidden nature trajectory. Each panel shows the ensemble median and 10–90% interval for one state variable through forecast step 400. The original CSE decision horizon remains step 301; steps 302–400 are a free diagnostic continuation of the same three branches. The pale green interval marks the seven model steps during which the discrete intervention packet is applied.

This view connects the algorithmic decision to the dynamical mechanism: the three branches first separate directly in $z$, after which the Lorenz coupling transmits that difference to $y$ and then to $x$.
"""
            ),
            code(
                """
(
    fig_counterfactual_cycle,
    axes_counterfactual_cycle,
    counterfactual_cycle_png,
    counterfactual_cycle_pdf,
    counterfactual_cycle_record,
) = counterfactual.plot_representative_counterfactual(
    counterfactual_data,
    FIGURES_DIR,
)
plt.show()

print('Representative cycle:',
      counterfactual_cycle_record['cycle_index'])
print('Selection rule:',
      counterfactual_cycle_record['selection_label'])
print('Stored z-intervention packet:',
      counterfactual_cycle_record['z_intervention'])
print('Saved PNG:', counterfactual_cycle_png)
print('Saved PDF:', counterfactual_cycle_pdf)
"""
            ),
            markdown(
                r"""
## Figure 7 — z-Only Interventions on the Lorenz Attractor

The left panel places the first 400 model steps on the full free nature-run attractor. The orange path is the uncontrolled evolution, which reaches the negative-$x$ lobe at step 306. The blue controlled path remains on the positive-$x$ lobe. The two `X` markers compare the free and controlled states at that same step.

The right panel enlarges the controlled positive-lobe trajectory. Magenta points mark every state at which a discrete $u_z$ increment was applied. Red and purple arrows show the sign of representative positive and negative $u_z$ increments, respectively. Their lengths are multiplied by 40 for visibility; their direction remains the exact direct intervention direction, parallel to the $z$ axis. The curved blue trajectory after each intervention shows the subsequent coupled response in $y$ and $x$.

The arrows therefore should not be interpreted as direct pushes in $x$ or $y$. They show where CSE acts on $z$; the Lorenz dynamics then convert those repeated vertical increments into motion that remains on the positive attractor lobe.
"""
            ),
            code(
                """
(
    fig_attractor_interventions,
    axes_attractor_interventions,
    attractor_interventions_png,
    attractor_interventions_pdf,
    attractor_interventions_diagnostics,
) = plot_3d_z_interventions_on_lorenz_attractor(
    nature_data,
    FIGURES_DIR,
    n_steps=400,
    arrow_scale=40.0,
    max_arrows=30,
    output_stem='figure_07_cse_z_only_3d_attractor_interventions',
)
plt.show()

print('Window:', attractor_interventions_diagnostics['window'])
print('First free crossing:',
      attractor_interventions_diagnostics['first_free_crossing'])
print('Intervention states shown:',
      len(attractor_interventions_diagnostics['intervention_steps']))
print('Representative direction arrows:',
      len(attractor_interventions_diagnostics['arrow_steps']))
print('Minimum free x:', attractor_interventions_diagnostics['free_min_x'])
print('Minimum controlled x:',
      attractor_interventions_diagnostics['controlled_min_x'])
print('Saved PNG:', attractor_interventions_png)
print('Saved PDF:', attractor_interventions_pdf)
"""
            ),
            markdown(
                r"""
## Figure 8 — From Direct z Increments to the Dynamical Response

This companion figure retains Figure 7 unchanged while making the causal sequence more explicit.

Panel A provides a clean comparison between the free trajectory and the dashed blue controlled trajectory. Panel B focuses on steps 250–330, the interval surrounding the free transition. Magenta points identify all nonzero $z$-intervention states. Ten red arrows show representative direct $z$ increments, with their orientation indicating the sign of $u_z$. The teal arrows show the corresponding accumulated effect of the complete intervention packet at the end of one DA cycle relative to an otherwise identical no-packet integration.

Panel C returns to actual scale. Circular markers identify the stored model steps and vertical lilac arrows show the actual discrete $u_z$ increments. Starting from the controlled state at cycle 0, the first stored difference is only $\Delta z=-0.05$. During subsequent free RK4 evolution, this becomes a change in $y$ and then in $x$. This panel separates the instantaneous intervention from the model-mediated response without using the uncontrolled nature trajectory to choose the intervention.
"""
            ),
            code(
                """
(
    fig_attractor_response,
    axes_attractor_response,
    attractor_response_png,
    attractor_response_pdf,
    attractor_response_diagnostics,
) = plot_3d_z_intervention_response_story(
    nature_data,
    FIGURES_DIR,
    n_steps=400,
    critical_window=(250, 330),
    detail_cycle=0,
    detail_horizon=24,
    response_curve_marker='o',
    response_interventions_as_arrows=True,
    direct_arrow_scale=40.0,
    response_arrow_scale=8.0,
    max_critical_arrows=10,
    simplify_global_comparison=True,
    simplify_critical_comparison=True,
    equal_all_panels=True,
    compact_top_panels=True,
    show_colorbar=False,
    use_signed_direct_arrow_colors=False,
    show_arrow_scales_in_legend=False,
    direct_arrow_legend_label='representative direct z increments',
    packet_effect_legend_label='corresponding effect after one cycle',
    intervention_state_legend_label='all z-intervention states',
    global_panel_title='Global attractor: free versus controlled path',
    window_label='Interventions around the free regime transition',
    response_panel_title='Actual-scale local response in cycle 0',
    response_annotation_xytext=(2.0, 0.10),
    output_stem='figure_08_cse_z_only_3d_attractor_response_story',
)
plt.show()

print('Critical window:',
      attractor_response_diagnostics['critical_window'])
print('Critical-window intervention states:',
      attractor_response_diagnostics['critical_intervention_count'])
print('Representative direct arrows:',
      attractor_response_diagnostics['critical_direct_arrow_steps'])
print('Actual detail packet u_z:',
      attractor_response_diagnostics['detail_packet_z'])
print('Final local [Δx, Δy, Δz]:',
      attractor_response_diagnostics['detail_final_delta'])
print('Saved PNG:', attractor_response_png)
print('Saved PDF:', attractor_response_pdf)
"""
            ),
            markdown(
                r"""
## Figure 9 — Early Separation during Steps 0–100

This figure replicates the structure of Figure 8 but restricts the trajectory to the first 100 model steps. It is designed to show how the controlled evolution begins to depart from the free nature run well before the uncontrolled regime transition at step 306.

Panel A places this early segment within the geometry of the full Lorenz attractor and directly compares the free and dashed blue controlled trajectories. Panel B enlarges steps 0–100. Magenta points show all nonzero $z$-intervention states, ten red arrows show representative direct increments, and teal arrows show the corresponding accumulated effect after one DA cycle. Equal-time connectors and the time colorbar are omitted to keep the comparison uncluttered.

Panel C isolates the first stored CSE packet at actual scale, using circular state markers and vertical lilac intervention arrows. It shows the causal ordering directly: the trajectories first differ in $z$, and the Lorenz dynamics subsequently transmit that difference to $y$ and then to $x$.
"""
            ),
            code(
                """
(
    fig_early_separation,
    axes_early_separation,
    early_separation_png,
    early_separation_pdf,
    early_separation_diagnostics,
) = plot_3d_z_intervention_response_story(
    nature_data,
    FIGURES_DIR,
    n_steps=100,
    critical_window=(0, 100),
    detail_cycle=0,
    detail_horizon=24,
    response_curve_marker='o',
    response_interventions_as_arrows=True,
    direct_arrow_scale=40.0,
    response_arrow_scale=8.0,
    max_critical_arrows=10,
    show_separation_connectors=False,
    separation_connector_count=6,
    simplify_global_comparison=True,
    simplify_critical_comparison=True,
    equal_all_panels=True,
    compact_top_panels=True,
    show_colorbar=False,
    use_signed_direct_arrow_colors=False,
    show_arrow_scales_in_legend=False,
    direct_arrow_legend_label='representative direct z increments',
    packet_effect_legend_label='corresponding effect after one cycle',
    intervention_state_legend_label='all z-intervention states',
    global_panel_title='Global attractor: free versus controlled path',
    window_label='Early interventions and dynamical response',
    response_panel_title='Actual-scale local response in cycle 0',
    response_annotation_xytext=(2.0, 0.10),
    figure_title=(
        'How z-only control begins to separate from the free nature trajectory'
    ),
    output_stem='figure_09_cse_z_only_early_separation_steps_0_100',
)
plt.show()

print('Early window:', early_separation_diagnostics['critical_window'])
print('State-space separation at step 100:',
      early_separation_diagnostics['separation_at_window_end'])
print('Maximum separation in steps 0–100:',
      early_separation_diagnostics['maximum_separation_in_window'])
print('Representative direct arrows:',
      early_separation_diagnostics['critical_direct_arrow_steps'])
print('Saved PNG:', early_separation_png)
print('Saved PDF:', early_separation_pdf)
"""
            ),
            markdown(
                r"""
## Figure 10 — From the Initial Separation to the Free Regime Transition

Figure 9 remains unchanged. This extended view follows both trajectories from step 0 through step 330, so the free nature run can be seen entering the negative-$x$ regime after its first crossing at step 306.

All three panels use the same width, with Panel C centred below Panels A and B. Panel A provides a clean free-versus-controlled comparison, while Panel B focuses on the intervention geometry and the one-cycle dynamical response. The crossing markers, equal-time separation connectors, and time colorbar are omitted here to keep the comparison direct.

In Panel B, the ten red arrows are representative direct $z$ increments; their orientation indicates whether $u_z$ is positive or negative. The teal arrows show the corresponding difference between the packet and no-packet trajectories after one DA cycle. The magenta points retain all nonzero $z$-intervention states. Arrow lengths are enlarged for visibility.

Panel C uses circular markers to identify each stored model step. The vertical lilac arrows show the actual discrete $u_z$ increments applied during the first DA cycle.
"""
            ),
            code(
                """
(
    fig_transition_separation,
    axes_transition_separation,
    transition_separation_png,
    transition_separation_pdf,
    transition_separation_diagnostics,
) = plot_3d_z_intervention_response_story(
    nature_data,
    FIGURES_DIR,
    n_steps=330,
    critical_window=(0, 330),
    detail_cycle=0,
    detail_horizon=24,
    response_curve_marker='o',
    response_interventions_as_arrows=True,
    direct_arrow_scale=40.0,
    response_arrow_scale=8.0,
    max_critical_arrows=10,
    show_separation_connectors=False,
    separation_connector_count=8,
    highlight_free_crossing=False,
    simplify_global_comparison=True,
    simplify_critical_comparison=True,
    equal_top_panels=False,
    equal_all_panels=True,
    compact_top_panels=True,
    show_colorbar=False,
    use_signed_direct_arrow_colors=False,
    signed_direct_arrow_legend=False,
    show_arrow_scales_in_legend=False,
    direct_arrow_legend_label='representative direct z increments',
    packet_effect_legend_label='corresponding effect after one cycle',
    intervention_state_legend_label='all z-intervention states',
    global_panel_title='Global attractor: free versus controlled path',
    window_label='Interventions and dynamical response',
    response_panel_title='Actual-scale local response in cycle 0',
    response_annotation_xytext=(2.0, 0.10),
    figure_title=(
        'From the initial z-only control response to the free-path regime shift'
    ),
    output_stem=(
        'figure_10_cse_z_only_separation_to_negative_regime_steps_0_330'
    ),
)
plt.show()

print('Extended window:',
      transition_separation_diagnostics['critical_window'])
print('First free crossing:',
      transition_separation_diagnostics['first_free_crossing'])
print('State-space separation at step 330:',
      transition_separation_diagnostics['separation_at_window_end'])
print('Maximum separation in steps 0–330:',
      transition_separation_diagnostics['maximum_separation_in_window'])
print('Saved PNG:', transition_separation_png)
print('Saved PDF:', transition_separation_pdf)
"""
            ),
        ]
    )
    nbf.write(mechanism_notebook, MECHANISM_NOTEBOOK)


if BUILD_EXPERIMENT_NOTEBOOK:
    print(f"Wrote {EXPERIMENT_NOTEBOOK}")
print(f"Updated {MECHANISM_NOTEBOOK}")
