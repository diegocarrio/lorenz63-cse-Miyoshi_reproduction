"""Append the Figure 3(a) T-D grid to the isolated same-lobe notebook.

The grid saves one pickle per event and reuses the existing 4T0, D=0.05
pilot files. It never reads or writes sign_invariant result files.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "7_run_cse_qiwen_xyz_control_sign_invariant_qiwenevents_batch.ipynb"
TARGET = ROOT / "7_run_cse_qiwen_xyz_control_same_lobe_pair_qiwenevents_batch.ipynb"


def markdown(cell_id, source):
    return {
        "cell_type": "markdown", "id": cell_id, "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(cell_id, source):
    return {
        "cell_type": "code", "id": cell_id, "metadata": {},
        "execution_count": None, "outputs": [],
        "source": source.splitlines(keepends=True),
    }


with SOURCE.open(encoding="utf-8") as stream:
    source_notebook = json.load(stream)
with TARGET.open(encoding="utf-8") as stream:
    notebook = json.load(stream)

if any(cell.get("id") == "same-lobe-grid-run" for cell in notebook["cells"]):
    raise RuntimeError("The Figure 3(a) grid is already present; refusing to append twice.")

intro = "".join(notebook["cells"][3]["source"])
old_intro = "It does **not** run the full\n5×9 T–D grid. The previously saved `sign_invariant` rate is the comparator."
new_intro = (
    "The 40-event pilot is followed by a separate 5×9 T–D grid. The grid is "
    "not executed automatically and saves each event separately; the "
    "[4T0], D=0.05 point reuses the pilot files. The previously saved "
    "`sign_invariant` rate is the comparator."
)
if old_intro not in intro:
    raise RuntimeError("The expected notebook introduction was not found.")
notebook["cells"][3]["source"] = intro.replace(old_intro, new_intro).splitlines(keepends=True)

grid_run = r'''# Full Figure 3(a) grid: five warning horizons, nine D values, 40 events.
# This is expensive (up to 1,800 CSE runs and large diagnostic pickles).
# Results are isolated in the same_lobe_pair experiment directory. An
# interrupted run resumes from saved events without rerunning them.
import os
import pickle
import tempfile
import time

FIG3A_T_MULTIPLES = {
    '[1.5T0]': 1.5,
    '[2T0]': 2.0,
    '[2.5T0]': 2.5,
    '[3T0]': 3.0,
    '[4T0]': 4.0,
}
FIG3A_D_VALUES = [0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
FIG3A_N_EVENTS = 40
FIG3A_REGIME_REFERENCE = 'same_lobe_pair'
FIG3A_CANDIDATE_POOL = 'full'
assert REGIME_REFERENCE_MODE == FIG3A_REGIME_REFERENCE
assert CONTROL_INDICES is None
assert len(t_start_series) >= FIG3A_N_EVENTS


def fig3a_event_path(T_label, D_run, event_index):
    # Reuse the independently saved pilot at this exact grid point.
    if T_label == '[4T0]' and D_run == 0.05:
        return RESULTS_DIR / (
            f'cse_qiwen_xyz_control_same_lobe_pair_event_{event_index:02d}'
            '_D0p05_4T0_qiwenevents.pkl'
        )
    t_tag = T_label.strip('[]').replace('.', 'p')
    d_tag = f'{D_run:g}'.replace('.', 'p')
    return RESULTS_DIR / (
        f'cse_qiwen_xyz_control_same_lobe_pair_fig3a_{t_tag}'
        f'_D{d_tag}_event_{event_index:02d}{OUTPUT_TAG}.pkl'
    )


def check_fig3a_event(row, T_label, D_run, event_index):
    multiple_run = FIG3A_T_MULTIPLES[T_label]
    position_run = int(np.ceil(T_0 * multiple_run))
    expected = {
        'event_index': event_index,
        'D': D_run,
        'regime_reference': FIG3A_REGIME_REFERENCE,
        'candidate_pool': FIG3A_CANDIDATE_POOL,
        'control_indices': None,
        't_start': t_start_series[event_index],
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(
                f'{fig3a_event_path(T_label, D_run, event_index)}: '
                f'{key}={row.get(key)!r}, expected {value!r}'
            )
    summary = row.get('cse_summary', {})
    for key, value in {
        'position': position_run,
        'seed': 1000 + event_index,
        'requested_num_cycles': experiment_step,
    }.items():
        if summary.get(key) != value:
            raise ValueError(
                f'{fig3a_event_path(T_label, D_run, event_index)}: '
                f'summary {key}={summary.get(key)!r}, expected {value!r}'
            )


for T_label, multiple_run in FIG3A_T_MULTIPLES.items():
    position_run = int(np.ceil(T_0 * multiple_run))
    print(f'=== T={T_label}, position={position_run} ===')
    for D_run in FIG3A_D_VALUES:
        for event_index in range(FIG3A_N_EVENTS):
            result_path = fig3a_event_path(T_label, D_run, event_index)
            if result_path.is_file():
                with result_path.open('rb') as stream:
                    row = pickle.load(stream)
                status = 'reused'
            else:
                started = time.time()
                row = run_single_cse_core(
                    event_index=event_index,
                    D_run=D_run,
                    seed=1000 + event_index,
                    das=das,
                    x_nature=x_nature,
                    t_start_series=t_start_series,
                    osse_Xa_history=osse_Xa_history,
                    osse_summary=osse_summary,
                    experiment_step=experiment_step,
                    position=position_run,
                    Ta=Ta,
                    fcst_dt=fcst_dt,
                    path_figures=PathFigures,
                    total_events=FIG3A_N_EVENTS,
                    regime_reference=FIG3A_REGIME_REFERENCE,
                    candidate_pool=FIG3A_CANDIDATE_POOL,
                    ambiguous_regime_policy=AMBIGUOUS_REGIME_POLICY,
                    control_indices=CONTROL_INDICES,
                    T_0=T_0,
                    multiple_number=multiple_run,
                )
                check_fig3a_event(row, T_label, D_run, event_index)
                with tempfile.NamedTemporaryFile(
                    mode='wb', prefix=result_path.stem + '_', suffix='.tmp',
                    dir=RESULTS_DIR, delete=False,
                ) as stream:
                    pickle.dump(row, stream, protocol=pickle.HIGHEST_PROTOCOL)
                    temporary_path = Path(stream.name)
                os.replace(temporary_path, result_path)
                status = f'computed in {(time.time() - started) / 60:.1f} min'
            check_fig3a_event(row, T_label, D_run, event_index)
            print(
                f'T={T_label}, D={D_run:g}, '
                f'event={event_index + 1}/{FIG3A_N_EVENTS}, '
                f'success={row["success"]} ({status})'
            )
            del row
'''

grid_summary_prefix = r'''# Post-process saved same-lobe Figure 3(a) event files only.
# Missing and partial grid points remain blank in the success matrix; the
# aggregate table still shows each partial point's event count and rate.
# This cell does not run CSE or touch the sign_invariant outputs.
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIG3A_T_MULTIPLES = globals().get('FIG3A_T_MULTIPLES', {
    '[1.5T0]': 1.5, '[2T0]': 2.0, '[2.5T0]': 2.5,
    '[3T0]': 3.0, '[4T0]': 4.0,
})
FIG3A_D_VALUES = globals().get(
    'FIG3A_D_VALUES', [0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
)
FIG3A_N_EVENTS = globals().get('FIG3A_N_EVENTS', 40)

# Historical values transcribed in the existing sign_invariant notebook.
QIWEN_FIG3A_CONTROL_XYZ_VALUES = {
    '[1.5T0]': {0.02: 0.0, 0.03: 0.0, 0.04: 0.0, 0.05: 0.0, 0.1: 0.0, 0.2: 0.0, 0.3: 0.0, 0.4: 0.0, 0.5: 0.025},
    '[2T0]': {0.02: 0.0, 0.03: 0.0, 0.04: 0.0, 0.05: 0.0, 0.1: 0.0, 0.2: 0.025, 0.3: 0.0, 0.4: 0.0, 0.5: 0.0},
    '[2.5T0]': {0.02: 0.0, 0.03: 0.0, 0.04: 0.0, 0.05: 0.025, 0.1: 0.250, 0.2: 0.275, 0.3: 0.250, 0.4: 0.200, 0.5: 0.175},
    '[3T0]': {0.02: 0.0, 0.03: 0.050, 0.04: 0.425, 0.05: 0.550, 0.1: 0.800, 0.2: 0.675, 0.3: 0.400, 0.4: 0.100, 0.5: 0.200},
    '[4T0]': {0.02: 0.050, 0.03: 0.975, 0.04: 0.925, 0.05: 0.975, 0.1: 0.825, 0.2: 0.825, 0.3: 0.725, 0.4: 0.500, 0.5: 0.525},
}


def fig3a_event_path(T_label, D_run, event_index):
    if T_label == '[4T0]' and D_run == 0.05:
        return RESULTS_DIR / (
            f'cse_qiwen_xyz_control_same_lobe_pair_event_{event_index:02d}'
            '_D0p05_4T0_qiwenevents.pkl'
        )
    t_tag = T_label.strip('[]').replace('.', 'p')
    d_tag = f'{D_run:g}'.replace('.', 'p')
    return RESULTS_DIR / (
        f'cse_qiwen_xyz_control_same_lobe_pair_fig3a_{t_tag}'
        f'_D{d_tag}_event_{event_index:02d}{OUTPUT_TAG}.pkl'
    )


event_rows = []
aggregate_rows = []
for T_label, multiple_run in FIG3A_T_MULTIPLES.items():
    for D_run in FIG3A_D_VALUES:
        rows = []
        for event_index in range(FIG3A_N_EVENTS):
            result_path = fig3a_event_path(T_label, D_run, event_index)
            if not result_path.is_file():
                continue
            with result_path.open('rb') as stream:
                row = pickle.load(stream)
            if (row.get('event_index') != event_index
                    or row.get('D') != D_run
                    or row.get('regime_reference') != 'same_lobe_pair'
                    or row.get('cse_summary', {}).get('position')
                    != int(np.ceil(T_0 * multiple_run))):
                raise ValueError(f'Unexpected configuration in {result_path}')
            event_rows.append({
                'T': T_label, 'multiple_number': multiple_run,
                'position': int(np.ceil(T_0 * multiple_run)),
                'D': D_run, 'event': event_index,
                'success': bool(row['success']),
                'control_ratio': row['control_ratio'],
                'false_alarm_ratio': row['false_alarm_ratio'],
                'success_control_ratio': row['success_control_ratio'],
                'failure_cycle': row['failure_cycle'],
                'result_file': str(result_path),
            })
            rows.append({
                'success': bool(row['success']),
                'control_ratio': row['control_ratio'],
                'false_alarm_ratio': row['false_alarm_ratio'],
                'success_control_ratio': row['success_control_ratio'],
            })
            del row
        n_events = len(rows)
        aggregate_rows.append({
            'T': T_label, 'multiple_number': multiple_run,
            'position': int(np.ceil(T_0 * multiple_run)), 'D': D_run,
            'n_events': n_events,
            'success_ratio': float(np.mean([r['success'] for r in rows])) if rows else np.nan,
            'mean_control_ratio': float(np.nanmean([r['control_ratio'] for r in rows])) if rows else np.nan,
            'mean_false_alarm_ratio': float(np.nanmean([r['false_alarm_ratio'] for r in rows])) if rows else np.nan,
            'mean_success_control_ratio': float(np.nanmean([r['success_control_ratio'] for r in rows])) if rows else np.nan,
            'n_failures': sum(not r['success'] for r in rows) if rows else np.nan,
        })
        del rows

fig3g_event_table = pd.DataFrame(event_rows)
fig3g_aggregate_table = pd.DataFrame(aggregate_rows)
fig3g_aggregate_table['complete_success_ratio'] = fig3g_aggregate_table[
    'success_ratio'
].where(fig3g_aggregate_table['n_events'] == FIG3A_N_EVENTS)
fig3g_success_matrix = fig3g_aggregate_table.pivot(
    index='T', columns='D', values='complete_success_ratio'
).reindex(index=list(FIG3A_T_MULTIPLES), columns=FIG3A_D_VALUES)
qiwen_fig3g_matrix = pd.DataFrame.from_dict(
    QIWEN_FIG3A_CONTROL_XYZ_VALUES, orient='index'
).reindex(index=list(FIG3A_T_MULTIPLES), columns=FIG3A_D_VALUES)
fig3g_difference_matrix = fig3g_success_matrix - qiwen_fig3g_matrix

fig3g_event_table_file = RESULTS_DIR / 'cse_qiwen_xyz_control_40_fig3a_grid_events_same_lobe_pair_qiwenevents.csv'
fig3g_aggregate_table_file = RESULTS_DIR / 'cse_qiwen_xyz_control_40_fig3a_grid_aggregates_same_lobe_pair_qiwenevents.csv'
fig3g_success_matrix_file = RESULTS_DIR / 'cse_qiwen_xyz_control_40_fig3a_success_matrix_same_lobe_pair_qiwenevents.csv'
fig3g_difference_matrix_file = RESULTS_DIR / 'cse_qiwen_xyz_control_40_fig3a_minus_qiwen_matrix_same_lobe_pair_qiwenevents.csv'
fig3g_event_table.to_csv(fig3g_event_table_file, index=False)
fig3g_aggregate_table.to_csv(fig3g_aggregate_table_file, index=False)
fig3g_success_matrix.to_csv(fig3g_success_matrix_file)
fig3g_difference_matrix.to_csv(fig3g_difference_matrix_file)

display(fig3g_aggregate_table[['T', 'D', 'n_events', 'success_ratio']])
display(fig3g_success_matrix)
display(fig3g_difference_matrix)
print('Only cells with n_events=40 are complete 40-event success rates.')

PathFigures = EXPERIMENT_FIGURES_DIR
PathFigures.mkdir(parents=True, exist_ok=True)
'''

source_plot = "".join(source_notebook["cells"][22]["source"])
plot_start = source_plot.index("PAPER_TABLE_COLORS = {")
plot_tail = source_plot[plot_start:]
plot_tail = plot_tail.replace(
    "'(a) Diego control xyz'", "'(a) Diego same-lobe pair xyz'"
).replace(
    "cse_qiwen_xyz_control_fig3a_grid_comparison_sign_invariant_qiwenevents.png",
    "cse_qiwen_xyz_control_fig3a_grid_comparison_same_lobe_pair_qiwenevents.png",
)

notebook["cells"].extend([
    markdown("same-lobe-grid-heading", "## Batch Run: Figure 3(a) T-D Grid\n"),
    code("same-lobe-grid-run", grid_run),
    markdown("same-lobe-grid-summary-heading", """## Figure 3(a) Grid Summary and Comparison

The Qiwen table is a historical benchmark. The same-lobe-pair results are
separate from the previous `sign_invariant` results. Cells with fewer than
40 saved events are preliminary and show their event count.
"""),
    code("same-lobe-grid-summary", grid_summary_prefix + plot_tail),
])

with TARGET.open("w", encoding="utf-8") as stream:
    json.dump(notebook, stream, ensure_ascii=False, indent=1)
    stream.write("\n")
print(TARGET)
