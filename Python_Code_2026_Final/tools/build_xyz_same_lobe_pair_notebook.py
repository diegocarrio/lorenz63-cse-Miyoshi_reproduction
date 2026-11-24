"""Build an isolated 40-event xyz-control pilot for same-lobe pairing."""

import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "7_run_cse_qiwen_xyz_control_sign_invariant_qiwenevents_batch.ipynb"
TARGET = ROOT / "7_run_cse_qiwen_xyz_control_same_lobe_pair_qiwenevents_batch.ipynb"


def set_source(cell, value):
    cell["source"] = value.splitlines(keepends=True)


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


if TARGET.exists():
    raise FileExistsError(f"Refusing to overwrite {TARGET}")

with SOURCE.open(encoding="utf-8") as stream:
    source_notebook = json.load(stream)

cells = deepcopy(source_notebook["cells"][:16])
if len(cells) != 16:
    raise ValueError("The source notebook no longer has the expected setup cells.")
for index, cell in enumerate(cells):
    cell["id"] = f"same-lobe-setup-{index:02d}"
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

set_source(cells[0], """# Same-lobe-pair CSE pilot

This notebook is a separate version of the xyz-observation/xyz-control Qiwen
batch. It writes only to `xyz_control_same_lobe_pair_qiwenevents`; the existing
`sign_invariant` notebook and results remain untouched. Run the cells in order.
""")
set_source(cells[2], "# Qiwen CSE xyz control — Same-lobe safe–crossing pairing\n")
set_source(cells[3], """The only scientific change from the existing `sign_invariant` CSE is in
`past_safe_control`. When all current ensemble members forecast a crossing,
choose one crossing member first. A past candidate is safe only if its aligned
forecast stays strictly in **that member's starting lobe**. The sign comes
from the analysis-member forecast, never from the controlled nature run.
The old `sign_invariant` mode is preserved unchanged in the shared CSE code.

This pilot uses the same xyz OSSE, Qiwen event catalogue, 40 event indices,
seeds `1000 + event_index`, `D=0.05`, and warning horizon `T=4T0` as the
previous grid point. It saves each event separately, so an interrupted run
can resume without rewriting completed events. It does **not** run the full
5×9 T–D grid. The previously saved `sign_invariant` rate is the comparator.
""")

old_name = "EXPERIMENT_NAME = 'xyz_control_sign_invariant_qiwenevents'"
new_name = "EXPERIMENT_NAME = 'xyz_control_same_lobe_pair_qiwenevents'"
cell_1 = "".join(cells[1]["source"])
if old_name not in cell_1:
    raise ValueError("The source experiment path was not found.")
set_source(cells[1], cell_1.replace(old_name, new_name))

set_source(cells[5], """import importlib
import pickle
from pathlib import Path

import numpy as np
from class_da_sys import da_system
import class_cse_qiwen as cse_qiwen

cse_qiwen = importlib.reload(cse_qiwen)
detect_regime_changes = cse_qiwen.detect_regime_changes
lorenz63 = cse_qiwen.lorenz63
run_single_cse_core = cse_qiwen.run_single_cse
""")

cell_7 = "".join(cells[7]["source"])
if "REGIME_REFERENCE_MODE = 'sign_invariant'" not in cell_7:
    raise ValueError("The source control setup was not found.")
set_source(cells[7], cell_7.replace(
    "REGIME_REFERENCE_MODE = 'sign_invariant'",
    "REGIME_REFERENCE_MODE = 'same_lobe_pair'",
))

cell_15 = "".join(cells[15]["source"])
if "regime_reference='sign_invariant'" not in cell_15:
    raise ValueError("The source single-event wrapper was not found.")
set_source(cells[15], cell_15.replace(
    "regime_reference='sign_invariant'",
    "regime_reference='same_lobe_pair'",
))

cells.extend([
    markdown("same-lobe-run-heading", """## Run the 40-event pilot

Only this new result directory is written. Each event is saved atomically in
its own pickle, with the mode and configuration checked before resuming.
"""),
    code("same-lobe-run", """import os
import tempfile
import time

PILOT_D = 0.05
PILOT_D_TAG = '0p05'
N_PILOT_EVENTS = 40
assert REGIME_REFERENCE_MODE == 'same_lobe_pair'
assert CONTROL_INDICES is None
assert multiple_number == 4
assert len(t_start_series) >= N_PILOT_EVENTS


def pilot_result_path(event_index):
    return RESULTS_DIR / (
        f'cse_qiwen_xyz_control_same_lobe_pair_event_{event_index:02d}'
        f'_D{PILOT_D_TAG}_4T0_qiwenevents.pkl'
    )


def check_pilot_result(row, event_index):
    expected = {
        'event_index': event_index,
        'D': PILOT_D,
        'regime_reference': REGIME_REFERENCE_MODE,
        'candidate_pool': 'full',
        'control_indices': None,
        't_start': t_start_series[event_index],
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(
                f'Existing event {event_index} has {key}={row.get(key)!r}, '
                f'expected {value!r}; refusing to reuse it.'
            )
    summary = row.get('cse_summary', {})
    for key, value in {
        'position': position,
        'seed': 1000 + event_index,
        'requested_num_cycles': experiment_step,
    }.items():
        if summary.get(key) != value:
            raise ValueError(
                f'Existing event {event_index} has summary {key}='
                f'{summary.get(key)!r}, expected {value!r}; '
                'refusing to reuse it.'
            )


pilot_rows = []
for event_index in range(N_PILOT_EVENTS):
    result_path = pilot_result_path(event_index)
    if result_path.exists():
        with result_path.open('rb') as stream:
            row = pickle.load(stream)
        status = 'reused'
    else:
        started = time.time()
        row = run_single_cse(
            event_index, PILOT_D, seed=1000 + event_index,
            regime_reference=REGIME_REFERENCE_MODE,
            candidate_pool='full',
        )
        check_pilot_result(row, event_index)
        with tempfile.NamedTemporaryFile(
            mode='wb', prefix=result_path.stem + '_', suffix='.tmp',
            dir=RESULTS_DIR, delete=False,
        ) as stream:
            pickle.dump(row, stream, protocol=pickle.HIGHEST_PROTOCOL)
            temporary_path = Path(stream.name)
        os.replace(temporary_path, result_path)
        status = f'computed in {(time.time() - started) / 60:.1f} min'
    check_pilot_result(row, event_index)
    pilot_rows.append({
        'event_index': event_index,
        'success': bool(row['success']),
        'failure_cycle': row['failure_cycle'],
        'failure_reason': row.get('failure_reason'),
        'control_ratio': float(row['control_ratio']),
        'n_past_safe': int(row['branch_counts']['past_safe_control']),
    })
    print(
        f'[{event_index + 1}/{N_PILOT_EVENTS}] '
        f'success={row["success"]} '
        f'past_safe={row["branch_counts"]["past_safe_control"]} '
        f'({status})'
    )
    del row
"""),
    markdown("same-lobe-summary-heading", """## Pilot success rate and saved-reference comparison

This cell only reads saved event files and the earlier aggregate CSV. It does
not run CSE. The old `nature` setting is not used as a publication baseline.
"""),
    code("same-lobe-summary", """import csv

event_summaries = []
for event_index in range(N_PILOT_EVENTS):
    result_path = pilot_result_path(event_index)
    if not result_path.is_file():
        print(f'Missing paired event: {event_index}')
        continue
    with result_path.open('rb') as stream:
        row = pickle.load(stream)
    check_pilot_result(row, event_index)
    event_summaries.append({
        'event': event_index,
        'success': bool(row['success']),
        'failure_cycle': row['failure_cycle'],
        'failure_reason': row.get('failure_reason'),
        'past_safe_cycles': int(row['branch_counts']['past_safe_control']),
        'control_ratio': float(row['control_ratio']),
    })
    del row

n_done = len(event_summaries)
n_success = sum(item['success'] for item in event_summaries)
print(f'Same-lobe pair: {n_success}/{n_done} successful '
      f'({100 * n_success / n_done:.1f}%)' if n_done else 'No paired events saved.')
print('Event  Success  Failure cycle  Reason')
for item in event_summaries:
    print(f\"{item['event']:>5}  {str(item['success']):>7}  \"
          f\"{str(item['failure_cycle']):>13}  {item['failure_reason']}\")

baseline_csv = (
    PROJECT_ROOT / 'outputs' / '05_cse_qiwen'
    / 'xyz_control_sign_invariant_qiwenevents' / 'results'
    / 'cse_qiwen_xyz_control_40_fig3a_grid_aggregates_sign_invariant_qiwenevents.csv'
)
if baseline_csv.is_file():
    with baseline_csv.open(newline='') as stream:
        baseline_rows = list(csv.DictReader(stream))
    matching = [
        item for item in baseline_rows
        if item['T'] == '[4T0]' and float(item['D']) == PILOT_D
    ]
    if len(matching) == 1:
        baseline = matching[0]
        print('Saved sign_invariant baseline: '
              f\"{baseline['success_ratio']} success ratio \"
              f\"over {baseline['n_events']} events\")
else:
    print('Sign-invariant aggregate CSV not found; paired results remain valid.')
"""),
])

notebook = {
    "cells": cells,
    "metadata": source_notebook["metadata"],
    "nbformat": 4,
    "nbformat_minor": 5,
}
with TARGET.open("x", encoding="utf-8") as stream:
    json.dump(notebook, stream, ensure_ascii=False, indent=1)
    stream.write("\n")
print(TARGET)
