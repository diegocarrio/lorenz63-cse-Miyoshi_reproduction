"""Retrospective phase-lead diagnostic for saved z-only CSE interventions.

The true pre-push state is used only for scientific diagnosis. Nothing in this
module changes the controller, its observations, or its intervention history.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from class_Lorenz63 import run_RK4
from class_cse_qiwen import lorenz63


REQUIRED_COLUMNS = {
    "branch", "u_z", "x_pre", "y_pre", "z_pre",
    "x_post", "y_post", "z_post",
}


def _read_interventions(path):
    """Read the already saved pre-intervention table without changing it."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing intervention table: {path}")

    records = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} lacks required columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            try:
                state = np.array(
                    [float(row[name]) for name in ("x_pre", "y_pre", "z_pre")],
                    dtype=float,
                )
                post_state = np.array(
                    [float(row[name]) for name in ("x_post", "y_post", "z_post")],
                    dtype=float,
                )
                u_z = float(row["u_z"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value in {path}:{row_number}") from exc
            if not np.all(np.isfinite(state)) or not np.all(np.isfinite(post_state)):
                raise ValueError(f"Non-finite state in {path}:{row_number}")
            if not np.isfinite(u_z) or u_z == 0.0:
                raise ValueError(f"Expected a nonzero z-push in {path}:{row_number}")
            if not np.allclose(
                post_state - state, [0.0, 0.0, u_z], rtol=0.0, atol=1e-10
            ):
                raise ValueError(f"Pre/post-push states do not align in {path}:{row_number}")
            if not row["branch"]:
                raise ValueError(f"Missing control branch in {path}:{row_number}")
            records.append((state, u_z, row["branch"]))

    if not records:
        raise ValueError(f"No applied interventions in {path}")
    return records


def _one_free_step(state, dt):
    """Use exactly the Lorenz-63 RK4 step used by the CSE, with no push."""
    return run_RK4(
        lorenz63, 10.0, 8.0 / 3.0, 28.0, state, dt, dt
    ).integrate()[1, :].copy()


def summarize_phase_lead(event_tables, dt, max_lead_steps=8, phase_deadband=0.0):
    """Compare each saved push with the unforced future sign of x times dx/dt.

    Each short forecast starts from the controlled truth immediately before a
    recorded intervention, omits that intervention and all later ones, and
    lasts at most max_lead_steps. It does not rerun the DA or CSE.

    event_tables maps a readable event label to its existing intervention CSV.
    Returns one summary row per event, branch and lead (including branch='all').
    Undefined phases, abs(x * dx/dt) <= phase_deadband, are reported separately
    and excluded from the agreement denominator.
    """
    if not event_tables:
        raise ValueError("Provide at least one saved intervention table.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite.")
    if isinstance(max_lead_steps, bool) or int(max_lead_steps) != max_lead_steps:
        raise ValueError("max_lead_steps must be a nonnegative integer.")
    max_lead_steps = int(max_lead_steps)
    if max_lead_steps < 0:
        raise ValueError("max_lead_steps must be nonnegative.")
    if not np.isfinite(phase_deadband) or phase_deadband < 0:
        raise ValueError("phase_deadband must be nonnegative and finite.")

    counts = defaultdict(lambda: [0, 0, 0])  # pushes, defined phases, matches
    event_order = list(event_tables)
    branch_order = ["all"]

    for label, path in event_tables.items():
        for initial_state, u_z, branch in _read_interventions(path):
            if branch not in branch_order:
                branch_order.append(branch)
            state = initial_state.copy()
            push_sign = np.sign(u_z)
            for lead in range(max_lead_steps + 1):
                q = state[0] * lorenz63(state, 0.0)[0]
                if not np.isfinite(q):
                    raise ValueError(f"Non-finite phase in event {label}, lead {lead}.")
                defined = abs(q) > phase_deadband
                matched = defined and np.sign(q) == push_sign
                for group in ("all", branch):
                    tally = counts[(label, group, lead)]
                    tally[0] += 1
                    tally[1] += int(defined)
                    tally[2] += int(matched)
                if lead < max_lead_steps:
                    state = _one_free_step(state, dt)

    summary = []
    for label in event_order:
        for branch in branch_order:
            if (label, branch, 0) not in counts:
                continue
            for lead in range(max_lead_steps + 1):
                pushes, defined, matches = counts[(label, branch, lead)]
                summary.append({
                    "event": label,
                    "branch": branch,
                    "lead_steps": lead,
                    "lead_time": lead * float(dt),
                    "n_pushes": pushes,
                    "n_defined": defined,
                    "n_undefined": pushes - defined,
                    "n_agree": matches,
                    "agreement_percent": (
                        100.0 * matches / defined if defined else float("nan")
                    ),
                })
    return summary


def plot_phase_lead(summary, figure_path):
    """Compare the overall result and both control branches by forecast lead."""
    overall = [row for row in summary if row["branch"] == "all"]
    if not overall:
        raise ValueError("Summary contains no overall phase-lead rows.")
    events = list(dict.fromkeys(row["event"] for row in overall))
    groups = [
        ("all", "All pushes"),
        ("current_safe_control", "Current-safe control"),
        ("past_safe_control", "Past-safe control"),
    ]
    available = [(group, title) for group, title in groups
                 if any(row["branch"] == group for row in summary)]
    plotted = [row["agreement_percent"] for row in summary
               if row["branch"] in {group for group, _ in available}]
    if any(not np.isfinite(value) for value in plotted):
        raise ValueError("Some event/branch/lead has no defined phase.")
    lower = max(0, 5 * np.floor((min(plotted) - 4) / 5))
    upper = min(100, 5 * np.ceil((max(plotted) + 4) / 5))
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(
        1, len(available), figsize=(5.2 * len(available), 5.0),
        sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes)
    for ax, (group, title) in zip(axes, available):
        for event_index, label in enumerate(events):
            rows = [row for row in summary
                    if row["event"] == label and row["branch"] == group]
            if not rows:
                continue
            ax.plot(
                [row["lead_steps"] for row in rows],
                [row["agreement_percent"] for row in rows],
                marker="o", linewidth=2, color=colors[event_index], label=label,
            )
        ax.set_title(title)
        ax.set_xticks([row["lead_steps"] for row in overall
                       if row["event"] == events[0]])
        ax.set_ylim(lower, upper)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Sign agreement with $u_z$ (%)")
    axes[-1].legend(title="Saved CSE event", frameon=False, loc="lower left")
    fig.suptitle("Push sign versus unforced future sign of x × dx/dt")
    fig.supxlabel("Lead after the pre-push state (model steps)", y=0.045)
    fig.text(
        0.5, 0.005,
        "Short free forecasts from true pre-push states; retrospective diagnostic only.",
        ha="center", va="bottom", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return figure_path
