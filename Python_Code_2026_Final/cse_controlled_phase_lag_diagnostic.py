"""Retrospective lag diagnostic on a saved controlled Lorenz-63 trajectory.

This diagnostic reads the already applied interventions and the corresponding
controlled-nature trajectory. Positive lags refer to states after the push,
including the effects of that push and any later pushes. It is descriptive,
not a no-push forecast or a causal estimate of one intervention's effect.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cse_intervention_state_plot import extract_pre_intervention_records


def summarize_controlled_phase_lags(result, lags=range(-20, 21), sigma=10.0):
    """Return sign agreement of each saved u_z[i] with q[i + lag].

    q = x * sigma * (y - x) is evaluated on the saved controlled trajectory.
    The same interventions are retained at every lag so percentages are
    directly comparable across the requested lag window. Exact zeros in q
    are reported and excluded from the agreement denominator.
    """
    if not result.get("success", False):
        raise ValueError("Select a saved successful CSE result.")
    lag_values = list(lags)
    if not lag_values or 0 not in lag_values:
        raise ValueError("lags must be nonempty and include zero.")
    if any(isinstance(value, bool) or int(value) != value for value in lag_values):
        raise ValueError("All lags must be integer model steps.")
    lag_values = sorted(set(int(value) for value in lag_values))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite.")

    summary = result.get("cse_summary", {})
    Ta = int(summary["Ta"])
    dt = float(summary["fcst_dt"])
    n_steps = int(result["n_completed"]) * Ta
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("The saved model step is invalid.")
    trajectory = np.asarray(result["evo_history"], dtype=float)
    if trajectory.shape[0] != 3 or trajectory.shape[1] < n_steps + 1:
        raise ValueError("Saved controlled trajectory has an unexpected shape.")
    trajectory = trajectory[:, :n_steps + 1]
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("Saved controlled trajectory contains non-finite states.")

    records = extract_pre_intervention_records(result, Ta)
    if not records:
        raise ValueError("No applied z-interventions were found.")
    indices = np.array([row["relative_step"] for row in records], dtype=int)
    pushes = np.array([row["u_z"] for row in records], dtype=float)
    saved_indices = np.asarray(result["perturbed_time"], dtype=int)
    if not np.array_equal(indices, saved_indices):
        raise ValueError("Intervention indices disagree with perturbed_time.")
    if not np.allclose(
        trajectory[:2, indices],
        np.array([[row["x_pre"], row["y_pre"]] for row in records]).T[0:2],
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("Pre-push x,y do not align with the controlled trajectory.")

    q = trajectory[0] * float(sigma) * (trajectory[1] - trajectory[0])
    baseline_sign = np.sign(q[indices])
    all_baseline_defined = baseline_sign != 0
    all_baseline_agree = int(np.count_nonzero(
        (np.sign(pushes) == baseline_sign) & all_baseline_defined
    ))

    common = (indices + lag_values[0] >= 0) & (
        indices + lag_values[-1] <= n_steps
    )
    if not np.any(common):
        raise ValueError("No intervention has the full requested lag window.")
    indices = indices[common]
    pushes = pushes[common]
    push_signs = np.sign(pushes)

    rows = []
    for lag in lag_values:
        phase_signs = np.sign(q[indices + lag])
        defined = phase_signs != 0
        n_defined = int(np.count_nonzero(defined))
        n_agree = int(np.count_nonzero((phase_signs == push_signs) & defined))
        rows.append({
            "lag": lag,
            "lag_time": lag * dt,
            "n_pushes": int(indices.size),
            "n_defined": n_defined,
            "n_zero_q": int(indices.size - n_defined),
            "n_agree": n_agree,
            "agreement_percent": (
                100.0 * n_agree / n_defined if n_defined else float("nan")
            ),
        })

    metadata = {
        "dt": dt,
        "n_all_pushes": len(records),
        "n_common_pushes": int(indices.size),
        "n_excluded_boundaries": int(len(records) - indices.size),
        "lag0_all_defined": int(np.count_nonzero(all_baseline_defined)),
        "lag0_all_agree": all_baseline_agree,
        "lag0_all_percent": (
            100.0 * all_baseline_agree / np.count_nonzero(all_baseline_defined)
        ),
    }
    return rows, metadata


def plot_controlled_phase_lags(rows, metadata, figure_path):
    """Save the requested one-panel lag curve and mark the maximum."""
    if any(not np.isfinite(row["agreement_percent"]) for row in rows):
        raise ValueError("At least one lag has no defined q signs.")
    lag_zero = next(row for row in rows if row["lag"] == 0)
    best = max(rows, key=lambda row: row["agreement_percent"])
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    ax.plot(
        [row["lag"] for row in rows],
        [row["agreement_percent"] for row in rows],
        color="#2457A7", marker="o", markersize=3.5, linewidth=1.8,
    )
    ax.axvline(0, color="0.35", linestyle="--", linewidth=1.3)
    ax.scatter(
        [best["lag"]], [best["agreement_percent"]],
        color="#D2612D", s=75, zorder=5, label="Maximum agreement",
    )
    ax.set_xlabel("Lag L (model steps; positive = after intervention)")
    ax.set_ylabel("Sign agreement (%)")
    ax.set_title("Applied $u_z(i)$ versus controlled-trajectory phase $q(i+L)$")
    ax.set_xlim(rows[0]["lag"], rows[-1]["lag"])
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    annotation = (
        f"Lag 0: {lag_zero['agreement_percent']:.2f}%\n"
        f"Best lag: {best['lag']:+d} steps "
        f"({best['lag_time']:+.3f} time units)\n"
        f"Maximum: {best['agreement_percent']:.2f}%"
    )
    ax.text(
        0.02, 0.98, annotation, transform=ax.transAxes,
        va="top", ha="left", fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.95},
    )
    fig.text(
        0.5, 0.01,
        f"Same {metadata['n_common_pushes']:,} interventions at every lag; "
        "future states include subsequent control.",
        ha="center", va="bottom", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(figure_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return figure_path
