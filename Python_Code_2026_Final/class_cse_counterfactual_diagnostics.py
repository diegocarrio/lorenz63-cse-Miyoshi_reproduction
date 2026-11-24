"""Ensemble-only counterfactual diagnostics for the z-only CSE experiment.

The routines in this module do not alter the CSE decision rule.  They replay
the intervention packet selected by an already completed CSE run on copies of
the analysis ensemble and compare it with zero and opposite interventions.
"""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np


BRANCH_NAMES = ("free", "cse", "opposite")
BRANCH_MULTIPLIERS = np.array([0.0, 1.0, -1.0], dtype=float)
BRANCH_LABELS = {
    "free": "No intervention",
    "cse": "CSE-selected $u_z$",
    "opposite": "Opposite $-u_z$",
}
BRANCH_COLORS = {
    "free": "0.35",
    "cse": "tab:blue",
    "opposite": "tab:orange",
}


def _lorenz63_vectorized(state, sigma=10.0, beta=8.0 / 3.0, rho=28.0):
    """Evaluate Lorenz-63 for arrays whose final axis is (x, y, z)."""
    state = np.asarray(state, dtype=float)
    x = state[..., 0]
    y = state[..., 1]
    z = state[..., 2]
    return np.stack(
        (
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ),
        axis=-1,
    )


def _rk4_step_vectorized(state, dt, sigma=10.0, beta=8.0 / 3.0, rho=28.0):
    """Advance one classical RK4 step for one or more Lorenz-63 states."""
    k1 = dt * _lorenz63_vectorized(state, sigma=sigma, beta=beta, rho=rho)
    k2 = dt * _lorenz63_vectorized(
        state + 0.5 * k1, sigma=sigma, beta=beta, rho=rho
    )
    k3 = dt * _lorenz63_vectorized(
        state + 0.5 * k2, sigma=sigma, beta=beta, rho=rho
    )
    k4 = dt * _lorenz63_vectorized(state + k3, sigma=sigma, beta=beta, rho=rho)
    return state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def simulate_counterfactual_ensemble(
    initial_ensemble,
    perturbations,
    position,
    dt,
    sigma=10.0,
    beta=8.0 / 3.0,
    rho=28.0,
):
    """Forecast free, selected-CSE, and opposite branches from one ensemble.

    Parameters
    ----------
    initial_ensemble : array-like, shape (3, ensemble_size)
        Analysis ensemble at the beginning of a CSE cycle.
    perturbations : array-like, shape (Ta, 3)
        Stored intervention packet. Row zero must be zero; rows 1..Ta-1 are
        applied after the corresponding free RK4 step, matching CSE exactly.
    position : int
        Forecast horizon in model steps.
    dt : float
        Model time step.

    Returns
    -------
    ndarray, shape (3, ensemble_size, 3, position + 1)
        Branch order is ``free``, ``cse``, ``opposite``.
    """
    initial_ensemble = np.asarray(initial_ensemble, dtype=float)
    perturbations = np.asarray(perturbations, dtype=float)

    if initial_ensemble.ndim != 2 or initial_ensemble.shape[0] != 3:
        raise ValueError("initial_ensemble must have shape (3, ensemble_size).")
    if perturbations.ndim != 2 or perturbations.shape[1] != 3:
        raise ValueError("perturbations must have shape (Ta, 3).")
    if not np.allclose(perturbations[0], 0.0):
        raise ValueError("The first intervention row must be zero.")
    if int(position) < 1:
        raise ValueError("position must be at least one model step.")

    ensemble_size = initial_ensemble.shape[1]
    states = np.repeat(initial_ensemble.T[None, :, :], len(BRANCH_NAMES), axis=0)
    trajectories = np.empty(
        (len(BRANCH_NAMES), ensemble_size, 3, int(position) + 1),
        dtype=float,
    )
    trajectories[..., 0] = states

    for step in range(1, int(position) + 1):
        states = _rk4_step_vectorized(
            states,
            float(dt),
            sigma=float(sigma),
            beta=float(beta),
            rho=float(rho),
        )
        if step < perturbations.shape[0]:
            states = states + (
                BRANCH_MULTIPLIERS[:, None, None]
                * perturbations[step][None, None, :]
            )
        trajectories[..., step] = states

    return trajectories


def _summarize_branch(trajectories):
    """Summarize regime risk for one branch forecast."""
    trajectories = np.asarray(trajectories, dtype=float)
    initial_sign = np.sign(trajectories[:, 0, 0])
    if np.any(initial_sign == 0.0):
        raise ValueError("An ensemble member starts exactly at x=0.")

    signed_x = initial_sign[:, None] * trajectories[:, 0, :]
    crossed = np.any(signed_x <= 0.0, axis=1)
    first_crossing = np.full(trajectories.shape[0], np.nan, dtype=float)
    for member_index in np.flatnonzero(crossed):
        first_crossing[member_index] = float(
            np.flatnonzero(signed_x[member_index] <= 0.0)[0]
        )

    minimum_margin = np.min(signed_x, axis=1)
    crossing_steps = first_crossing[np.isfinite(first_crossing)]
    return {
        "crossing_count": int(np.count_nonzero(crossed)),
        "crossing_fraction": float(np.mean(crossed)),
        "median_first_crossing_step": (
            float(np.median(crossing_steps)) if crossing_steps.size else np.nan
        ),
        "median_minimum_signed_x": float(np.median(minimum_margin)),
        "q25_minimum_signed_x": float(np.quantile(minimum_margin, 0.25)),
        "q75_minimum_signed_x": float(np.quantile(minimum_margin, 0.75)),
        "member_crossed": crossed,
        "member_first_crossing_step": first_crossing,
        "member_minimum_signed_x": minimum_margin,
    }


def _representative_cycles(controlled_cycles, free_risk):
    """Choose representative cycles using only pre-intervention forecasts."""
    controlled_cycles = np.asarray(controlled_cycles, dtype=int)
    free_risk = np.asarray(free_risk, dtype=float)
    chosen = []
    labels = {}

    def add(cycle, label):
        cycle = int(cycle)
        if cycle not in chosen:
            chosen.append(cycle)
            labels[cycle] = label

    add(controlled_cycles[0], "first CSE intervention")
    add(controlled_cycles[int(np.argmax(free_risk))], "maximum ensemble-predicted risk")
    all_change = np.flatnonzero(np.isclose(free_risk, 1.0))
    if all_change.size:
        add(controlled_cycles[int(all_change[0])], "first all-members-changing forecast")
    return chosen, labels


def build_counterfactual_diagnostics(
    cse_result,
    *,
    position,
    Ta,
    dt,
    representative_position=None,
):
    """Build ensemble counterfactuals without changing or rerunning CSE logic."""
    required = {
        "Xa_cycle_initial_history",
        "xf_4d_box_history",
        "change_ens_history",
        "intervention_history",
        "branch_history",
        "n_completed",
        "cse_summary",
    }
    missing = required.difference(cse_result)
    if missing:
        raise KeyError(f"CSE result is missing required keys: {sorted(missing)}")

    n_completed = int(cse_result["n_completed"])
    representative_position = (
        int(position)
        if representative_position is None
        else int(representative_position)
    )
    if representative_position < int(position):
        raise ValueError("representative_position cannot be shorter than position.")
    if n_completed != len(cse_result["intervention_history"]):
        raise ValueError("intervention_history length does not match n_completed.")

    controlled_cycles = np.array(
        [
            cycle
            for cycle, item in enumerate(cse_result["intervention_history"])
            if bool(item["controlled"])
        ],
        dtype=int,
    )
    if controlled_cycles.size == 0:
        raise ValueError("The CSE run contains no controlled cycles.")

    ensemble_size = np.asarray(
        cse_result["Xa_cycle_initial_history"][controlled_cycles[0]]
    ).shape[1]
    n_controlled = controlled_cycles.size
    metrics = {
        "crossing_fraction": np.empty((n_controlled, len(BRANCH_NAMES))),
        "crossing_count": np.empty((n_controlled, len(BRANCH_NAMES)), dtype=int),
        "median_first_crossing_step": np.empty(
            (n_controlled, len(BRANCH_NAMES))
        ),
        "median_minimum_signed_x": np.empty(
            (n_controlled, len(BRANCH_NAMES))
        ),
        "q25_minimum_signed_x": np.empty((n_controlled, len(BRANCH_NAMES))),
        "q75_minimum_signed_x": np.empty((n_controlled, len(BRANCH_NAMES))),
    }
    free_forecast_max_abs_error = np.empty(n_controlled)
    records = []

    for row, cycle in enumerate(controlled_cycles):
        item = cse_result["intervention_history"][int(cycle)]
        perturbations = np.asarray(item["perturbations"], dtype=float)
        if perturbations.shape != (int(Ta), 3):
            raise ValueError(
                f"Cycle {cycle} perturbations have shape {perturbations.shape}, "
                f"expected {(int(Ta), 3)}."
            )
        if not np.allclose(perturbations[:, :2], 0.0):
            raise ValueError(f"Cycle {cycle} contains a direct x or y intervention.")

        trajectories = simulate_counterfactual_ensemble(
            cse_result["Xa_cycle_initial_history"][int(cycle)],
            perturbations,
            position=int(position),
            dt=float(dt),
        )
        stored_free = np.asarray(cse_result["xf_4d_box_history"][int(cycle)])
        free_forecast_max_abs_error[row] = float(
            np.max(np.abs(trajectories[0] - stored_free))
        )

        branch_metrics = {}
        for branch_index, branch_name in enumerate(BRANCH_NAMES):
            summary = _summarize_branch(trajectories[branch_index])
            branch_metrics[branch_name] = summary
            for metric_name in metrics:
                metrics[metric_name][row, branch_index] = summary[metric_name]

        expected_count = len(cse_result["change_ens_history"][int(cycle)])
        if branch_metrics["free"]["crossing_count"] != expected_count:
            raise AssertionError(
                f"Cycle {cycle}: replay found "
                f"{branch_metrics['free']['crossing_count']} changing members, "
                f"but CSE stored {expected_count}."
            )

        records.append(
            {
                "cycle_index": int(cycle),
                "time_index": int(item["time_index"]),
                "absolute_time_index": int(item["absolute_time_index"]),
                "cse_branch": item["branch"],
                "p_change_before": int(expected_count),
                "risk_before": float(expected_count / ensemble_size),
                "z_intervention": perturbations[:, 2].copy(),
                "metadata": dict(item["metadata"]),
            }
        )

    if np.max(free_forecast_max_abs_error) > 1e-10:
        raise AssertionError(
            "Free counterfactual replay does not reproduce the stored ensemble "
            f"forecast (max error={np.max(free_forecast_max_abs_error):.3e})."
        )

    representative_cycles, representative_labels = _representative_cycles(
        controlled_cycles,
        metrics["crossing_fraction"][:, 0],
    )
    representative = {}
    for cycle in representative_cycles:
        row = int(np.flatnonzero(controlled_cycles == cycle)[0])
        item = cse_result["intervention_history"][int(cycle)]
        trajectories = simulate_counterfactual_ensemble(
            cse_result["Xa_cycle_initial_history"][int(cycle)],
            np.asarray(item["perturbations"], dtype=float),
            position=representative_position,
            dt=float(dt),
        )
        representative[cycle] = {
            "selection_label": representative_labels[cycle],
            "cycle_index": int(cycle),
            "time_index": int(records[row]["time_index"]),
            "absolute_time_index": int(records[row]["absolute_time_index"]),
            "z_intervention": records[row]["z_intervention"].copy(),
            "forecast_step": np.arange(representative_position + 1, dtype=int),
            "cse_forecast_horizon": int(position),
            "display_forecast_horizon": representative_position,
            "branches": {},
        }
        for branch_index, branch_name in enumerate(BRANCH_NAMES):
            branch_trajectories = trajectories[branch_index]
            cse_horizon_summary = _summarize_branch(
                branch_trajectories[..., :int(position) + 1]
            )
            display_horizon_summary = _summarize_branch(branch_trajectories)
            representative[cycle]["branches"][branch_name] = {
                "quantiles": np.quantile(
                    branch_trajectories,
                    [0.1, 0.5, 0.9],
                    axis=0,
                ),
                "member_crossed_at_cse_horizon": cse_horizon_summary[
                    "member_crossed"
                ],
                "member_crossed_at_display_horizon": display_horizon_summary[
                    "member_crossed"
                ],
            }

    free_risk = metrics["crossing_fraction"][:, 0]
    cse_risk = metrics["crossing_fraction"][:, 1]
    opposite_risk = metrics["crossing_fraction"][:, 2]
    free_margin = metrics["median_minimum_signed_x"][:, 0]
    cse_margin = metrics["median_minimum_signed_x"][:, 1]
    opposite_margin = metrics["median_minimum_signed_x"][:, 2]
    aggregate = {
        "mean_crossing_fraction": {
            name: float(np.mean(metrics["crossing_fraction"][:, index]))
            for index, name in enumerate(BRANCH_NAMES)
        },
        "median_crossing_fraction": {
            name: float(np.median(metrics["crossing_fraction"][:, index]))
            for index, name in enumerate(BRANCH_NAMES)
        },
        "mean_median_minimum_signed_x": {
            name: float(np.mean(metrics["median_minimum_signed_x"][:, index]))
            for index, name in enumerate(BRANCH_NAMES)
        },
        "cse_lower_risk_than_free_fraction": float(np.mean(cse_risk < free_risk)),
        "cse_higher_risk_than_free_fraction": float(np.mean(cse_risk > free_risk)),
        "cse_higher_margin_than_free_fraction": float(np.mean(cse_margin > free_margin)),
        "cse_lower_risk_than_opposite_fraction": float(
            np.mean(cse_risk < opposite_risk)
        ),
        "cse_higher_margin_than_opposite_fraction": float(
            np.mean(cse_margin > opposite_margin)
        ),
    }

    return {
        "schema_version": 1,
        "purpose": (
            "Post-decision ensemble counterfactuals at cycles selected by the "
            "original CSE algorithm"
        ),
        "information_policy": {
            "cycle_selection": (
                "cycles already activated by the original CSE ensemble forecast"
            ),
            "uncontrolled_nature_future_used_to_select_cycles": False,
            "counterfactual_branches_feed_back_into_cse": False,
        },
        "source_cse_summary": dict(cse_result["cse_summary"]),
        "branch_names": BRANCH_NAMES,
        "ensemble_size": int(ensemble_size),
        "position": int(position),
        "representative_position": representative_position,
        "Ta": int(Ta),
        "dt": float(dt),
        "controlled_cycles": controlled_cycles,
        "records": records,
        "metrics": metrics,
        "representative_cycles": representative,
        "aggregate": aggregate,
        "validation": {
            "free_forecast_max_abs_error": free_forecast_max_abs_error,
            "maximum_free_forecast_abs_error": float(
                np.max(free_forecast_max_abs_error)
            ),
            "all_interventions_z_only": True,
            "all_free_crossing_counts_match_cse": True,
        },
    }


def save_counterfactual_diagnostics(diagnostics, output_path):
    """Atomically save a NumPy-version-independent diagnostic dictionary."""
    def to_builtin(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: to_builtin(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(to_builtin(item) for item in value)
        if isinstance(value, list):
            return [to_builtin(item) for item in value]
        return value

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temporary_path, "wb") as stream:
        pickle.dump(to_builtin(diagnostics), stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(output_path)
    return output_path


def load_counterfactual_diagnostics(path):
    """Load and validate a saved counterfactual diagnostic dictionary."""
    path = Path(path)
    with open(path, "rb") as stream:
        diagnostics = pickle.load(stream)
    if diagnostics.get("schema_version") != 1:
        raise ValueError("Unsupported counterfactual diagnostic schema.")
    policy = diagnostics.get("information_policy", {})
    if policy.get("uncontrolled_nature_future_used_to_select_cycles") is not False:
        raise ValueError(
            "Diagnostic provenance does not exclude future uncontrolled-nature "
            "information from cycle selection."
        )
    return diagnostics


def _rolling_mean(values, window):
    values = np.asarray(values, dtype=float)
    if int(window) <= 1:
        return values.copy()
    kernel = np.ones(int(window), dtype=float)
    numerator = np.convolve(values, kernel, mode="same")
    denominator = np.convolve(np.ones_like(values), kernel, mode="same")
    return numerator / denominator


def plot_counterfactual_risk_summary(
    diagnostics,
    output_dir,
    *,
    rolling_window=25,
    output_stem="ensemble_counterfactual_risk_summary",
):
    """Plot aggregate ensemble risk and margin changes at controlled cycles."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cycles = np.asarray(diagnostics["controlled_cycles"], dtype=int)
    crossing = np.asarray(diagnostics["metrics"]["crossing_fraction"])
    margin = np.asarray(diagnostics["metrics"]["median_minimum_signed_x"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    for index, name in enumerate(BRANCH_NAMES):
        axes[0].plot(
            cycles,
            _rolling_mean(crossing[:, index], rolling_window),
            color=BRANCH_COLORS[name],
            linewidth=2.0,
            label=BRANCH_LABELS[name],
        )
    axes[0].set_ylabel("Crossing fraction")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_title(
        f"Forecast risk at CSE-selected cycles ({rolling_window}-cycle rolling mean)"
    )
    axes[0].legend(ncol=3, loc="upper right")

    axes[1].axhline(0.0, color="0.6", linewidth=1.0)
    axes[1].plot(
        cycles,
        _rolling_mean(crossing[:, 1] - crossing[:, 0], rolling_window),
        color=BRANCH_COLORS["cse"],
        linewidth=2.0,
        label="CSE minus no intervention",
    )
    axes[1].plot(
        cycles,
        _rolling_mean(crossing[:, 2] - crossing[:, 0], rolling_window),
        color=BRANCH_COLORS["opposite"],
        linewidth=1.7,
        label="Opposite minus no intervention",
    )
    axes[1].set_ylabel("Change in crossing fraction")
    axes[1].legend(ncol=2, loc="upper right")

    axes[2].axhline(0.0, color="0.6", linewidth=1.0)
    axes[2].plot(
        cycles,
        _rolling_mean(margin[:, 1] - margin[:, 0], rolling_window),
        color=BRANCH_COLORS["cse"],
        linewidth=2.0,
        label="CSE minus no intervention",
    )
    axes[2].plot(
        cycles,
        _rolling_mean(margin[:, 2] - margin[:, 0], rolling_window),
        color=BRANCH_COLORS["opposite"],
        linewidth=1.7,
        label="Opposite minus no intervention",
    )
    axes[2].set_xlabel("CSE cycle index")
    axes[2].set_ylabel("Change in median minimum\nsigned x margin")
    axes[2].legend(ncol=2, loc="upper right")

    for ax in axes:
        ax.grid(alpha=0.2)

    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return fig, axes, png_path, pdf_path, diagnostics["aggregate"]


def plot_representative_counterfactual(
    diagnostics,
    output_dir,
    *,
    cycle=None,
    output_stem="ensemble_counterfactual_representative_cycle",
):
    """Plot ensemble quantiles for an ensemble-selected representative cycle."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    representative = diagnostics["representative_cycles"]
    if cycle is None:
        cycle = next(
            (
                key
                for key, value in representative.items()
                if value["selection_label"] == "first CSE intervention"
            ),
            next(iter(representative)),
        )
    cycle = int(cycle)
    if cycle not in representative:
        raise KeyError(f"Cycle {cycle} was not saved as a representative cycle.")

    record = representative[cycle]
    steps = np.asarray(record["forecast_step"], dtype=int)
    component_names = ("x", "y", "z")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    for component, ax in enumerate(axes):
        for name in BRANCH_NAMES:
            quantiles = np.asarray(record["branches"][name]["quantiles"])
            lower = quantiles[0, component]
            median = quantiles[1, component]
            upper = quantiles[2, component]
            ax.fill_between(
                steps,
                lower,
                upper,
                color=BRANCH_COLORS[name],
                alpha=0.10,
                linewidth=0,
            )
            ax.plot(
                steps,
                median,
                color=BRANCH_COLORS[name],
                linewidth=1.8,
                label=BRANCH_LABELS[name],
            )
        if component < 2:
            ax.axhline(0.0, color="0.55", linestyle="--", linewidth=1.0)
        ax.axvspan(1, diagnostics["Ta"] - 1, color="tab:green", alpha=0.08)
        ax.set_ylabel(component_names[component])
        ax.grid(alpha=0.2)

    crossing_fractions = {
        name: float(
            np.mean(
                record["branches"][name]["member_crossed_at_cse_horizon"]
            )
        )
        for name in BRANCH_NAMES
    }
    axes[0].set_title(
        f"Cycle {cycle}: {record['selection_label']} | crossing fraction by "
        f"step {record['cse_forecast_horizon']}: "
        f"free={crossing_fractions['free']:.2f}, "
        f"CSE={crossing_fractions['cse']:.2f}, "
        f"opposite={crossing_fractions['opposite']:.2f}\n"
        "Median and 10–90% ensemble interval; green band marks the intervention packet"
    )
    axes[0].legend(ncol=3, loc="upper right")
    axes[-1].set_xlabel("Forecast step from the CSE decision")

    tagged_stem = f"{output_stem}_{cycle:04d}"
    png_path = output_dir / f"{tagged_stem}.png"
    pdf_path = output_dir / f"{tagged_stem}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return fig, axes, png_path, pdf_path, record
