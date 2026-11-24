"""Shared loading, validation, and plotting helpers for CLE TARGET notebooks."""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np


def lorenz63_rhs(states, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """Evaluate the free Lorenz-63 tendency for one state or an ensemble."""
    states = np.asarray(states, dtype=float)
    x = states[..., 0]
    y = states[..., 1]
    z = states[..., 2]
    return np.stack(
        [
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ],
        axis=-1,
    )


def rk4_free_step(states, dt, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """Advance the unforced Lorenz-63 model by one RK4 step."""

    def rhs(values):
        return lorenz63_rhs(values, sigma=sigma, rho=rho, beta=beta)

    states = np.asarray(states, dtype=float)
    k1 = rhs(states)
    k2 = rhs(states + 0.5 * dt * k1)
    k3 = rhs(states + 0.5 * dt * k2)
    k4 = rhs(states + dt * k3)
    return states + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def load_and_validate_cse_target_inputs(input_data_dir, tolerance=1e-10):
    """Load the CSE TARGET files and verify their discrete-step alignment.

    The required identity is

        TARGET[n + 1] = RK4_free(TARGET[n]) + intervention[n + 1].

    Interventions are therefore discrete state increments, not continuous
    forcing terms in the Lorenz-63 right-hand side.
    """
    input_data_dir = Path(input_data_dir)
    target_file = input_data_dir / "cse_target.pkl"
    interventions_file = input_data_dir / "cse_interventions.pkl"
    summary_file = input_data_dir / "cse_summary.pkl"

    for required_file in (target_file, interventions_file, summary_file):
        if not required_file.exists():
            raise FileNotFoundError(
                f"Missing {required_file}. Run notebook 6, including "
                "Run Single CSE Event, first."
            )

    with target_file.open("rb") as file_handle:
        target_array = np.asarray(pickle.load(file_handle), dtype=float)
    with interventions_file.open("rb") as file_handle:
        intervention_history = pickle.load(file_handle)
    with summary_file.open("rb") as file_handle:
        cse_summary = pickle.load(file_handle)

    if target_array.ndim != 3:
        raise ValueError(
            f"Expected TARGET shape (cycles, Ta, xdim), got {target_array.shape}."
        )

    num_cycles, assimilation_window, state_dimension = target_array.shape
    if len(intervention_history) != num_cycles:
        raise ValueError("TARGET and intervention histories have different cycle counts.")

    intervention_array = np.stack(
        [np.asarray(item["perturbations"], dtype=float) for item in intervention_history]
    )
    controlled_flags = np.asarray(
        [bool(item["controlled"]) for item in intervention_history]
    )
    branches = [item["branch"] for item in intervention_history]

    if intervention_array.shape != target_array.shape:
        raise ValueError(
            f"Interventions {intervention_array.shape} do not match "
            f"TARGET {target_array.shape}."
        )
    if int(cse_summary["Ta"]) != assimilation_window:
        raise ValueError("Ta in cse_summary does not match the TARGET array.")
    if not np.allclose(intervention_array[:, 0, :], 0.0, atol=0.0, rtol=0.0):
        raise ValueError("The first intervention row of each cycle must be exactly zero.")

    dt = float(cse_summary.get("fcst_dt", 0.01))
    target_flat = target_array.reshape(-1, state_dimension)
    intervention_flat = intervention_array.reshape(-1, state_dimension)
    reconstructed_next = (
        rk4_free_step(target_flat[:-1], dt=dt) + intervention_flat[1:]
    )
    reconstruction_error = np.linalg.norm(
        reconstructed_next - target_flat[1:], axis=1
    )
    max_reconstruction_error = float(np.max(reconstruction_error))
    if max_reconstruction_error > tolerance:
        raise ValueError(
            "Stored TARGET is inconsistent with free RK4 plus discrete impulses: "
            f"max error={max_reconstruction_error:.3e}."
        )

    computed_flags = np.any(
        np.linalg.norm(intervention_array, axis=2) > 0, axis=1
    )
    if not np.array_equal(computed_flags, controlled_flags):
        raise ValueError("Stored controlled flags do not match intervention norms.")

    return {
        "target_file": target_file,
        "interventions_file": interventions_file,
        "summary_file": summary_file,
        "target_array": target_array,
        "intervention_history": intervention_history,
        "cse_summary": cse_summary,
        "intervention_array": intervention_array,
        "controlled_flags": controlled_flags,
        "branches": branches,
        "num_cycles": num_cycles,
        "Ta": assimilation_window,
        "xdim": state_dimension,
        "dt": dt,
        "target_flat": target_flat,
        "intervention_flat": intervention_flat,
        "reconstruction_error": reconstruction_error,
        "max_reconstruction_error": max_reconstruction_error,
    }


def print_cse_target_validation_summary(target_data):
    """Print the compact input and discrete-map validation summary."""
    controlled_flags = target_data["controlled_flags"]
    branches = target_data["branches"]
    cse_summary = target_data["cse_summary"]
    print("TARGET shape:", target_data["target_array"].shape)
    print("Intervention shape:", target_data["intervention_array"].shape)
    print("Controlled cycles:", int(controlled_flags.sum()))
    print("Free cycles:", int(target_data["num_cycles"] - controlled_flags.sum()))
    print("Branches:", {name: branches.count(name) for name in sorted(set(branches))})
    print("Event / D:", cse_summary.get("run_cse_event_index"), cse_summary.get("D"))
    print("Intervention application:", cse_summary.get("intervention_application"))
    print(
        "Maximum discrete-map reconstruction error:",
        target_data["max_reconstruction_error"],
    )
    print("✓ TARGET and interventions are exactly aligned with the discrete CSE map.")


def plot_cse_target_and_interventions(
    target_data,
    figures_dir,
    control_label="xyz",
):
    """Reproduce the TARGET and discrete-intervention overview plot."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    target_flat = target_data["target_flat"]
    intervention_flat = target_data["intervention_flat"]
    controlled_flags = target_data["controlled_flags"]
    cycle_start_indices = np.arange(target_data["num_cycles"]) * target_data["Ta"]
    time_axis = np.arange(target_flat.shape[0])
    intervention_norm_flat = np.linalg.norm(intervention_flat, axis=1)

    fig, axes = plt.subplots(
        2, 1, figsize=(16, 8), sharex=True, constrained_layout=True
    )
    axes[0].plot(
        time_axis,
        target_flat[:, 0],
        color="tab:red",
        linewidth=1.2,
        label="controlled TARGET x",
    )
    axes[0].scatter(
        cycle_start_indices[controlled_flags],
        target_flat[cycle_start_indices[controlled_flags], 0],
        s=9,
        color="black",
        alpha=0.65,
        label="controlled cycle start",
    )
    axes[0].axhline(0.0, color="0.5", linestyle="--", linewidth=0.9)
    axes[0].set_ylabel("x")
    axes[0].set_title("Controlled TARGET trajectory")
    axes[0].legend(loc="best")

    axes[1].plot(time_axis, intervention_norm_flat, color="tab:blue", linewidth=1.0)
    axes[1].set_xlabel("Model step from CSE start")
    axes[1].set_ylabel(r"$\|u_n\|$")
    axes[1].set_title("Exact discrete intervention norm")
    axes[1].grid(alpha=0.2)

    output_path = figures_dir / (
        f"target_and_discrete_interventions_{control_label}_control.png"
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig, axes, output_path

