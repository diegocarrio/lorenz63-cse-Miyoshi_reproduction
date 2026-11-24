"""Diagnostics for explaining control of the Lorenz-63 nature run."""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np


def load_nature_run_comparison(path):
    """Load and validate a direct free-versus-controlled nature-run artifact."""
    path = Path(path)
    with path.open("rb") as file_handle:
        data = pickle.load(file_handle)

    required = {
        "nature_control",
        "nature_free",
        "interventions",
        "branch_history",
        "summary",
        "provenance",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise KeyError(f"Nature-run artifact is missing fields: {missing}")

    nature_control = np.asarray(data["nature_control"], dtype=float)
    nature_free = np.asarray(data["nature_free"], dtype=float)
    interventions = np.asarray(data["interventions"], dtype=float)
    if nature_control.ndim != 2 or nature_control.shape[0] != 3:
        raise ValueError("nature_control must have shape (3, time).")
    if nature_free.shape != nature_control.shape:
        raise ValueError("nature_free and nature_control must have identical shapes.")
    if interventions.shape != nature_control.shape:
        raise ValueError("interventions must align with both nature trajectories.")
    if not np.allclose(nature_control[:, 0], nature_free[:, 0]):
        raise ValueError("The free and controlled nature runs must share their initial state.")
    if data["provenance"].get("target_export_used") is not False:
        raise ValueError("Expected a direct CSE nature-run artifact, not a TARGET export.")

    data = dict(data)
    data["nature_control"] = nature_control
    data["nature_free"] = nature_free
    data["interventions"] = interventions
    data["path"] = path
    return data


def plot_nature_free_vs_nature_control(
    data,
    figures_dir,
    zoom_before=200,
    zoom_after=300,
    output_stem="z_only_nature_free_vs_nature_control",
):
    """Compare the free and z-only-controlled nature runs in three panels."""
    nature_control = np.asarray(data["nature_control"], dtype=float).T
    nature_free = np.asarray(data["nature_free"], dtype=float).T
    interventions = np.asarray(data["interventions"], dtype=float).T

    if not np.all(interventions[:, :2] == 0.0):
        raise ValueError("Expected exactly zero direct interventions in x and y.")
    if not np.any(interventions[:, 2] != 0.0):
        raise ValueError("No nonzero z interventions were found.")

    time_axis = np.arange(nature_control.shape[0])
    initial_sign = np.sign(nature_control[0, 0])
    if initial_sign == 0:
        raise ValueError("The initial x state must be nonzero to define the regime.")

    free_crossings = np.flatnonzero(nature_free[:, 0] * initial_sign <= 0.0)
    if not free_crossings.size:
        raise ValueError("The free nature run does not leave its initial regime.")
    first_free_crossing = int(free_crossings[0])
    controlled_crossings = np.flatnonzero(
        nature_control[:, 0] * initial_sign <= 0.0
    )

    zoom_start = max(0, first_free_crossing - int(zoom_before))
    zoom_stop = min(
        nature_control.shape[0],
        first_free_crossing + int(zoom_after) + 1,
    )
    zoom_slice = slice(zoom_start, zoom_stop)
    active_zoom = (
        np.flatnonzero(interventions[zoom_slice, 2] != 0.0) + zoom_start
    )

    free_color = "#4D4D4D"
    controlled_color = "#D55E00"
    intervention_color = "#0072B2"
    crossing_color = "#7A5195"

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 9),
        gridspec_kw={"height_ratios": (1.15, 1.15, 0.7)},
        constrained_layout=True,
    )
    fig.suptitle(
        "z-only CSE keeps the nature run in its initial Lorenz regime",
        fontsize=15,
        fontweight="semibold",
    )

    overview_ax, zoom_ax, intervention_ax = axes
    overview_ax.plot(
        time_axis,
        nature_free[:, 0],
        color=free_color,
        linewidth=0.8,
        alpha=0.72,
        label="free nature run",
    )
    overview_ax.plot(
        time_axis,
        nature_control[:, 0],
        color=controlled_color,
        linewidth=1.0,
        label="controlled nature run (z-only CSE)",
    )
    overview_ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    overview_ax.axvline(
        first_free_crossing,
        color=crossing_color,
        linestyle=":",
        linewidth=1.4,
        label=f"first free crossing (step {first_free_crossing})",
    )
    overview_ax.axvspan(
        zoom_start,
        zoom_stop - 1,
        color=intervention_color,
        alpha=0.08,
        label="zoomed interval",
    )
    overview_ax.set_xlim(0, nature_control.shape[0] - 1)
    overview_ax.set_ylabel("x")
    overview_ax.set_title(
        "(a) Free switching versus controlled regime retention",
        loc="left",
    )
    overview_ax.legend(loc="upper right", ncols=2, frameon=True)
    overview_ax.grid(alpha=0.18)

    zoom_time = time_axis[zoom_slice]
    zoom_ax.plot(
        zoom_time,
        nature_free[zoom_slice, 0],
        color=free_color,
        linewidth=1.5,
        label="free nature run",
    )
    zoom_ax.plot(
        zoom_time,
        nature_control[zoom_slice, 0],
        color=controlled_color,
        linewidth=1.7,
        label="controlled nature run",
    )
    zoom_ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    zoom_ax.axvline(
        first_free_crossing,
        color=crossing_color,
        linestyle=":",
        linewidth=1.4,
    )
    zoom_ax.annotate(
        "free nature run leaves\nthe initial regime",
        xy=(first_free_crossing, nature_free[first_free_crossing, 0]),
        xytext=(0.57, 0.18 if initial_sign > 0 else 0.82),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": crossing_color, "lw": 1.2},
        color=crossing_color,
        fontsize=10,
        ha="left",
    )
    controlled_margin = float(np.min(nature_control[:, 0] * initial_sign))
    zoom_ax.text(
        0.99,
        0.96,
        f"minimum controlled regime margin = {controlled_margin:.2f}",
        transform=zoom_ax.transAxes,
        ha="right",
        va="top",
        color=controlled_color,
        bbox={"facecolor": "white", "edgecolor": controlled_color, "alpha": 0.9},
    )
    zoom_ax.set_xlim(zoom_start, zoom_stop - 1)
    zoom_ax.set_ylabel("x")
    zoom_ax.set_title("(b) First avoided transition", loc="left")
    zoom_ax.grid(alpha=0.18)

    intervention_ax.axhline(0.0, color="black", linewidth=0.8)
    intervention_ax.vlines(
        active_zoom,
        0.0,
        interventions[active_zoom, 2],
        color=intervention_color,
        linewidth=1.0,
        alpha=0.8,
    )
    intervention_ax.scatter(
        active_zoom,
        interventions[active_zoom, 2],
        color=intervention_color,
        s=9,
        zorder=3,
    )
    intervention_ax.axvline(
        first_free_crossing,
        color=crossing_color,
        linestyle=":",
        linewidth=1.4,
    )
    max_intervention = float(np.max(np.abs(interventions[:, 2])))
    intervention_ax.set_ylim(-1.2 * max_intervention, 1.2 * max_intervention)
    intervention_ax.set_xlim(zoom_start, zoom_stop - 1)
    intervention_ax.set_xlabel("Model step from CSE start")
    intervention_ax.set_ylabel(r"$u_z$")
    intervention_ax.set_title(
        f"(c) Discrete state increments: "
        rf"$u_x=u_y=0$, $\max |u_z|={max_intervention:.2f}$",
        loc="left",
    )
    intervention_ax.grid(axis="y", alpha=0.18)

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / f"{output_stem}.png"
    pdf_path = figures_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    diagnostics = {
        "first_free_crossing": first_free_crossing,
        "controlled_crossing": (
            None if not controlled_crossings.size else int(controlled_crossings[0])
        ),
        "controlled_regime_margin": controlled_margin,
        "nonzero_z_interventions": int(np.count_nonzero(interventions[:, 2])),
        "max_abs_z_intervention": max_intervention,
    }
    return fig, axes, png_path, pdf_path, diagnostics


def plot_z_intervention_propagation(
    data,
    figures_dir,
    n_steps=20,
    output_stem="z_intervention_propagation_first_20_steps",
):
    """Show how direct z increments propagate to y and x over early steps."""
    nature_control = np.asarray(data["nature_control"], dtype=float)
    nature_free = np.asarray(data["nature_free"], dtype=float)
    interventions = np.asarray(data["interventions"], dtype=float)

    if not 1 <= int(n_steps) < nature_control.shape[1]:
        raise ValueError("n_steps must be within the stored trajectory.")
    if not np.all(interventions[:2] == 0.0):
        raise ValueError("Expected exactly zero direct interventions in x and y.")

    stop = int(n_steps) + 1
    steps = np.arange(stop)
    delta = nature_control[:, :stop] - nature_free[:, :stop]
    uz = interventions[2, :stop]
    tolerance = 1.0e-14

    def first_nonzero(values):
        indices = np.flatnonzero(np.abs(values) > tolerance)
        return None if not indices.size else int(indices[0])

    first_u = first_nonzero(uz)
    first_delta_z = first_nonzero(delta[2])
    first_delta_y = first_nonzero(delta[1])
    first_delta_x = first_nonzero(delta[0])
    if first_u is None:
        raise ValueError("No z intervention occurs in the requested interval.")

    colors = {
        "uz": "#0072B2",
        "dz": "#009E73",
        "dy": "#7A5195",
        "dx": "#D55E00",
        "guide": "#777777",
    }
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(13, 10),
        sharex=True,
        constrained_layout=True,
    )
    fig.suptitle(
        "Propagation of z-only CSE increments during the first 20 model steps",
        fontsize=15,
        fontweight="semibold",
    )

    cycle_length = int(data["summary"]["Ta"])
    for ax in axes:
        ax.axhline(0.0, color="black", linewidth=0.8)
        for boundary in range(0, stop, cycle_length):
            ax.axvline(
                boundary,
                color=colors["guide"],
                linestyle=":",
                linewidth=0.8,
                alpha=0.55,
            )
        ax.grid(axis="y", alpha=0.18)

    active_steps = np.flatnonzero(uz != 0.0)
    axes[0].vlines(
        active_steps,
        0.0,
        uz[active_steps],
        color=colors["uz"],
        linewidth=2.0,
    )
    axes[0].scatter(
        active_steps,
        uz[active_steps],
        color=colors["uz"],
        s=32,
        zorder=3,
    )
    axes[0].set_ylabel(r"$u_z$")
    axes[0].set_title(
        r"(a) Direct action: discrete $z$ increments ($u_x=u_y=0$)",
        loc="left",
    )
    axes[0].annotate(
        f"first direct increment\n$u_z={uz[first_u]:+.2f}$",
        xy=(first_u, uz[first_u]),
        xytext=(3.0, -0.020 if uz[first_u] < 0 else 0.020),
        arrowprops={"arrowstyle": "->", "color": colors["uz"]},
        color=colors["uz"],
    )

    series = (
        (2, colors["dz"], r"$\Delta z$", "(b) Immediate response in the directly actuated variable"),
        (1, colors["dy"], r"$\Delta y$", r"(c) Coupled response: $\delta\dot{y}=-x\,\delta z$"),
        (0, colors["dx"], r"$\Delta x$", r"(d) Downstream response: $\delta y$ changes $\dot{x}=\sigma(y-x)$"),
    )
    for ax, (component, color, ylabel, title) in zip(axes[1:], series):
        ax.plot(
            steps,
            delta[component],
            color=color,
            marker="o",
            markersize=4.5,
            linewidth=1.8,
        )
        ax.fill_between(
            steps,
            0.0,
            delta[component],
            color=color,
            alpha=0.10,
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")

    propagation = (
        (axes[1], first_delta_z, delta[2], colors["dz"], "direct jump appears here"),
        (axes[2], first_delta_y, delta[1], colors["dy"], "y responds after free RK4 evolution"),
        (axes[3], first_delta_x, delta[0], colors["dx"], "x responds through the changed y dynamics"),
    )
    for ax, step, values, color, label in propagation:
        if step is None:
            continue
        ax.scatter(step, values[step], color=color, s=55, zorder=4)
        ax.annotate(
            f"step {step}: {values[step]:+.6f}\n{label}",
            xy=(step, values[step]),
            xytext=(44, 16),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": color},
            color=color,
            va="bottom",
        )

    axes[-1].set_xlabel("Model step from CSE start")
    axes[-1].set_xlim(0, int(n_steps))
    axes[-1].set_xticks(np.arange(0, stop, 2))

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / f"{output_stem}.png"
    pdf_path = figures_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    diagnostics = {
        "first_z_intervention_step": first_u,
        "first_delta_z_step": first_delta_z,
        "first_delta_y_step": first_delta_y,
        "first_delta_x_step": first_delta_x,
        "first_z_intervention": float(uz[first_u]),
        "delta_at_step_1": delta[:, 1].copy(),
        "delta_at_step_2": delta[:, 2].copy(),
    }
    return fig, axes, png_path, pdf_path, diagnostics


def plot_state_values_controlled_vs_free(
    data,
    figures_dir,
    n_steps=20,
    output_stem="state_values_controlled_vs_free_first_20_steps",
):
    """Compare the actual controlled and free z, y, x values at early steps."""
    nature_control = np.asarray(data["nature_control"], dtype=float)
    nature_free = np.asarray(data["nature_free"], dtype=float)
    interventions = np.asarray(data["interventions"], dtype=float)

    if not 1 <= int(n_steps) < nature_control.shape[1]:
        raise ValueError("n_steps must be within the stored trajectory.")
    if not np.all(interventions[:2] == 0.0):
        raise ValueError("Expected exactly zero direct interventions in x and y.")

    stop = int(n_steps) + 1
    steps = np.arange(stop)
    uz = interventions[2, :stop]
    active_steps = np.flatnonzero(uz != 0.0)
    first_u = int(active_steps[0])

    colors = {
        "free": "#4D4D4D",
        "uz": "#0072B2",
        "z": "#009E73",
        "y": "#7A5195",
        "x": "#D55E00",
        "guide": "#777777",
    }
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(13, 10),
        sharex=True,
        constrained_layout=True,
    )
    fig.suptitle(
        "Controlled and free state values during the first 20 model steps",
        fontsize=15,
        fontweight="semibold",
    )

    cycle_length = int(data["summary"]["Ta"])
    for ax in axes:
        for boundary in range(0, stop, cycle_length):
            ax.axvline(
                boundary,
                color=colors["guide"],
                linestyle=":",
                linewidth=0.8,
                alpha=0.55,
            )
        ax.axvline(
            first_u,
            color=colors["uz"],
            linestyle="--",
            linewidth=1.0,
            alpha=0.35,
        )
        ax.grid(axis="y", alpha=0.18)

    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].vlines(
        active_steps,
        0.0,
        uz[active_steps],
        color=colors["uz"],
        linewidth=2.0,
    )
    axes[0].scatter(
        active_steps,
        uz[active_steps],
        color=colors["uz"],
        s=32,
        zorder=3,
    )
    axes[0].set_ylabel(r"$u_z$")
    axes[0].set_title(
        r"(a) Applied discrete increments ($u_x=u_y=0$)",
        loc="left",
    )

    components = (
        (2, "z", colors["z"], "(b) Directly actuated state"),
        (1, "y", colors["y"], "(c) First coupled state"),
        (0, "x", colors["x"], "(d) Downstream state"),
    )
    final_differences = {}
    for panel_index, (component, label, color, title) in enumerate(
        components,
        start=1,
    ):
        ax = axes[panel_index]
        controlled_values = nature_control[component, :stop]
        free_values = nature_free[component, :stop]
        difference = controlled_values - free_values
        final_differences[label] = float(difference[-1])

        ax.plot(
            steps,
            free_values,
            color=colors["free"],
            linestyle="--",
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            label="free nature run",
        )
        ax.plot(
            steps,
            controlled_values,
            color=color,
            marker="o",
            markersize=4.0,
            linewidth=1.8,
            label="controlled nature run",
        )
        ax.fill_between(
            steps,
            free_values,
            controlled_values,
            color=color,
            alpha=0.14,
            label="control-induced separation",
        )
        ax.set_ylabel(rf"${label}$")
        ax.set_title(title, loc="left")
        ax.text(
            0.985,
            0.94,
            (
                f"step {n_steps}:  free = {free_values[-1]:.6f}\n"
                f"controlled = {controlled_values[-1]:.6f}\n"
                rf"$\Delta {label}$ = {difference[-1]:+.6f}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=color,
            bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.9},
        )

    axes[1].legend(loc="lower left", ncols=3, frameon=True)
    axes[-1].set_xlabel("Model step from CSE start")
    axes[-1].set_xlim(0, int(n_steps))
    axes[-1].set_xticks(np.arange(0, stop, 2))

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / f"{output_stem}.png"
    pdf_path = figures_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    diagnostics = {
        "first_z_intervention_step": first_u,
        "final_step": int(n_steps),
        "final_differences": final_differences,
        "free_final_state": nature_free[:, int(n_steps)].copy(),
        "controlled_final_state": nature_control[:, int(n_steps)].copy(),
    }
    return fig, axes, png_path, pdf_path, diagnostics


def plot_state_values_around_free_crossing(
    data,
    figures_dir,
    center_step=306,
    half_window=10,
    output_stem="state_values_around_free_crossing_step_306",
):
    """Compare actual state values around the free nature-run regime crossing."""
    nature_control = np.asarray(data["nature_control"], dtype=float)
    nature_free = np.asarray(data["nature_free"], dtype=float)
    interventions = np.asarray(data["interventions"], dtype=float)

    center_step = int(center_step)
    half_window = int(half_window)
    start = center_step - half_window
    stop = center_step + half_window + 1
    if start < 0 or stop > nature_control.shape[1]:
        raise ValueError("The requested crossing window is outside the trajectory.")
    if not np.all(interventions[:2] == 0.0):
        raise ValueError("Expected exactly zero direct interventions in x and y.")

    initial_sign = np.sign(nature_control[0, 0])
    free_crossings = np.flatnonzero(nature_free[0] * initial_sign <= 0.0)
    if not free_crossings.size or int(free_crossings[0]) != center_step:
        raise ValueError(
            "center_step must be the first free x-regime crossing for this diagnostic."
        )
    if nature_control[0, center_step] * initial_sign <= 0.0:
        raise ValueError("The controlled run also crosses at center_step.")

    steps = np.arange(start, stop)
    uz = interventions[2, start:stop]
    active_steps = steps[uz != 0.0]
    active_values = uz[uz != 0.0]
    colors = {
        "free": "#4D4D4D",
        "uz": "#0072B2",
        "z": "#009E73",
        "y": "#7A5195",
        "x": "#D55E00",
        "crossing": "#CC79A7",
        "guide": "#777777",
    }

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(13, 10),
        sharex=True,
        constrained_layout=True,
    )
    fig.suptitle(
        "z-only CSE avoids the free regime transition at model step 306",
        fontsize=15,
        fontweight="semibold",
    )

    cycle_length = int(data["summary"]["Ta"])
    first_boundary = (start // cycle_length) * cycle_length
    for ax in axes:
        for boundary in range(first_boundary, stop, cycle_length):
            if boundary >= start:
                ax.axvline(
                    boundary,
                    color=colors["guide"],
                    linestyle=":",
                    linewidth=0.8,
                    alpha=0.55,
                )
        ax.axvline(
            center_step,
            color=colors["crossing"],
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
        )
        ax.grid(axis="y", alpha=0.18)

    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].vlines(
        active_steps,
        0.0,
        active_values,
        color=colors["uz"],
        linewidth=2.0,
    )
    axes[0].scatter(
        active_steps,
        active_values,
        color=colors["uz"],
        s=32,
        zorder=3,
    )
    axes[0].set_ylabel(r"$u_z$")
    axes[0].set_title(
        r"(a) Applied discrete increments ($u_x=u_y=0$)",
        loc="left",
    )
    axes[0].text(
        center_step + 0.25,
        0.0,
        "first free $x=0$ crossing",
        color=colors["crossing"],
        va="bottom",
    )

    components = (
        (2, "z", colors["z"], "(b) Directly actuated state"),
        (1, "y", colors["y"], "(c) y remains positive only in the controlled run"),
        (0, "x", colors["x"], "(d) x crossing is avoided by the controlled run"),
    )
    crossing_values = {}
    for panel_index, (component, label, color, title) in enumerate(
        components,
        start=1,
    ):
        ax = axes[panel_index]
        controlled_values = nature_control[component, start:stop]
        free_values = nature_free[component, start:stop]
        crossing_values[label] = {
            "free": float(nature_free[component, center_step]),
            "controlled": float(nature_control[component, center_step]),
        }

        if label in {"x", "y"}:
            ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.plot(
            steps,
            free_values,
            color=colors["free"],
            linestyle="--",
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            label="free nature run",
        )
        ax.plot(
            steps,
            controlled_values,
            color=color,
            marker="o",
            markersize=4.0,
            linewidth=1.8,
            label="controlled nature run",
        )
        ax.fill_between(
            steps,
            free_values,
            controlled_values,
            color=color,
            alpha=0.14,
            label="control-induced separation",
        )
        ax.scatter(
            center_step,
            nature_free[component, center_step],
            color=colors["free"],
            edgecolor=colors["crossing"],
            linewidth=1.8,
            s=70,
            zorder=5,
        )
        ax.scatter(
            center_step,
            nature_control[component, center_step],
            color=color,
            edgecolor=colors["crossing"],
            linewidth=1.8,
            s=70,
            zorder=5,
        )
        ax.set_ylabel(rf"${label}$")
        ax.set_title(title, loc="left")
        ax.text(
            0.985,
            0.94,
            (
                f"step {center_step}:  free = "
                f"{nature_free[component, center_step]:+.6f}\n"
                f"controlled = {nature_control[component, center_step]:+.6f}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=color,
            bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.9},
        )

    axes[1].legend(loc="lower left", ncols=3, frameon=True)
    axes[-1].set_xlabel("Model step from CSE start")
    axes[-1].set_xlim(start, stop - 1)
    axes[-1].set_xticks(np.arange(start, stop, 2))

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / f"{output_stem}.png"
    pdf_path = figures_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    diagnostics = {
        "window": (start, stop - 1),
        "crossing_step": center_step,
        "crossing_values": crossing_values,
        "controlled_min_x": float(np.min(nature_control[0, start:stop])),
        "controlled_min_y": float(np.min(nature_control[1, start:stop])),
        "free_max_x_before_crossing": float(
            np.max(nature_free[0, start:center_step + 1])
        ),
    }
    return fig, axes, png_path, pdf_path, diagnostics


def plot_3d_z_interventions_on_lorenz_attractor(
    data,
    figures_dir,
    *,
    n_steps=400,
    arrow_scale=40.0,
    max_arrows=30,
    output_stem="figure_07_cse_z_only_3d_attractor_interventions",
):
    """Show where z-only interventions act and which Lorenz lobe is retained.

    The left panel provides the full free nature-run attractor as context and
    compares free and controlled paths over the selected window.  The right
    panel marks every intervention state and displays a representative subset
    of the direct z increments as scaled vertical arrows.  Scaling changes only
    arrow visibility; trajectory coordinates remain unmodified.
    """
    nature_control = np.asarray(data["nature_control"], dtype=float)
    nature_free = np.asarray(data["nature_free"], dtype=float)
    interventions = np.asarray(data["interventions"], dtype=float)
    if nature_control.shape != nature_free.shape:
        raise ValueError("Controlled and free nature trajectories must match.")
    if interventions.shape != nature_control.shape:
        raise ValueError("Interventions must align with the nature trajectories.")
    if not np.allclose(interventions[:2], 0.0):
        raise ValueError("The requested figure requires strictly z-only control.")

    n_steps = int(n_steps)
    if n_steps < 1 or n_steps >= nature_control.shape[1]:
        raise ValueError("n_steps must lie inside the saved trajectory.")
    stop = n_steps + 1
    steps = np.arange(stop)
    Ta = int(data["summary"]["Ta"])

    intervention_norm = np.linalg.norm(interventions, axis=0)
    intervention_steps = np.flatnonzero(intervention_norm[:stop] > 0.0)
    if intervention_steps.size == 0:
        raise ValueError("No intervention occurs inside the requested window.")

    # Use one direct-increment arrow per controlled DA cycle, then thin only if
    # necessary. Every intervention state is still shown as a colored point.
    cycle_ids = intervention_steps // Ta
    _, first_in_cycle = np.unique(cycle_ids, return_index=True)
    arrow_steps = intervention_steps[np.sort(first_in_cycle)]
    if arrow_steps.size > int(max_arrows):
        keep = np.linspace(0, arrow_steps.size - 1, int(max_arrows), dtype=int)
        arrow_steps = arrow_steps[np.unique(keep)]

    initial_sign = np.sign(nature_free[0, 0])
    free_crossings = np.flatnonzero(nature_free[0, :stop] * initial_sign <= 0.0)
    first_free_crossing = int(free_crossings[0]) if free_crossings.size else None

    colors = {
        "attractor": "0.72",
        "free": "tab:orange",
        "controlled": "tab:blue",
        "intervention": "magenta",
        "positive_uz": "crimson",
        "negative_uz": "purple",
        "start": "tab:green",
    }
    fig = plt.figure(figsize=(18, 8), constrained_layout=True)
    ax_global = fig.add_subplot(1, 2, 1, projection="3d")
    ax_zoom = fig.add_subplot(1, 2, 2, projection="3d")

    ax_global.plot(
        nature_free[0],
        nature_free[1],
        nature_free[2],
        color=colors["attractor"],
        linewidth=0.45,
        alpha=0.38,
        label="free nature attractor (full saved interval)",
    )
    ax_global.plot(
        nature_free[0, :stop],
        nature_free[1, :stop],
        nature_free[2, :stop],
        color=colors["free"],
        linewidth=2.0,
        alpha=0.9,
        label=f"free path, steps 0–{n_steps}",
    )
    ax_global.plot(
        nature_control[0, :stop],
        nature_control[1, :stop],
        nature_control[2, :stop],
        color=colors["controlled"],
        linewidth=2.2,
        label=f"controlled path, steps 0–{n_steps}",
    )
    ax_global.scatter(
        nature_control[0, 0],
        nature_control[1, 0],
        nature_control[2, 0],
        color=colors["start"],
        marker="*",
        s=130,
        edgecolor="black",
        linewidth=0.6,
        label="common initial state",
        zorder=6,
    )
    if first_free_crossing is not None:
        ax_global.scatter(
            nature_free[0, first_free_crossing],
            nature_free[1, first_free_crossing],
            nature_free[2, first_free_crossing],
            color=colors["free"],
            marker="X",
            s=75,
            edgecolor="black",
            linewidth=0.5,
            label=f"free crossing, step {first_free_crossing}",
            zorder=7,
        )
        ax_global.scatter(
            nature_control[0, first_free_crossing],
            nature_control[1, first_free_crossing],
            nature_control[2, first_free_crossing],
            color=colors["controlled"],
            marker="X",
            s=75,
            edgecolor="black",
            linewidth=0.5,
            label=f"controlled state, step {first_free_crossing}",
            zorder=7,
        )
    ax_global.text(15.0, 15.0, 38.0, "positive-x lobe", color=colors["controlled"])
    ax_global.text(-17.0, -18.0, 38.0, "negative-x lobe", color=colors["free"])
    ax_global.set_title("Global attractor: free transition versus controlled retention")
    ax_global.legend(loc="upper left", fontsize=8)

    ax_zoom.plot(
        nature_control[0, :stop],
        nature_control[1, :stop],
        nature_control[2, :stop],
        color=colors["controlled"],
        linewidth=2.0,
        label="controlled trajectory",
    )
    ax_zoom.scatter(
        nature_control[0, intervention_steps],
        nature_control[1, intervention_steps],
        nature_control[2, intervention_steps],
        color=colors["intervention"],
        s=10,
        alpha=0.62,
        depthshade=False,
        label="states receiving a z increment",
        zorder=5,
    )

    positive_arrow_steps = arrow_steps[interventions[2, arrow_steps] > 0.0]
    negative_arrow_steps = arrow_steps[interventions[2, arrow_steps] < 0.0]
    for selected_steps, color, label in (
        (positive_arrow_steps, colors["positive_uz"], r"$u_z>0$ direction"),
        (negative_arrow_steps, colors["negative_uz"], r"$u_z<0$ direction"),
    ):
        if selected_steps.size:
            ax_zoom.quiver(
                nature_control[0, selected_steps],
                nature_control[1, selected_steps],
                nature_control[2, selected_steps],
                np.zeros(selected_steps.size),
                np.zeros(selected_steps.size),
                float(arrow_scale) * interventions[2, selected_steps],
                color=color,
                linewidth=1.1,
                arrow_length_ratio=0.35,
                normalize=False,
            )
            # A short dummy 3D line provides a reliable legend handle.
            ax_zoom.plot([], [], [], color=color, linewidth=2.0, label=label)

    ax_zoom.scatter(
        nature_control[0, 0],
        nature_control[1, 0],
        nature_control[2, 0],
        color=colors["start"],
        marker="*",
        s=130,
        edgecolor="black",
        linewidth=0.6,
        label="initial state",
        zorder=7,
    )
    ax_zoom.set_title("z-only intervention locations and direct push directions")
    ax_zoom.legend(
        loc="upper left",
        fontsize=8,
        title=f"Direct $u_z$ arrows shown at {arrow_scale:g}× length",
        title_fontsize=8,
    )

    for ax in (ax_global, ax_zoom):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=24, azim=-58)
        ax.grid(alpha=0.2)

    global_limits = (
        (float(np.min(nature_free[0])), float(np.max(nature_free[0]))),
        (float(np.min(nature_free[1])), float(np.max(nature_free[1]))),
        (float(np.min(nature_free[2])), float(np.max(nature_free[2]))),
    )
    ax_global.set_xlim(global_limits[0])
    ax_global.set_ylim(global_limits[1])
    ax_global.set_zlim(global_limits[2])
    ax_global.set_box_aspect(tuple(high - low for low, high in global_limits))

    controlled_window = nature_control[:, :stop]
    zoom_limits = []
    for component in range(3):
        low = float(np.min(controlled_window[component]))
        high = float(np.max(controlled_window[component]))
        padding = max(0.7, 0.08 * (high - low))
        zoom_limits.append((low - padding, high + padding))
    ax_zoom.set_xlim(zoom_limits[0])
    ax_zoom.set_ylim(zoom_limits[1])
    ax_zoom.set_zlim(zoom_limits[2])
    ax_zoom.set_box_aspect(tuple(high - low for low, high in zoom_limits))

    fig.suptitle(
        "How z-only CSE interventions keep the Lorenz-63 trajectory on the positive lobe",
        fontsize=16,
    )
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / f"{output_stem}.png"
    pdf_path = figures_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    diagnostics = {
        "window": (0, n_steps),
        "first_free_crossing": first_free_crossing,
        "intervention_steps": intervention_steps,
        "arrow_steps": arrow_steps,
        "positive_arrow_count": int(positive_arrow_steps.size),
        "negative_arrow_count": int(negative_arrow_steps.size),
        "arrow_scale": float(arrow_scale),
        "controlled_min_x": float(np.min(nature_control[0, :stop])),
        "free_min_x": float(np.min(nature_free[0, :stop])),
    }
    return fig, (ax_global, ax_zoom), png_path, pdf_path, diagnostics


def plot_3d_z_intervention_response_story(
    data,
    figures_dir,
    *,
    n_steps=400,
    critical_window=(250, 330),
    detail_cycle=0,
    detail_horizon=24,
    response_curve_marker=None,
    response_interventions_as_arrows=False,
    direct_arrow_scale=40.0,
    response_arrow_scale=8.0,
    max_critical_arrows=6,
    show_separation_connectors=False,
    separation_connector_count=6,
    highlight_free_crossing=False,
    simplify_global_comparison=False,
    simplify_critical_comparison=False,
    equal_top_panels=False,
    equal_all_panels=False,
    compact_top_panels=False,
    show_colorbar=True,
    use_signed_direct_arrow_colors=True,
    signed_direct_arrow_legend=False,
    show_arrow_scales_in_legend=True,
    direct_arrow_legend_label=None,
    packet_effect_legend_label=None,
    intervention_state_legend_label="all intervention states",
    global_panel_title=None,
    window_label="Critical window",
    response_panel_title=None,
    response_annotation_xytext=None,
    figure_title=None,
    output_stem="figure_08_cse_z_only_3d_attractor_response_story",
):
    """Combine attractor context, critical-window pushes, and local response.

    This companion to Figure 7 distinguishes the instantaneous z increment
    from the subsequent model-mediated displacement.  It also encodes time on
    the controlled trajectory and provides an unscaled one-packet response
    diagnostic from a common initial state.
    """
    from matplotlib.colors import Normalize
    from class_cse_counterfactual_diagnostics import (
        simulate_counterfactual_ensemble,
    )

    nature_control = np.asarray(data["nature_control"], dtype=float)
    nature_free = np.asarray(data["nature_free"], dtype=float)
    interventions = np.asarray(data["interventions"], dtype=float)
    if nature_control.shape != nature_free.shape:
        raise ValueError("Controlled and free nature trajectories must match.")
    if interventions.shape != nature_control.shape:
        raise ValueError("Interventions must align with the trajectories.")
    if not np.allclose(interventions[:2], 0.0):
        raise ValueError("This diagnostic requires strictly z-only control.")

    Ta = int(data["summary"]["Ta"])
    dt = float(data["summary"]["fcst_dt"])
    n_steps = int(n_steps)
    if n_steps < 1 or n_steps >= nature_control.shape[1]:
        raise ValueError("n_steps must lie inside the saved trajectory.")
    critical_start, critical_stop = (int(value) for value in critical_window)
    if not 0 <= critical_start < critical_stop <= n_steps:
        raise ValueError("critical_window must lie inside 0..n_steps.")

    stop = n_steps + 1
    steps = np.arange(stop)
    norm = Normalize(vmin=0, vmax=n_steps)
    cmap = "viridis"
    colors = {
        "attractor": "0.78",
        "free": "tab:orange",
        "controlled": "tab:blue",
        "intervention": "magenta",
        "positive_uz": "crimson",
        "negative_uz": "purple",
        "packet_effect": "teal",
        "separation": "0.25",
        "start": "tab:green",
        "delta_x": "tab:blue",
        "delta_y": "tab:orange",
        "delta_z": "purple",
    }

    fig = plt.figure(figsize=(20, 12), constrained_layout=True)
    if equal_all_panels:
        grid = fig.add_gridspec(
            2,
            4,
            width_ratios=(1.0, 1.0, 1.0, 1.0),
            height_ratios=(1.0, 0.62),
            wspace=0.0,
        )
        ax_global = fig.add_subplot(grid[0, :2], projection="3d")
        ax_critical = fig.add_subplot(grid[0, 2:], projection="3d")
        ax_response = fig.add_subplot(grid[1, 1:3])
    elif equal_top_panels:
        grid = fig.add_gridspec(
            2,
            2,
            width_ratios=(1.0, 1.0),
            height_ratios=(1.0, 0.52),
        )
        ax_global = fig.add_subplot(grid[0, 0], projection="3d")
        ax_critical = fig.add_subplot(grid[0, 1], projection="3d")
        ax_response = fig.add_subplot(grid[1, :])
    else:
        grid = fig.add_gridspec(
            2,
            2,
            width_ratios=(1.18, 1.0),
            height_ratios=(1.0, 0.62),
        )
        ax_global = fig.add_subplot(grid[:, 0], projection="3d")
        ax_critical = fig.add_subplot(grid[0, 1], projection="3d")
        ax_response = fig.add_subplot(grid[1, 1])

    if compact_top_panels:
        ax_global.set_anchor("E")
        ax_critical.set_anchor("W")

    # Panel A: full attractor and a time-ordered controlled path.
    ax_global.plot(
        nature_free[0],
        nature_free[1],
        nature_free[2],
        color=colors["attractor"],
        linewidth=0.42,
        alpha=0.34,
    )
    ax_global.plot(
        nature_free[0, :stop],
        nature_free[1, :stop],
        nature_free[2, :stop],
        color=colors["free"],
        linewidth=1.7,
        alpha=0.86,
        label="free path",
    )
    initial_sign = np.sign(nature_free[0, 0])
    free_crossings = np.flatnonzero(nature_free[0, :stop] * initial_sign <= 0.0)
    first_free_crossing = int(free_crossings[0]) if free_crossings.size else None
    if simplify_global_comparison:
        ax_global.plot(
            nature_control[0, :stop],
            nature_control[1, :stop],
            nature_control[2, :stop],
            color=colors["controlled"],
            linestyle="--",
            linewidth=2.2,
            alpha=0.95,
            label="controlled path",
            zorder=5,
        )
    else:
        ax_global.plot(
            nature_control[0, :stop],
            nature_control[1, :stop],
            nature_control[2, :stop],
            color="0.35",
            linewidth=0.7,
            alpha=0.35,
        )
        ax_global.scatter(
            nature_control[0, :stop],
            nature_control[1, :stop],
            nature_control[2, :stop],
            c=steps,
            cmap=cmap,
            norm=norm,
            s=8,
            depthshade=False,
            label="controlled path (color = time)",
        )
        ax_global.scatter(
            nature_control[0, 0],
            nature_control[1, 0],
            nature_control[2, 0],
            color=colors["start"],
            marker="*",
            s=140,
            edgecolor="black",
            linewidth=0.6,
            label="common initial state",
            zorder=7,
        )
    if first_free_crossing is not None and not simplify_global_comparison:
        ax_global.scatter(
            nature_free[0, first_free_crossing],
            nature_free[1, first_free_crossing],
            nature_free[2, first_free_crossing],
            color=colors["free"],
            marker="X",
            s=75,
            edgecolor="black",
            linewidth=0.5,
            label=f"free crossing (step {first_free_crossing})",
            zorder=7,
        )
        ax_global.scatter(
            nature_control[0, first_free_crossing],
            nature_control[1, first_free_crossing],
            nature_control[2, first_free_crossing],
            c=[first_free_crossing],
            cmap=cmap,
            norm=norm,
            marker="X",
            s=80,
            edgecolor="black",
            linewidth=0.6,
            label="controlled state at same step",
            zorder=8,
        )
    ax_global.text(15.0, 15.0, 38.0, "positive-x lobe", color="tab:blue")
    ax_global.text(-17.0, -18.0, 38.0, "negative-x lobe", color=colors["free"])
    if global_panel_title is None:
        global_panel_title = "Global attractor and time-ordered controlled path"
    ax_global.set_title(f"A  {global_panel_title}", loc="left")
    ax_global.legend(loc="upper left", fontsize=8)

    # Panel B: critical window with direct z directions and net packet effects.
    critical_slice = slice(critical_start, critical_stop + 1)
    critical_steps = np.arange(critical_start, critical_stop + 1)
    ax_critical.plot(
        nature_free[0, critical_slice],
        nature_free[1, critical_slice],
        nature_free[2, critical_slice],
        color=colors["free"],
        linewidth=2.0,
        label="free path",
    )
    critical_time_scatter = None
    if simplify_critical_comparison:
        ax_critical.plot(
            nature_control[0, critical_slice],
            nature_control[1, critical_slice],
            nature_control[2, critical_slice],
            color=colors["controlled"],
            linestyle="--",
            linewidth=2.0,
            alpha=0.95,
            label="controlled path",
            zorder=3,
        )
    else:
        ax_critical.plot(
            nature_control[0, critical_slice],
            nature_control[1, critical_slice],
            nature_control[2, critical_slice],
            color="0.35",
            linewidth=0.7,
            alpha=0.4,
        )
        critical_time_scatter = ax_critical.scatter(
            nature_control[0, critical_slice],
            nature_control[1, critical_slice],
            nature_control[2, critical_slice],
            c=critical_steps,
            cmap=cmap,
            norm=norm,
            s=13,
            depthshade=False,
            label="controlled path (color = time)",
        )
    intervention_norm = np.linalg.norm(interventions, axis=0)
    critical_interventions = np.flatnonzero(
        (intervention_norm > 0.0)
        & (np.arange(intervention_norm.size) >= critical_start)
        & (np.arange(intervention_norm.size) <= critical_stop)
    )
    ax_critical.scatter(
        nature_control[0, critical_interventions],
        nature_control[1, critical_interventions],
        nature_control[2, critical_interventions],
        color=colors["intervention"],
        s=14,
        alpha=0.5,
        depthshade=False,
        label=intervention_state_legend_label,
        zorder=5,
    )

    separation_norm = np.linalg.norm(nature_control - nature_free, axis=0)
    connector_steps = np.asarray([], dtype=int)
    if show_separation_connectors:
        separation_connector_count = max(2, int(separation_connector_count))
        connector_steps = np.unique(
            np.linspace(
                critical_start,
                critical_stop,
                separation_connector_count,
                dtype=int,
            )
        )
        for connector_step in connector_steps:
            ax_critical.plot(
                [
                    nature_free[0, connector_step],
                    nature_control[0, connector_step],
                ],
                [
                    nature_free[1, connector_step],
                    nature_control[1, connector_step],
                ],
                [
                    nature_free[2, connector_step],
                    nature_control[2, connector_step],
                ],
                color=colors["separation"],
                linestyle="--",
                linewidth=1.0,
                alpha=0.65,
                zorder=2,
            )

    crossing_in_window = (
        first_free_crossing is not None
        and critical_start <= first_free_crossing <= critical_stop
    )
    if highlight_free_crossing and crossing_in_window:
        ax_critical.scatter(
            nature_free[0, first_free_crossing],
            nature_free[1, first_free_crossing],
            nature_free[2, first_free_crossing],
            color=colors["free"],
            marker="X",
            s=90,
            edgecolor="black",
            linewidth=0.6,
            label=f"free enters negative-$x$ regime (step {first_free_crossing})",
            zorder=8,
        )
        ax_critical.scatter(
            nature_control[0, first_free_crossing],
            nature_control[1, first_free_crossing],
            nature_control[2, first_free_crossing],
            c=[first_free_crossing],
            cmap=cmap,
            norm=norm,
            marker="X",
            s=90,
            edgecolor="black",
            linewidth=0.6,
            label="controlled state at the same step",
            zorder=8,
        )

    cycle_starts = np.unique((critical_interventions // Ta) * Ta)
    cycle_starts = cycle_starts[cycle_starts + Ta <= nature_control.shape[1] - 1]
    if cycle_starts.size > int(max_critical_arrows):
        keep = np.linspace(
            0,
            cycle_starts.size - 1,
            int(max_critical_arrows),
            dtype=int,
        )
        cycle_starts = cycle_starts[np.unique(keep)]

    packet_effects = []
    direct_steps = []
    for cycle_start in cycle_starts:
        packet = interventions[:, cycle_start:cycle_start + Ta].T
        nonzero_local = np.flatnonzero(np.linalg.norm(packet, axis=1) > 0.0)
        if nonzero_local.size == 0:
            continue
        direct_step = int(cycle_start + nonzero_local[0])
        direct_steps.append(direct_step)
        uz = float(interventions[2, direct_step])
        if use_signed_direct_arrow_colors:
            direct_color = (
                colors["positive_uz"] if uz > 0.0 else colors["negative_uz"]
            )
        else:
            direct_color = colors["positive_uz"]
        ax_critical.quiver(
            nature_control[0, direct_step],
            nature_control[1, direct_step],
            nature_control[2, direct_step],
            0.0,
            0.0,
            float(direct_arrow_scale) * uz,
            color=direct_color,
            linewidth=1.4,
            arrow_length_ratio=0.32,
            normalize=False,
        )

        local = simulate_counterfactual_ensemble(
            nature_control[:, cycle_start:cycle_start + 1],
            packet,
            position=Ta,
            dt=dt,
        )
        no_packet_end = local[0, 0, :, Ta]
        packet_end = local[1, 0, :, Ta]
        effect = packet_end - no_packet_end
        packet_effects.append(effect)
        ax_critical.plot(
            [no_packet_end[0], packet_end[0]],
            [no_packet_end[1], packet_end[1]],
            [no_packet_end[2], packet_end[2]],
            color=colors["packet_effect"],
            linewidth=1.2,
            alpha=0.9,
        )
        ax_critical.scatter(
            no_packet_end[0],
            no_packet_end[1],
            no_packet_end[2],
            facecolor="none",
            edgecolor=colors["packet_effect"],
            s=28,
            linewidth=0.9,
        )
        ax_critical.quiver(
            no_packet_end[0],
            no_packet_end[1],
            no_packet_end[2],
            *(float(response_arrow_scale) * effect),
            color=colors["packet_effect"],
            linewidth=1.5,
            arrow_length_ratio=0.3,
            normalize=False,
        )

    if signed_direct_arrow_legend:
        ax_critical.plot(
            [], [], [], color=colors["positive_uz"], linewidth=2.0,
            label=r"positive direct increment ($u_z>0$)",
        )
        ax_critical.plot(
            [], [], [], color=colors["negative_uz"], linewidth=2.0,
            label=r"negative direct increment ($u_z<0$)",
        )
    else:
        direct_label = (
            direct_arrow_legend_label
            if direct_arrow_legend_label is not None
            else "direct $u_z$ direction"
        )
        if show_arrow_scales_in_legend:
            direct_label += f" ({direct_arrow_scale:g}×)"
        ax_critical.plot(
            [], [], [], color=colors["positive_uz"], linewidth=2.0,
            label=direct_label,
        )
    packet_effect_label = (
        packet_effect_legend_label
        if packet_effect_legend_label is not None
        else "one-cycle dynamical response"
    )
    if show_arrow_scales_in_legend:
        packet_effect_label += f" ({response_arrow_scale:g}× arrow)"
    ax_critical.plot(
        [], [], [], color=colors["packet_effect"], linewidth=2.0,
        label=packet_effect_label,
    )
    if show_separation_connectors:
        ax_critical.plot(
            [],
            [],
            [],
            color=colors["separation"],
            linestyle="--",
            linewidth=1.2,
            label="free–controlled separation at equal times",
        )
    ax_critical.set_title(
        f"B  {window_label}, steps {critical_start}–{critical_stop}", loc="left"
    )
    ax_critical.legend(loc="upper left", fontsize=7)

    # Panel C: actual-scale causal propagation for the first selected packet.
    detail_start = int(detail_cycle) * Ta
    if detail_start + Ta > nature_control.shape[1]:
        raise ValueError("detail_cycle lies outside the saved trajectory.")
    detail_packet = interventions[:, detail_start:detail_start + Ta].T
    detail = simulate_counterfactual_ensemble(
        nature_control[:, detail_start:detail_start + 1],
        detail_packet,
        position=int(detail_horizon),
        dt=dt,
    )
    local_steps = np.arange(int(detail_horizon) + 1)
    delta = detail[1, 0] - detail[0, 0]
    ax_response.axhline(0.0, color="0.55", linewidth=0.9)
    ax_response.axvspan(1, Ta - 1, color="tab:green", alpha=0.08)
    ax_response.plot(
        local_steps,
        delta[2],
        color=colors["delta_z"],
        linewidth=2.0,
        marker=response_curve_marker,
        markersize=4.0,
        label=r"$\Delta z$ (direct, then dynamic)",
    )
    ax_response.plot(
        local_steps,
        delta[1],
        color=colors["delta_y"],
        linewidth=2.0,
        marker=response_curve_marker,
        markersize=4.0,
        label=r"$\Delta y$ (indirect)",
    )
    ax_response.plot(
        local_steps,
        delta[0],
        color=colors["delta_x"],
        linewidth=2.0,
        marker=response_curve_marker,
        markersize=4.0,
        label=r"$\Delta x$ (indirect)",
    )
    packet_steps = np.flatnonzero(np.linalg.norm(detail_packet, axis=1) > 0.0)
    if packet_steps.size:
        if response_interventions_as_arrows:
            ax_response.quiver(
                packet_steps,
                np.zeros(packet_steps.size),
                np.zeros(packet_steps.size),
                detail_packet[packet_steps, 2],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color=colors["intervention"],
                width=0.0035,
                headwidth=4.5,
                headlength=5.5,
                headaxislength=4.5,
                zorder=6,
            )
            ax_response.plot(
                [],
                [],
                linestyle="none",
                marker=r"$\downarrow$",
                markersize=10,
                color=colors["intervention"],
                label=r"actual $u_z$ increments",
            )
        else:
            ax_response.scatter(
                packet_steps,
                detail_packet[packet_steps, 2],
                color=colors["intervention"],
                marker="v",
                s=30,
                label=r"actual $u_z$ increments",
                zorder=5,
            )
        first_direct = int(packet_steps[0])
        if response_annotation_xytext is None:
            annotation_xytext = (
                first_direct + 2.0,
                delta[2, first_direct] - 0.10,
            )
            annotation_bbox = None
        else:
            annotation_xytext = tuple(response_annotation_xytext)
            annotation_bbox = {
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            }
        ax_response.annotate(
            (
                f"step {first_direct}: $\\Delta z={delta[2, first_direct]:+.2f}$, "
                "$\\Delta y=\\Delta x=0$"
            ),
            xy=(first_direct, delta[2, first_direct]),
            xytext=annotation_xytext,
            arrowprops={"arrowstyle": "->", "color": colors["delta_z"]},
            color=colors["delta_z"],
            fontsize=9,
            bbox=annotation_bbox,
        )
    ax_response.set_xlabel("Steps after the beginning of the selected DA cycle")
    ax_response.set_ylabel("CSE packet minus no-packet state")
    if response_panel_title is None:
        response_panel_title = (
            f"Actual-scale local response to packet in cycle {detail_cycle}"
        )
    ax_response.set_title(f"C  {response_panel_title}", loc="left")
    ax_response.legend(loc="best", fontsize=8, ncols=2)
    ax_response.grid(alpha=0.2)

    for ax in (ax_global, ax_critical):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=24, azim=-58)
        ax.grid(alpha=0.2)

    global_limits = (
        (float(np.min(nature_free[0])), float(np.max(nature_free[0]))),
        (float(np.min(nature_free[1])), float(np.max(nature_free[1]))),
        (float(np.min(nature_free[2])), float(np.max(nature_free[2]))),
    )
    ax_global.set_xlim(global_limits[0])
    ax_global.set_ylim(global_limits[1])
    ax_global.set_zlim(global_limits[2])
    ax_global.set_box_aspect(tuple(high - low for low, high in global_limits))

    critical_points = np.concatenate(
        (
            nature_free[:, critical_slice],
            nature_control[:, critical_slice],
        ),
        axis=1,
    )
    critical_limits = []
    for component in range(3):
        low = float(np.min(critical_points[component]))
        high = float(np.max(critical_points[component]))
        padding = max(0.7, 0.08 * (high - low))
        critical_limits.append((low - padding, high + padding))
    ax_critical.set_xlim(critical_limits[0])
    ax_critical.set_ylim(critical_limits[1])
    ax_critical.set_zlim(critical_limits[2])
    ax_critical.set_box_aspect(
        tuple(high - low for low, high in critical_limits)
    )

    if show_colorbar:
        if critical_time_scatter is None:
            raise ValueError(
                "show_colorbar requires the time-colored critical trajectory."
            )
        colorbar = fig.colorbar(
            critical_time_scatter,
            ax=(ax_global, ax_critical),
            shrink=0.55,
            pad=0.02,
            aspect=35,
        )
        colorbar.set_label("Model step from CSE start")
    if figure_title is None:
        figure_title = (
            "From direct z increments to retention on the positive Lorenz lobe"
        )
    fig.suptitle(figure_title, fontsize=16)

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / f"{output_stem}.png"
    pdf_path = figures_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    packet_effects = np.asarray(packet_effects, dtype=float)
    diagnostics = {
        "window": (0, n_steps),
        "critical_window": (critical_start, critical_stop),
        "first_free_crossing": first_free_crossing,
        "critical_intervention_count": int(critical_interventions.size),
        "critical_direct_arrow_steps": np.asarray(direct_steps, dtype=int),
        "separation_connector_steps": connector_steps,
        "separation_at_window_end": float(separation_norm[critical_stop]),
        "maximum_separation_in_window": float(
            np.max(separation_norm[critical_start:critical_stop + 1])
        ),
        "packet_effects": packet_effects,
        "detail_cycle": int(detail_cycle),
        "detail_start_step": detail_start,
        "detail_packet_z": detail_packet[:, 2].copy(),
        "detail_final_delta": delta[:, -1].copy(),
        "direct_arrow_scale": float(direct_arrow_scale),
        "response_arrow_scale": float(response_arrow_scale),
    }
    return (
        fig,
        (ax_global, ax_critical, ax_response),
        png_path,
        pdf_path,
        diagnostics,
    )
