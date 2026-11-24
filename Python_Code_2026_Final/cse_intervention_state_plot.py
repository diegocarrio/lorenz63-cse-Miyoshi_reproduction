"""Retrospective state-versus-intervention diagnostic for z-only CSE runs."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def extract_pre_intervention_records(result, Ta):
    """Pair every applied z push with the controlled truth just before it.

    The CSE saves the state at a push time *after* adding the perturbation.
    Subtracting that saved perturbation reconstructs the pre-push state; the
    free nature trajectory is not the appropriate trajectory here.
    """
    Ta = int(Ta)
    if Ta < 2:
        raise ValueError("Ta must be at least 2 model steps.")

    controlled = np.asarray(result["evo_history"], dtype=float)
    interventions = result["intervention_history"]
    n_completed = int(result["n_completed"])
    if len(interventions) != n_completed:
        raise ValueError("Intervention history does not match completed cycles.")
    if controlled.shape[0] != 3 or controlled.shape[1] < n_completed * Ta + 1:
        raise ValueError("Controlled trajectory has an unexpected shape.")

    records = []
    for cycle, item in enumerate(interventions):
        perturbations = np.asarray(item["perturbations"], dtype=float)
        if perturbations.shape != (Ta, 3):
            raise ValueError(f"Unexpected perturbation shape in cycle {cycle}.")
        if not np.allclose(perturbations[:, :2], 0.0, atol=1e-12):
            raise ValueError("This diagnostic expects z-only interventions.")
        if not np.allclose(perturbations[0], 0.0, atol=1e-12):
            raise ValueError("A push at the initial step breaks the time alignment.")

        for within_cycle in range(1, Ta):
            push = perturbations[within_cycle]
            u_z = float(push[2])
            if np.isclose(u_z, 0.0, atol=1e-14):
                continue
            relative_step = cycle * Ta + within_cycle
            post_state = controlled[:, relative_step]
            pre_state = post_state - push
            records.append({
                "relative_step": relative_step,
                "absolute_step": int(result["t0"]) + relative_step,
                "cycle": cycle,
                "step_within_cycle": within_cycle,
                "branch": item["branch"],
                "u_z": u_z,
                "x_pre": float(pre_state[0]),
                "y_pre": float(pre_state[1]),
                "z_pre": float(pre_state[2]),
                "x_post": float(post_state[0]),
                "y_post": float(post_state[1]),
                "z_post": float(post_state[2]),
            })
    return records


def plot_pre_intervention_state(result, Ta, figures_dir, results_dir):
    """Save a state-conditioned sign figure and its per-push source table."""
    records = extract_pre_intervention_records(result, Ta)
    if not records:
        raise ValueError("No nonzero z interventions are available to plot.")

    figures_dir = Path(figures_dir)
    results_dir = Path(results_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    d_tag = f"{float(result['D']):g}".replace(".", "p")
    stem = (
        "sign_of_uz_vs_pre_intervention_state_z_obs_sign_invariant_"
        f"event_{int(result['event_index'])}_D{d_tag}_qiwenevents"
    )
    # Preserve the earlier x-binned and y-only versions for comparison.
    figure_path = figures_dir / f"{stem}_x_y_low_high.png"
    table_path = results_dir / f"{stem}.csv"

    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    x_pre = np.array([row["x_pre"] for row in records])
    y_pre = np.array([row["y_pre"] for row in records])
    z_pre = np.array([row["z_pre"] for row in records])
    u_z = np.array([row["u_z"] for row in records])
    branches = np.array([row["branch"] for row in records])
    controlled = np.asarray(result["evo_history"], dtype=float)
    n_steps = int(result["n_completed"]) * int(Ta)

    positive_color = "#D55E00"
    negative_color = "#0072B2"
    branch_styles = (
        ("current_safe_control", "o", "current-safe"),
        ("past_safe_control", "^", "past-safe"),
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), layout="constrained")
    fig.suptitle(r"Sign of $u_z$ versus the pre-intervention state", fontsize=16)

    # Split the same pushes separately at the median of x and of y. The bars
    # are descriptive sign frequencies, not success rates or control frequency.
    ax = axes[0]
    median_x = float(np.median(x_pre))
    median_y = float(np.median(y_pre))
    bar_groups = (
        (0.0, "Lower x", x_pre <= median_x),
        (1.0, "Higher x", x_pre > median_x),
        (2.5, "Lower y", y_pre <= median_y),
        (3.5, "Higher y", y_pre > median_y),
    )
    for row, _, subset in bar_groups:
        positive_share = float(np.mean(u_z[subset] > 0))
        negative_share = 1.0 - positive_share
        ax.barh(row, positive_share, height=0.73, color=positive_color)
        ax.barh(row, negative_share, left=positive_share, height=0.73,
                color=negative_color)
        ax.text(positive_share / 2.0, row, f"{positive_share:.0%}\npositive",
                ha="center", va="center", color="white", fontsize=10)
        ax.text(positive_share + negative_share / 2.0, row,
                f"{negative_share:.0%}\nnegative", ha="center", va="center",
                color="white", fontsize=10)
    ax.set_yticks([row for row, _, _ in bar_groups],
                  [label for _, label, _ in bar_groups])
    ax.axhline(1.75, color="0.8", linewidth=0.8)
    ax.invert_yaxis()
    ax.set_xticks([0.0, 0.5, 1.0], ["0%", "50%", "100%"])
    ax.set(xlim=(0.0, 1.0), xlabel="Share of applied pushes",
           title=f"Does x or y separate the signs?\nMedian x={median_x:.2f}; y={median_y:.2f}")

    for ax, ordinate, ordinate_index, ylabel, title in (
        (axes[1], y_pre, 1, r"$y$ before push", "Position in the x–y orbit"),
        (axes[2], z_pre, 2, r"$z$ before push", "Position in the x–z orbit"),
    ):
        ax.plot(controlled[0, :n_steps + 1], controlled[ordinate_index, :n_steps + 1],
                color="0.78", linewidth=0.42, alpha=0.55, rasterized=True,
                zorder=1)
        for branch_name, marker, _ in branch_styles:
            for is_positive, color in ((False, negative_color), (True, positive_color)):
                select = (branches == branch_name) & ((u_z > 0) == is_positive)
                ax.scatter(x_pre[select], ordinate[select], s=13, marker=marker,
                           color=color, alpha=0.45, linewidths=0, rasterized=True,
                           zorder=2)
        ax.set(xlabel=r"$x$ before push", ylabel=ylabel, title=title)

    legend_handles = [
        Line2D([0], [0], marker=marker, linestyle="None", color=color,
               markersize=7, label=f"{branch_label}, $u_z$ {'> 0' if positive else '< 0'}")
        for _, marker, branch_label in branch_styles
        for positive, color in ((True, positive_color), (False, negative_color))
    ]
    fig.legend(handles=legend_handles, ncol=4, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.045), fontsize=9)
    fig.savefig(figure_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return {"figure_path": figure_path, "table_path": table_path,
            "n_pushes": len(records),
            "n_positive": int(np.sum(u_z > 0)),
            "n_negative": int(np.sum(u_z < 0))}


def plot_uz_sign_vs_dxdt_phase(
    intervention_table_path,
    figure_path,
    sigma=10.0,
):
    """Plot push signs against the pre-push x tendency from saved CSE events.

    This reads the CSV written by ``plot_pre_intervention_state``. It does not
    rerun the controller or use the free nature trajectory as a substitute for
    the controlled state immediately before a push.
    """
    intervention_table_path = Path(intervention_table_path)
    figure_path = Path(figure_path)
    if not intervention_table_path.is_file():
        raise FileNotFoundError(
            f"Saved pre-intervention events not found: {intervention_table_path}"
        )
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Lorenz sigma must be finite and positive.")

    with intervention_table_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"x_pre", "y_pre", "u_z", "branch"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(
                f"Intervention table is missing: {sorted(required - set(reader.fieldnames or ()))}"
            )
        records = list(reader)
    if not records:
        raise ValueError("The saved intervention table contains no pushes.")

    x_pre = np.array([float(row["x_pre"]) for row in records])
    y_pre = np.array([float(row["y_pre"]) for row in records])
    u_z = np.array([float(row["u_z"]) for row in records])
    branches = np.array([row["branch"] for row in records])
    if not np.all(np.isfinite(x_pre) & np.isfinite(y_pre) & np.isfinite(u_z)):
        raise ValueError("Pre-intervention states and pushes must be finite.")
    if np.any(u_z == 0):
        raise ValueError("The saved intervention table must contain only applied pushes.")

    dx_dt = float(sigma) * (y_pre - x_pre)
    increasing = dx_dt > 0
    decreasing = dx_dt < 0
    stationary = dx_dt == 0
    push_positive = u_z > 0
    push_negative = u_z < 0
    n_total = len(records)
    n_increasing = int(np.sum(increasing))
    n_decreasing = int(np.sum(decreasing))
    n_stationary = int(np.sum(stationary))
    n_matching = int(np.sum(np.sign(u_z) == np.sign(dx_dt)))

    positive_color = "#D55E00"
    negative_color = "#0072B2"
    branch_styles = (
        ("current_safe_control", "o", "current-safe"),
        ("past_safe_control", "^", "past-safe"),
    )
    unknown_branches = set(branches) - {item[0] for item in branch_styles}
    if unknown_branches:
        raise ValueError(f"Unexpected intervention branches: {sorted(unknown_branches)}")

    fig, axes = plt.subplots(
        1, 2, figsize=(12.8, 5.8),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
        layout="constrained",
    )
    fig.suptitle(r"Sign of $u_z$ versus pre-intervention $\dot{x}$", fontsize=16)

    # Panel 1: the diagonal partitions the orbit by the instantaneous x trend.
    ax = axes[0]
    all_xy = np.concatenate((x_pre, y_pre))
    span = float(np.ptp(all_xy))
    padding = max(0.06 * span, 0.5)
    xy_limits = (float(np.min(all_xy) - padding),
                 float(np.max(all_xy) + padding))
    ax.plot(xy_limits, xy_limits, "k--", linewidth=1.3, zorder=1,
            label=r"$y=x$ ($dx/dt=0$)")
    for branch_name, marker, _ in branch_styles:
        for is_positive, color in ((True, positive_color),
                                   (False, negative_color)):
            select = (branches == branch_name) & (push_positive == is_positive)
            ax.scatter(x_pre[select], y_pre[select], s=14, marker=marker,
                       color=color, alpha=0.5, linewidths=0,
                       rasterized=True, zorder=2)
    ax.set(xlim=xy_limits, ylim=xy_limits, aspect="equal",
           xlabel=r"$x$ before push", ylabel=r"$y$ before push",
           title=r"Pre-intervention $x$–$y$ phase space")
    ax.text(0.03, 0.96, r"$dx/dt>0$", transform=ax.transAxes,
            ha="left", va="top", fontsize=10)
    ax.text(0.97, 0.04, r"$dx/dt<0$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10)

    # Panel 2: proportions are conditional on the x-direction category.
    ax = axes[1]
    direction_groups = (
        ("$dx/dt>0$", increasing),
        ("$dx/dt<0$", decreasing),
    )
    for row, (label, select) in enumerate(direction_groups):
        count = int(np.sum(select))
        if count:
            positive_share = float(np.mean(push_positive[select]))
            negative_share = float(np.mean(push_negative[select]))
            ax.barh(row, positive_share, height=0.58, color=positive_color)
            ax.barh(row, negative_share, left=positive_share, height=0.58,
                    color=negative_color)
            if positive_share >= 0.15:
                ax.text(positive_share / 2, row, f"{positive_share:.1%}",
                        ha="center", va="center", color="white", fontsize=10)
            if negative_share >= 0.15:
                ax.text(positive_share + negative_share / 2, row,
                        f"{negative_share:.1%}", ha="center", va="center",
                        color="white", fontsize=10)
        ax.text(1.02, row, f"n={count:,}", ha="left", va="center", fontsize=10)
    ax.set_yticks([0, 1], [item[0] for item in direction_groups])
    ax.invert_yaxis()
    ax.set(xlim=(0, 1.22), xlabel="Share of applied pushes",
           title="Push sign by direction of x")
    ax.set_xticks([0, 0.5, 1.0], ["0%", "50%", "100%"])
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)

    legend_handles = [
        Line2D([0], [0], marker=marker, linestyle="None", color=color,
               markersize=7, label=f"{branch_label}, $u_z$ {'>0' if positive else '<0'}")
        for _, marker, branch_label in branch_styles
        for positive, color in ((True, positive_color), (False, negative_color))
    ]
    legend_handles.append(Line2D([0], [0], color="black", linestyle="--",
                                 linewidth=1.3, label=r"$y=x$ ($dx/dt=0$)"))
    fig.legend(handles=legend_handles, ncol=3, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.06), fontsize=9)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    if figure_path.name.startswith("sign_of_uz_vs_pre_intervention_state"):
        plt.close(fig)
        raise ValueError("Choose a new filename; the existing figure is protected.")
    fig.savefig(figure_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"Direction-of-x diagnostic (sigma={sigma:g}; n={n_total:,})")
    print(f"{'Category':<14} {'Count':>7} {'% total':>9} {'u_z > 0':>11} {'u_z < 0':>11}")
    for label, select in (("dx/dt > 0", increasing),
                          ("dx/dt < 0", decreasing),
                          ("dx/dt = 0", stationary)):
        count = int(np.sum(select))
        pos_count = int(np.sum(select & push_positive))
        neg_count = int(np.sum(select & push_negative))
        print(f"{label:<14} {count:>7,} {count / n_total:>8.1%} "
              f"{pos_count:>5,} ({pos_count / count:>4.1%}) "
              f"{neg_count:>5,} ({neg_count / count:>4.1%})"
              if count else
              f"{label:<14} {count:>7,} {0.0:>8.1%} {'—':>11} {'—':>11}")
    print(f"sign(u_z) = sign(dx/dt): {n_matching:,}/{n_total:,} "
          f"({n_matching / n_total:.1%})")

    return {
        "figure_path": figure_path,
        "source_path": intervention_table_path,
        "n_total": n_total,
        "n_increasing": n_increasing,
        "n_decreasing": n_decreasing,
        "n_stationary": n_stationary,
        "n_matching": n_matching,
        "fraction_matching": n_matching / n_total,
    }


def plot_signed_push_times_on_free(result, x_nature, Ta, figures_dir):
    """Mark controlled-run push times on the corresponding free trajectory.

    Dots on the free nature run are *time markers* only. The perturbations were
    applied to the controlled run, never to the free reference trajectory.
    """
    records = extract_pre_intervention_records(result, Ta)
    if not records:
        raise ValueError("No nonzero z interventions are available to plot.")

    n_steps = int(result["n_completed"]) * int(Ta)
    t0 = int(result["t0"])
    controlled = np.asarray(result["evo_history"], dtype=float)[:, :n_steps + 1]
    x_nature = np.asarray(x_nature, dtype=float)
    if x_nature.shape[0] != 3 or x_nature.shape[1] < t0 + n_steps + 1:
        raise ValueError("The free nature trajectory does not cover this CSE event.")
    free = x_nature[:, t0:t0 + n_steps + 1]
    time_steps = np.arange(n_steps + 1)
    push_steps = np.array([row["relative_step"] for row in records], dtype=int)
    u_z = np.array([row["u_z"] for row in records], dtype=float)
    positive = u_z > 0
    negative = u_z < 0
    if not np.all(positive | negative):
        raise ValueError("Every recorded push must have a nonzero sign.")

    positive_color = "#D55E00"
    negative_color = "#0072B2"
    controlled_color = "#009E73"
    fig, axes = plt.subplots(
        4, 1, figsize=(14.5, 10), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 0.48]},
    )
    for state_index, state_label in enumerate(("x", "y", "z")):
        ax = axes[state_index]
        ax.plot(time_steps, free[state_index], color="0.50", linewidth=0.95,
                label="free nature run", zorder=1)
        ax.plot(time_steps, controlled[state_index], color=controlled_color,
                linestyle="--", linewidth=1.1,
                label="controlled nature run", zorder=2)
        ax.scatter(push_steps[positive], free[state_index, push_steps[positive]],
                   s=6, marker="o", color=positive_color, alpha=0.76,
                   linewidths=0, rasterized=True, zorder=3)
        ax.scatter(push_steps[negative], free[state_index, push_steps[negative]],
                   s=6, marker="o", color=negative_color, alpha=0.76,
                   linewidths=0, rasterized=True, zorder=3)
        if state_index < 2:
            ax.axhline(0.0, color="0.8", linewidth=0.7)
        ax.set_ylabel(state_label)
        ax.grid(alpha=0.18)

    axes[3].scatter(push_steps[positive], u_z[positive], s=6,
                    color=positive_color, alpha=0.76, linewidths=0,
                    rasterized=True)
    axes[3].scatter(push_steps[negative], u_z[negative], s=6,
                    color=negative_color, alpha=0.76, linewidths=0,
                    rasterized=True)
    axes[3].axhline(0.0, color="0.65", linewidth=0.8)
    axes[3].set(ylabel=r"$u_z$", xlabel="Steps after CSE initialization")
    axes[3].set_ylim(-1.3 * np.max(np.abs(u_z)), 1.3 * np.max(np.abs(u_z)))
    axes[3].grid(alpha=0.18)

    legend_handles = [
        Line2D([0], [0], color="0.50", linewidth=1.2,
               label="free nature run"),
        Line2D([0], [0], color=controlled_color, linestyle="--",
               linewidth=1.2, label="controlled nature run"),
        Line2D([0], [0], marker="o", linestyle="None", color=positive_color,
               markersize=5, label=r"time of $u_z>0$"),
        Line2D([0], [0], marker="o", linestyle="None", color=negative_color,
               markersize=5, label=r"time of $u_z<0$"),
    ]
    fig.suptitle("Free and controlled nature runs with signed z-push times",
                 y=0.99, fontsize=15)
    fig.legend(handles=legend_handles, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 0.956), frameon=False, fontsize=9)
    fig.text(
        0.5, 0.012,
        "Colored dots on the free run mark times only; pushes were applied to the controlled run.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.91))

    d_tag = f"{float(result['D']):g}".replace(".", "p")
    filename = (
        "z_obs_z_control_sign_invariant_free_vs_controlled_signed_pushes_"
        f"event_{int(result['event_index'])}_D{d_tag}_qiwenevents.png"
    )
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / filename
    fig.savefig(figure_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return {"figure_path": figure_path, "n_pushes": len(records),
            "n_positive": int(np.sum(positive)),
            "n_negative": int(np.sum(negative))}
