import pickle

import matplotlib.pyplot as plt
import numpy as np

from class_Lorenz63 import run_RK4


def lorenz63(state, t):
    x, y, z = state
    return np.array([
        10.0 * (y - x),
        x * (28.0 - z) - y,
        x * y - (8.0 / 3.0) * z,
    ])


def build_and_apply_perturbation(x_original, xf_safe_traj, xf_shift_traj,
                                 D, Ta, solver_params, rng=None,
                                 control_indices=None):
    """
    Build and apply Qiwen-style perturbations without modifying external state.

    xf_safe_traj, xf_shift_traj: arrays with shape (xdim, >= Ta), containing
    long forecasts from the current analysis. Following Qiwen cell 13:
      perturbation_all = xf_safe[:, 1:Ta] - xf_shift[:, 1:Ta]
      perturbation_vector = perturbation_all / (amp * column_norm)
    with amp = 1 / D, so every perturbation column has norm D.

    Truth update:
      1. Integrate one free step from x_original.
      2. For q in range(Ta - 1), add perturbation_vector[:, q], record that
         perturbed state, and integrate one step.

    Returns
    -------
    x_final : ndarray, shape (xdim,)
        Final state at t + Ta.
    history : ndarray, shape (Ta, xdim)
        history[0] is x_original, followed by the Ta - 1 perturbed states.
    perturbation_vector : ndarray, shape (xdim, Ta - 1)
        Perturbation columns, each with norm D.
    """
    del rng  # Reserved for future variants; the Qiwen construction is deterministic.

    x_current = np.asarray(x_original, dtype=float).ravel().copy()
    xf_safe_traj = np.asarray(xf_safe_traj, dtype=float)
    xf_shift_traj = np.asarray(xf_shift_traj, dtype=float)

    xdim = x_current.size
    if xf_safe_traj.shape[0] != xdim or xf_shift_traj.shape[0] != xdim:
        raise ValueError('Forecast trajectory state dimension does not match x_original.')
    if xf_safe_traj.shape[1] < Ta or xf_shift_traj.shape[1] < Ta:
        raise ValueError('Forecast trajectories must contain at least Ta columns.')

    perturbation_all = xf_safe_traj[:, 1:Ta] - xf_shift_traj[:, 1:Ta]
    if control_indices is not None:
        control_indices = np.asarray(control_indices, dtype=int)
        if control_indices.size == 0:
            raise ValueError('control_indices must contain at least one state index.')
        if np.any(control_indices < 0) or np.any(control_indices >= xdim):
            raise ValueError('control_indices contains an invalid state index.')
        control_mask = np.zeros(xdim, dtype=bool)
        control_mask[control_indices] = True
        perturbation_all = np.where(control_mask[:, None], perturbation_all, 0.0)

    column_norms = np.linalg.norm(perturbation_all, axis=0)
    if np.any(column_norms == 0):
        raise ValueError('Cannot normalize a zero perturbation column.')

    perturbation_vector = D * perturbation_all / column_norms

    model_func = solver_params.get('model_func', lorenz63)
    sigma = solver_params.get('sigma', 10.0)
    beta = solver_params.get('beta', 8.0 / 3.0)
    rho = solver_params.get('rho', 28.0)
    dt = solver_params.get('dt')
    if dt is None:
        raise ValueError("solver_params must include 'dt'.")

    history = np.zeros((Ta, xdim))
    history[0] = x_current

    one_step = run_RK4(model_func, sigma, beta, rho, x_current, dt, dt).integrate()
    x_current = one_step[1, :].copy()

    for q in range(Ta - 1):
        x_perturbed = x_current + perturbation_vector[:, q]
        history[q + 1] = x_perturbed
        one_step = run_RK4(model_func, sigma, beta, rho, x_perturbed, dt, dt).integrate()
        x_current = one_step[1, :].copy()

    x_final = x_current
    return x_final, history, perturbation_vector


def run_perturbation_unit_test(
    x_nature,
    t0,
    edim,
    D,
    Ta,
    solver_params,
    cycle=0,
    safe_member=0,
    shift_member=None,
    xf_4d_box_history=None,
    Xa_initial=None,
    position=None,
    control_indices=None,
):
    if xf_4d_box_history is not None:
        xf_4d_box = xf_4d_box_history[cycle]
    else:
        if Xa_initial is None or position is None:
            raise ValueError(
                'Provide xf_4d_box_history, or provide both Xa_initial and position.'
            )

        model_func = solver_params.get('model_func', lorenz63)
        sigma = solver_params.get('sigma', 10.0)
        beta = solver_params.get('beta', 8.0 / 3.0)
        rho = solver_params.get('rho', 28.0)
        dt = solver_params.get('dt')
        if dt is None:
            raise ValueError("solver_params must include 'dt'.")

        Xa_initial = np.asarray(Xa_initial)
        xdim = Xa_initial.shape[0]
        xf_4d_box = np.zeros((edim, xdim, position + 1))
        for k in range(edim):
            traj = run_RK4(
                model_func,
                sigma,
                beta,
                rho,
                np.ravel(Xa_initial[:, k]),
                dt,
                dt * position,
            ).integrate()
            xf_4d_box[k] = traj[:position + 1, :].T

    if shift_member is None:
        shift_member = 1 if edim > 1 else 0

    x_original_cycle = np.asarray(x_nature[:, t0]).ravel().copy()

    x_final, perturbation_history, perturbation_vector = build_and_apply_perturbation(
        x_original_cycle,
        xf_4d_box[safe_member],
        xf_4d_box[shift_member],
        D,
        Ta,
        solver_params,
        control_indices=control_indices,
    )

    column_norms = np.linalg.norm(perturbation_vector, axis=0)
    norm_error = np.max(np.abs(column_norms - D)) if column_norms.size else 0.0
    history_initial_error = np.max(np.abs(perturbation_history[0] - x_original_cycle))

    model_func = solver_params.get('model_func', lorenz63)
    sigma = solver_params.get('sigma', 10.0)
    beta = solver_params.get('beta', 8.0 / 3.0)
    rho = solver_params.get('rho', 28.0)
    dt = solver_params.get('dt')
    if dt is None:
        raise ValueError("solver_params must include 'dt'.")

    x_final_free = run_RK4(
        model_func,
        sigma,
        beta,
        rho,
        x_original_cycle,
        dt,
        dt * Ta,
    ).integrate()[Ta, :]

    final_delta = x_final - x_final_free
    final_delta_norm = np.linalg.norm(final_delta)
    final_delta_max_abs = np.max(np.abs(final_delta))

    results = {
        'cycle': cycle,
        'safe_member': safe_member,
        'shift_member': shift_member,
        'x_final': x_final,
        'x_final_free': x_final_free,
        'perturbation_history': perturbation_history,
        'perturbation_vector': perturbation_vector,
        'column_norms': column_norms,
        'norm_error': norm_error,
        'norm_check': norm_error <= 1e-12,
        'history_length_check': len(perturbation_history) == Ta,
        'history_initial_error': history_initial_error,
        'history_initial_check': history_initial_error <= 1e-12,
        'final_delta': final_delta,
        'final_delta_norm': final_delta_norm,
        'final_delta_max_abs': final_delta_max_abs,
        'nonzero_final_difference': final_delta_norm > 0,
    }

    print('test members (safe, shift labels):', safe_member, shift_member)
    print('perturbation column norms:', column_norms)
    print('max |norm - D|:', norm_error)
    print('norm check <= 1e-12:', results['norm_check'])
    print('len(history) == Ta:', results['history_length_check'])
    print('history[0] == x_original:', results['history_initial_check'])
    print('|x_final - x_final_free| max abs:', final_delta_max_abs)
    print('|x_final - x_final_free| norm:', final_delta_norm)
    print('D * Ta:', D * Ta)
    print('nonzero final difference:', results['nonzero_final_difference'])

    return results


def integrate_and_align(candidate_Xa, s, Ta, position, regime_sign=None,
                        solver_params=None):
    """Integrate candidate members and align trajectories exactly as in the visual diagnostic."""
    params = solver_params or {}
    model_func = params.get('model_func', lorenz63)
    sigma = params.get('sigma', 10.0)
    beta = params.get('beta', 8.0 / 3.0)
    rho = params.get('rho', 28.0)
    dt = params.get('dt')
    if dt is None:
        raise ValueError("solver_params must include 'dt'.")

    candidate_Xa = np.asarray(candidate_Xa)
    alignment_start = s * Ta
    extended_steps = alignment_start + position
    t_final = dt * extended_steps + max(abs(dt) * 1e-9, 1e-12)
    aligned_member_data = []

    for k in range(candidate_Xa.shape[1]):
        traj = run_RK4(
            model_func,
            sigma,
            beta,
            rho,
            np.ravel(candidate_Xa[:, k]),
            dt,
            t_final,
        ).integrate().T

        aligned_traj = traj[:, alignment_start:alignment_start + position + 1]
        assert aligned_traj.shape[1] == position + 1, (
            f'aligned trajectory has {aligned_traj.shape[1]} points; '
            f'expected {position + 1}; s={s}; extended_steps={extended_steps}; '
            f'raw trajectory points={traj.shape[1]}'
        )

        aligned_x = aligned_traj[0, :]
        crossings = np.where(aligned_x[:-1] * aligned_x[1:] < 0)[0]
        first_cross = int(crossings[0] + 1) if crossings.size else None
        if regime_sign is None:
            aligned_margin = None
            full_margin = None
            safe = None
        else:
            aligned_margin = float(np.min(regime_sign * aligned_x))
            full_margin = float(np.min(regime_sign * traj[0, :]))
            safe = bool(np.all(regime_sign * aligned_x > 0.0))
        aligned_member_data.append((aligned_traj, first_cross, aligned_margin, full_margin, safe))

    return aligned_member_data


def _get_safe_search_candidate(time, s, Xa_history_control, osse_Xa_history,
                               osse_start_index):
    """Return the candidate ensemble, phase label, and source index for a safe-search depth."""
    if s <= time:
        return Xa_history_control[time - s], 'control', time - s

    osse_idx = osse_start_index - (s - time)
    if osse_idx < 0:
        return None, 'osse', osse_idx
    return osse_Xa_history[osse_idx], 'osse', osse_idx


def plot_safe_search_failure_diagnostic(time, Xa_history_control, osse_Xa_history,
                                        osse_start_index, position, Ta,
                                        regime_sign, solver_params,
                                        max_depth=400, path_figures=None,
                                        x_nature=None, t0=None):
    """Save the automatic visual diagnostic used when no past safe trajectory is found."""
    ncols = 4
    nrows = int(np.ceil(max_depth / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(24, max(5, 3.2 * nrows)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).ravel()
    summary_rows = []

    truth_x = np.asarray([])
    truth_rel = np.asarray([])
    if x_nature is not None and t0 is not None:
        current_abs_time = t0 + time * Ta
        truth_start = current_abs_time
        truth_end = min(x_nature.shape[1], current_abs_time + position + 1)
        truth_rel = np.arange(truth_start, truth_end) - current_abs_time
        truth_x = x_nature[0, truth_start:truth_end]

    for ax_idx, s in enumerate(range(1, max_depth + 1)):
        ax = axes[ax_idx]
        candidate, phase, candidate_idx = _get_safe_search_candidate(
            time, s, Xa_history_control, osse_Xa_history, osse_start_index
        )

        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        if truth_x.size:
            ax.plot(truth_rel, truth_x, color='black', linewidth=1.0, label='free truth')

        if candidate is None:
            ax.set_title(f's={s}, {phase}_idx={candidate_idx}, out of range', fontsize=8)
            summary_rows.append((s, 0, ['out']))
            continue

        aligned_member_data = integrate_and_align(
            candidate,
            s,
            Ta,
            position,
            regime_sign=regime_sign,
            solver_params=solver_params,
        )
        first_crosses = []
        safe_members = []
        rel_steps = np.arange(position + 1)
        for k, (aligned_traj, first_cross, aligned_margin, full_margin, safe) in enumerate(aligned_member_data):
            aligned_x = aligned_traj[0, :]
            if safe:
                safe_members.append(k)
                first_crosses.append(f'safe/{aligned_margin:.3f}')
                ax.plot(rel_steps, aligned_x, color='green', linewidth=2.2, alpha=0.9)
            else:
                label = first_cross if first_cross is not None else f'no-cross/{aligned_margin:.3f}'
                first_crosses.append(label)
                ax.plot(rel_steps, aligned_x, color='lightskyblue', linewidth=1.2, alpha=0.85)

        title_crosses = ', '.join(map(str, first_crosses))
        ax.set_title(
            f's={s}, {phase}_idx={candidate_idx}, n_safe={len(safe_members)}\n'
            f'cross={title_crosses}',
            fontsize=8,
        )
        summary_rows.append((s, len(safe_members), first_crosses))

    for ax in axes[max_depth:]:
        ax.axis('off')

    fig.supxlabel('Steps relative to current cycle')
    fig.supylabel('x')
    fig.suptitle(
        f'Safe-search failure diagnostic: cycle={time}, max_depth={max_depth}, '
        f'regime_sign={regime_sign}',
        fontsize=16,
    )

    output_path = None
    if path_figures is not None:
        output_path = path_figures / f'safe_search_fail_cycle_{time:03d}.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if output_path is not None:
        print(f'Saved safe-search failure diagnostic: {output_path}')
    print('s | n_safe | first aligned crossings by member')
    for s, n_safe, first_crosses in summary_rows:
        print(f'{s:3d} | {n_safe:6d} | {first_crosses}')
    return output_path


def find_past_safe_trajectory(time, Xa_history_control, osse_Xa_history,
                              osse_start_index, position, Ta, regime_sign,
                              solver_params, max_depth=400, path_figures=None,
                              x_nature=None, t0=None,
                              candidate_pool='full',
                              unchanged_members_history=None):
    """
    Find a safe trajectory from past analysis ensembles when p_change == edim.

    Search backward by s = 1, 2, ..., max_depth DA cycles:
      - Phase 1 (s <= time): candidate = Xa_history_control[time - s]
      - Phase 2 (s > time): candidate = osse_Xa_history[osse_start_index -
        (s - time)]; stop if the index is negative.

    For each candidate, use integrate_and_align to reproduce the visual
    diagnostic's integration, orientation, and indexing exactly. A member is
    safe if np.all(regime_sign * aligned_x > 0.0), meaning every point in
    the aligned window belongs to the current regime. first_cross is kept only
    as a diagnostic label.

    Returns
    -------
    aligned_traj : ndarray, shape (xdim, position + 1)
    s : int
    phase : str, either 'control' or 'osse'
    k : int
    aligned_margin : float
        min(regime_sign * aligned_traj[0, :]) over the accepted window.
    full_margin : float
        min(regime_sign * full_traj[0, :]) over the complete extended forecast.
    """
    if regime_sign == 0:
        raise ValueError('regime_sign must be nonzero.')
    if candidate_pool not in {'full', 'unchanged_bkp'}:
        raise ValueError(
            "candidate_pool must be either 'full' or 'unchanged_bkp', "
            f"got {candidate_pool!r}"
        )

    examined_candidates = 0
    for s in range(1, max_depth + 1):
        candidate, phase, candidate_idx = _get_safe_search_candidate(
            time, s, Xa_history_control, osse_Xa_history, osse_start_index
        )
        if candidate is None:
            break

        aligned_member_data = integrate_and_align(
            candidate,
            s,
            Ta,
            position,
            regime_sign=regime_sign,
            solver_params=solver_params,
        )
        if candidate_pool == 'unchanged_bkp' and phase == 'control':
            if unchanged_members_history is None:
                raise ValueError(
                    "unchanged_members_history is required when "
                    "candidate_pool='unchanged_bkp'."
                )
            source_cycle = time - s
            if source_cycle < 0 or source_cycle >= len(unchanged_members_history):
                raise ValueError(
                    f'No unchanged-member history for source_cycle={source_cycle}; '
                    f'time={time}; s={s}.'
                )
            member_indices = [int(k) for k in unchanged_members_history[source_cycle]]
            if not member_indices:
                continue
        else:
            member_indices = list(range(len(aligned_member_data)))

        pool_size = len(member_indices)
        examined_candidates += pool_size
        for k in member_indices:
            aligned_traj, first_cross, aligned_margin, full_margin, safe = aligned_member_data[k]
            assert aligned_traj.shape[1] == position + 1
            if safe:
                if not aligned_margin > 0:
                    plot_safe_search_failure_diagnostic(
                        time=time,
                        Xa_history_control=Xa_history_control,
                        osse_Xa_history=osse_Xa_history,
                        osse_start_index=osse_start_index,
                        position=position,
                        Ta=Ta,
                        regime_sign=regime_sign,
                        solver_params=solver_params,
                        max_depth=max_depth,
                        path_figures=path_figures,
                        x_nature=x_nature,
                        t0=t0,
                    )
                    assert aligned_margin > 0, (
                        f'Candidate marked safe but aligned_margin={aligned_margin}; '
                        f'time={time}; s={s}; phase={phase}; member={k}; '
                        f'first_cross={first_cross}; full_margin={full_margin}'
                    )
                return aligned_traj, s, phase, k, aligned_margin, full_margin, pool_size

    plot_safe_search_failure_diagnostic(
        time=time,
        Xa_history_control=Xa_history_control,
        osse_Xa_history=osse_Xa_history,
        osse_start_index=osse_start_index,
        position=position,
        Ta=Ta,
        regime_sign=regime_sign,
        solver_params=solver_params,
        max_depth=max_depth,
        path_figures=path_figures,
        x_nature=x_nature,
        t0=t0,
    )
    raise RuntimeError(
        f'No past safe trajectory found for cycle time={time} within max_depth={max_depth}; '
        f'examined_candidates={examined_candidates}.'
    )


def run_past_safe_trajectory_test(
    cycle,
    x_nature,
    t0,
    t_start,
    Xa_history_control,
    osse_Xa_history,
    osse_start_index,
    osse_summary,
    xf_4d_box_history,
    position,
    Ta,
    xdim,
    solver_params,
    path_figures,
    max_depth=400,
    expected_phase='osse',
    expected_s=1,
    expected_member=0,
    output_filename='past_safe_trajectory_test_cycle_000.png',
):
    regime_sign = np.sign(x_nature[0, t0])

    safe_aligned_traj, safe_s, safe_phase, safe_member, safe_margin, safe_full_margin, safe_pool_size = (
        find_past_safe_trajectory(
            time=cycle,
            Xa_history_control=Xa_history_control,
            osse_Xa_history=osse_Xa_history,
            osse_start_index=osse_start_index,
            position=position,
            Ta=Ta,
            regime_sign=regime_sign,
            solver_params=solver_params,
            max_depth=max_depth,
            path_figures=path_figures,
            x_nature=x_nature,
            t0=t0,
        )
    )

    safe_aligned_x = safe_aligned_traj[0, :]
    assert safe_margin == float(np.min(regime_sign * safe_aligned_x))

    # Print diagnostic info BEFORE assertions so we see what was found
    print('regime_sign:', regime_sign)
    print('phase:', safe_phase)
    print('s:', safe_s)
    print('member k:', safe_member)
    print('pool_size:', safe_pool_size)
    print('min(regime_sign * aligned_traj_x):', safe_margin)
    print('min(regime_sign * full_traj_x):', safe_full_margin)

    if expected_phase is not None:
        assert safe_phase == expected_phase, f'Expected phase {expected_phase}, got {safe_phase}'
    if expected_s is not None:
        assert safe_s == expected_s, f'Expected s={expected_s}, got s={safe_s}'
    if expected_member is not None:
        assert safe_member == expected_member, f'Expected member {expected_member}, got {safe_member}'
    assert safe_aligned_traj.shape == (xdim, position + 1)

    xf_4d_box = xf_4d_box_history[cycle]
    current_t0 = t0 + cycle * Ta
    alignment_start = safe_s * Ta
    extended_steps = alignment_start + position
    current_relative_steps = np.arange(position + 1)
    past_relative_steps = np.arange(-alignment_start, position + 1)

    if safe_phase == 'control':
        candidate_idx = cycle - safe_s
        candidate_Xa = Xa_history_control[candidate_idx]
        candidate_abs_start = t0 + candidate_idx * Ta
    elif safe_phase == 'osse':
        candidate_idx = osse_start_index - (safe_s - cycle)
        candidate_Xa = osse_Xa_history[candidate_idx]
        candidate_abs_start = osse_summary['t0'] + candidate_idx * Ta
    else:
        raise ValueError(f'Unexpected safe_phase: {safe_phase}')

    safe_full_traj = run_RK4(
        solver_params['model_func'],
        solver_params['sigma'],
        solver_params['beta'],
        solver_params['rho'],
        np.ravel(candidate_Xa[:, safe_member]),
        solver_params['dt'],
        solver_params['dt'] * extended_steps,
    ).integrate().T
    safe_full_x = safe_full_traj[0, :extended_steps + 1]
    np.testing.assert_allclose(
        safe_full_traj[:, alignment_start:alignment_start + position + 1],
        safe_aligned_traj,
    )

    truth_start = current_t0 - alignment_start
    truth_end = current_t0 + position
    truth_window = x_nature[0, truth_start:truth_end + 1]

    fig, ax = plt.subplots(figsize=(18, 6), constrained_layout=True)
    for k in range(xf_4d_box.shape[0]):
        ax.plot(
            current_relative_steps,
            xf_4d_box[k, 0, :],
            color='lightskyblue',
            linewidth=2.0,
            alpha=0.9,
            label='cycle-0 shift forecasts' if k == 0 else None,
        )

    ax.plot(
        past_relative_steps,
        safe_full_x,
        color='green',
        linewidth=2.5,
        label=f'past safe trajectory: s={safe_s} cycles back',
    )
    ax.plot(past_relative_steps, truth_window, color='black', linewidth=2.0, label='truth')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1.0)
    ax.axvline(
        -alignment_start,
        color='green',
        linestyle='--',
        linewidth=2.2,
        label=f'past safe trajectory initial time (-{alignment_start} steps)',
    )
    ax.axvline(0, color='tab:blue', linestyle=':', linewidth=2.2, label='cycle 0 / current CSE cycle')
    ax.axvline(
        t_start - current_t0,
        color='green',
        linestyle=':',
        linewidth=2.0,
        label=f'true event step ({t_start - current_t0})',
    )
    ax.set_xlim(-alignment_start, position)
    ax.set_xlabel('Relative forecast step from current CSE cycle')
    ax.set_ylabel('x')
    ax.set_title(
        f'Past safe trajectory test: phase={safe_phase}, s={safe_s} cycles back, '
        f'member={safe_member}, margin={safe_margin:.6f}, '
        f'past_abs_start={candidate_abs_start}'
    )
    ax.legend(loc='best')

    output_path = path_figures / output_filename
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.show()

    return {
        'fig': fig,
        'ax': ax,
        'output_path': output_path,
        'regime_sign': regime_sign,
        'safe_aligned_traj': safe_aligned_traj,
        'safe_s': safe_s,
        'safe_phase': safe_phase,
        'safe_member': safe_member,
        'safe_margin': safe_margin,
        'safe_full_margin': safe_full_margin,
        'candidate_idx': candidate_idx,
        'candidate_abs_start': candidate_abs_start,
        'alignment_start': alignment_start,
    }


def plot_past_safe_search_visual_diagnostic(
    time,
    x_nature,
    t0,
    Xa_history_control,
    osse_Xa_history,
    osse_start_index,
    osse_summary,
    position,
    Ta,
    solver_params,
    path_figures,
    max_depth=20,
    output_filename='past_safe_search_visual_diagnostic.png',
):
    regime_sign = np.sign(x_nature[0, t0])
    ncols = 4
    nrows = int(np.ceil(max_depth / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(24, max(5, 4 * nrows)),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).ravel()
    summary_rows = []

    for s in range(1, max_depth + 1):
        ax = axes[s - 1]
        alignment_start = s * Ta
        extended_steps = alignment_start + position
        relative_steps = np.arange(extended_steps + 1)

        if s <= time:
            candidate = Xa_history_control[time - s]
            phase = 'control'
            candidate_idx = time - s
            candidate_abs_start = t0 + candidate_idx * Ta
        else:
            osse_idx = osse_start_index - (s - time)
            phase = 'osse'
            candidate_idx = osse_idx
            candidate_abs_start = osse_summary['t0'] + osse_idx * Ta
            if osse_idx < 0:
                ax.set_axis_off()
                summary_rows.append((s, 0, ['no candidate']))
                continue
            candidate = osse_Xa_history[osse_idx]

        truth_start = t0 - alignment_start
        truth_end = t0 + position
        truth_steps = np.arange(extended_steps + 1)
        if truth_start >= 0 and truth_end < x_nature.shape[1]:
            truth_window = x_nature[0, truth_start:truth_end + 1]
            ax.plot(truth_steps, truth_window, color='black', linewidth=1.2, label='truth')

        safe_members = []
        first_cross_by_member = []
        candidate = np.asarray(candidate)

        model_func = solver_params.get('model_func', lorenz63)
        sigma = solver_params.get('sigma', 10.0)
        beta = solver_params.get('beta', 8.0 / 3.0)
        rho = solver_params.get('rho', 28.0)
        dt = solver_params.get('dt')
        if dt is None:
            raise ValueError("solver_params must include 'dt'.")

        for k in range(candidate.shape[1]):
            traj = run_RK4(
                model_func,
                sigma,
                beta,
                rho,
                np.ravel(candidate[:, k]),
                dt,
                dt * extended_steps,
            ).integrate().T

            aligned_traj = traj[:, alignment_start:alignment_start + position + 1]
            aligned_x = aligned_traj[0, :]
            crossings = np.where(aligned_x[:-1] * aligned_x[1:] < 0)[0]
            first_cross = 'safe' if crossings.size == 0 else int(crossings[0] + 1)
            first_cross_by_member.append(first_cross)

            is_safe = bool(np.all(regime_sign * aligned_x > 0))
            if is_safe:
                safe_members.append(k)

            ax.plot(
                relative_steps[:alignment_start + 1],
                traj[0, :alignment_start + 1],
                color='lightgray',
                linewidth=1.0,
                alpha=0.9,
            )
            ax.plot(
                relative_steps[alignment_start:],
                aligned_x,
                color='green' if is_safe else 'tab:blue',
                linewidth=2.6 if is_safe else 1.4,
                alpha=0.95,
            )

        ax.axvline(alignment_start, color='black', linestyle=':', linewidth=1.4, label='now')
        ax.axhline(0, color='gray', linestyle='--', linewidth=1.0)
        ax.set_title(
            f's={s}, {phase}_idx={candidate_idx}, n_safe={len(safe_members)}\n'
            f'cross={first_cross_by_member}',
            fontsize=10,
        )
        ax.set_xlim(0, extended_steps)
        ax.grid(color='0.9', linestyle=':', linewidth=0.7)
        if (s - 1) % ncols == 0:
            ax.set_ylabel('x')
        if s > max_depth - ncols:
            ax.set_xlabel('Relative step from candidate start')

        summary_rows.append((s, len(safe_members), first_cross_by_member))

    for ax in axes[max_depth:]:
        ax.set_axis_off()

    fig.suptitle('Past safe trajectory search diagnostic', fontsize=18)
    output_path = path_figures / output_filename
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.show()

    print('s | n_safe | first aligned crossings by member')
    for s, n_safe, first_cross_by_member in summary_rows:
        print(f'{s:2d} | {n_safe:6d} | {first_cross_by_member}')

    return {
        'fig': fig,
        'axes': axes,
        'summary_rows': summary_rows,
        'output_path': output_path,
        'regime_sign': regime_sign,
    }


def plot_found_past_safe_trajectory(time, xf_4d_box, xf_safe_traj,
                                    safe_s, safe_phase, safe_member,
                                    aligned_margin, full_margin,
                                    t0, Ta, position, x_nature, t_start,
                                    path_figures,
                                    show=False, save_figs=False):
    """Plot the past safe trajectory found during the controlled CSE loop."""
    if not (show or save_figs):
        return None

    current_t0 = t0 + time * Ta
    relative_steps = np.arange(position + 1)
    truth_window = x_nature[0, current_t0:current_t0 + position + 1]

    fig, ax = plt.subplots(figsize=(18, 6), constrained_layout=True)
    for k in range(xf_4d_box.shape[0]):
        ax.plot(
            relative_steps,
            xf_4d_box[k, 0, :],
            color='lightskyblue',
            linewidth=2.0,
            alpha=0.9,
            label='current shift forecasts' if k == 0 else None,
        )

    ax.plot(
        relative_steps,
        xf_safe_traj[0, :],
        color='green',
        linewidth=2.5,
        label='past safe trajectory',
    )
    ax.plot(relative_steps, truth_window, color='black', linewidth=2.0, label='free truth')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1.0)

    relative_event = t_start - current_t0
    if 0 <= relative_event <= position:
        ax.axvline(
            relative_event,
            color='green',
            linestyle=':',
            linewidth=2.0,
            label=f'true event step ({relative_event})',
        )

    ax.set_xlim(0, position)
    ax.set_xlabel('Relative forecast step from current cycle')
    ax.set_ylabel('x')
    ax.set_title(
        f'Past safe trajectory: cycle={time}, phase={safe_phase}, s={safe_s}, '
        f'member={safe_member}, aligned_margin={aligned_margin:.6f}, '
        f'full_margin={full_margin:.6f}'
    )
    ax.legend(loc='best')

    output_path = None
    if save_figs:
        output_path = path_figures / f'past_safe_trajectory_cycle_{time:03d}.png'
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def detect_regime_changes(
    x_nature,
    base_path,
    position=None,
    min_start_time=500,
    min_residence=385,
    output_filename='change_region_time_qiwen.pkl',
    verbose=True,
    event_selection='diego',
):
    if event_selection not in {'diego', 'qiwen'}:
        raise ValueError("event_selection must be either 'diego' or 'qiwen'.")

    x = np.asarray(x_nature[0, :]).ravel()
    crossings = np.where(x[:-1] * x[1:] < 0)[0] + 1  # first step after crossing x=0
    change_region_time = crossings.tolist()

    with open(base_path / output_filename, 'wb') as f:
        pickle.dump(change_region_time, f)

    check_position = position if position is not None else int(np.ceil(75.1 * 4))

    if event_selection == 'diego':
        t_start_series = []
        t_start_residence = []
        for idx, t in enumerate(change_region_time):
            if t < min_start_time:
                continue
            if not x_nature[0, t - 1] > 0:
                continue
            if idx == 0:
                continue
            t_prev = change_region_time[idx - 1]
            residence = t - t_prev
            if residence > min_residence:
                t_start_series.append(t)
                t_start_residence.append(residence)
    else:
        # Intentional reference fidelity: i == 0 uses negative indexing, and
        # both crossing directions are accepted without a polarity filter.
        selected = []
        for i, crossing in enumerate(change_region_time):
            if crossing >= min_start_time and crossing - change_region_time[i - 1] > min_residence:
                selected.append(crossing)
        t_start_series = selected[:100]
        t_start_residence = [
            t - change_region_time[change_region_time.index(t) - 1]
            for t in t_start_series
        ]

    mean_crossing_interval = (
        float(np.mean(np.diff(change_region_time)))
        if len(change_region_time) > 1
        else np.nan
    )

    if verbose:
        print('Total detected crossings:', len(change_region_time))
        print('Mean time between crossings:', mean_crossing_interval)
        print('len(t_start_series):', len(t_start_series))
        print('First 5 t_start_series values:', t_start_series[:5])
        print('First 5 residence times:', t_start_residence[:5])
        print('Sign at t - 1 for first 5:', [
            np.sign(x_nature[0, t - 1])
            for t in t_start_series[:5]
        ])
        print('Sign at t - position for first 5:', [
            np.sign(x_nature[0, t - check_position])
            for t in t_start_series[:5]
        ])

    return {
        'change_region_time': change_region_time,
        'event_selection': event_selection,
        'check_position': check_position,
        't_start_series': t_start_series,
        't_start_residence': t_start_residence,
        'mean_crossing_interval': mean_crossing_interval,
    }


def plot_regime_crossings(
    x_nature,
    t_start_series,
    Ta,
    path_figures,
    position=None,
    output_filename='regime_crossings_qiwen.png',
    event_selection='diego',
):
    if event_selection not in {'diego', 'qiwen'}:
        raise ValueError("event_selection must be either 'diego' or 'qiwen'.")

    plot_position = position if position is not None else int(np.ceil(75.1 * 4))

    def event_color(t):
        if event_selection == 'qiwen' and x_nature[0, t - 1] < 0:
            return 'orange'
        return 'green'

    fig, axes = plt.subplots(2, 1, figsize=(20, 8), constrained_layout=True)

    # Panel 1: wide view
    wide_start = 0
    wide_end = min(20000, x_nature.shape[1])
    wide_steps = np.arange(wide_start, wide_end)

    axes[0].plot(wide_steps, x_nature[0, wide_start:wide_end], label='nature x')
    axes[0].axhline(0, color='gray', linestyle='--')

    for t in t_start_series:
        if t < wide_end:
            axes[0].axvline(t, color=event_color(t), linewidth=2.0)
        control_start = (t - plot_position) - (t - plot_position) % Ta
        if wide_start <= control_start < wide_end:
            axes[0].axvline(control_start, color='blue', linewidth=2.0)

    axes[0].axvline(np.nan, color='green', linewidth=2.0, label='event (+ -> -)')
    if event_selection == 'qiwen':
        axes[0].axvline(np.nan, color='orange', linewidth=2.0, label='event (- -> +)')
    axes[0].axvline(np.nan, color='blue', linewidth=2.0, label='control start')
    axes[0].set_xlim(wide_start, wide_end - 1)
    axes[0].set_title('Regime crossings - wide view')
    axes[0].set_xlabel('Time step')
    axes[0].set_ylabel('x')
    axes[0].legend()

    # Panel 2: zoom around the first accepted t_start
    if not t_start_series:
        raise ValueError('t_start_series is empty; cannot build the zoom panel.')

    first_t_start = t_start_series[0]
    first_control_start = (first_t_start - plot_position) - (first_t_start - plot_position) % Ta
    zoom_start = max(0, first_control_start - 100)
    zoom_end = min(x_nature.shape[1], first_t_start + 150)
    zoom_steps = np.arange(zoom_start, zoom_end)

    axes[1].plot(zoom_steps, x_nature[0, zoom_start:zoom_end], label='nature x')
    axes[1].axhline(0, color='gray', linestyle='--')

    for t in t_start_series:
        if zoom_start <= t < zoom_end:
            axes[1].axvline(t, color=event_color(t), linewidth=2.0)
        control_start = (t - plot_position) - (t - plot_position) % Ta
        if zoom_start <= control_start < zoom_end:
            axes[1].axvline(control_start, color='blue', linewidth=2.0)

    axes[1].axvline(np.nan, color='green', linewidth=2.0, label='event (+ -> -)')
    if event_selection == 'qiwen':
        axes[1].axvline(np.nan, color='orange', linewidth=2.0, label='event (- -> +)')
    axes[1].axvline(np.nan, color='blue', linewidth=2.0, label='control start')
    axes[1].set_xlim(zoom_start, zoom_end - 1)
    axes[1].set_title(f'Zoom around first accepted t_start: {first_t_start}')
    axes[1].set_xlabel('Absolute time step')
    axes[1].set_ylabel('x')
    axes[1].legend()

    output_path = path_figures / output_filename
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.show()

    return fig, axes, output_path


def plot_cse_initialization(
    x_nature,
    Xa_initial,
    t0,
    t_start,
    osse_start_index,
    rmse_init,
    path_figures,
    output_filename='cse_initialization_qiwen.png',
):
    fig, ax = plt.subplots(figsize=(18, 6), constrained_layout=True)

    init_zoom_start = max(0, t0 - 200)
    init_zoom_end = min(x_nature.shape[1], t_start + 100)
    init_zoom_steps = np.arange(init_zoom_start, init_zoom_end)

    ax.plot(init_zoom_steps, x_nature[0, init_zoom_start:init_zoom_end], label='nature x')
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(t_start, color='green', linewidth=2.0, label='event')
    ax.axvline(t0, color='blue', linewidth=2.0, label='CSE start')

    xa_initial_x = np.asarray(Xa_initial[0, :]).ravel()
    ensemble_mean_x = float(np.mean(xa_initial_x))

    ax.scatter(
        np.full(xa_initial_x.shape, t0),
        xa_initial_x,
        marker='o',
        s=80,
        color='red',
        zorder=5,
        label='Xa_initial members',
    )
    ax.scatter(
        [t0],
        [ensemble_mean_x],
        marker='x',
        s=100,
        color='black',
        zorder=6,
        label='ensemble mean',
    )

    ax.set_xlim(init_zoom_start, init_zoom_end - 1)
    ax.set_xlabel('Absolute time step')
    ax.set_ylabel('x')
    ax.set_title(
        f'CSE initialization: t0={t0}, osse_start_index={osse_start_index}, '
        f'rmse_init={rmse_init:.6f}'
    )
    ax.legend()

    output_path = path_figures / output_filename
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.show()

    return fig, ax, output_path


def plot_lorenz_attractor_aligned_cse_starts(
    x_nature,
    t_start_series,
    Ta,
    path_figures,
    position=None,
    n_points=40,
    max_attractor_points=30000,
):
    n_attractor_regime_points = min(n_points, len(t_start_series))
    regime_points_label = f'first_{n_attractor_regime_points}_aligned_cse_starts'

    plot_position = position if position is not None else int(np.ceil(75.1 * 4))
    selected_event_times = np.asarray(
        t_start_series[:n_attractor_regime_points],
        dtype=int,
    )
    selected_cse_start_times = np.asarray([
        (t - plot_position) - (t - plot_position) % Ta
        for t in selected_event_times
    ], dtype=int)

    selected_cse_start_points = x_nature[:, selected_cse_start_times]
    colors_regime_points = plt.cm.viridis(
        np.linspace(0.0, 1.0, n_attractor_regime_points)
    )

    attractor_stride = max(1, int(np.ceil(x_nature.shape[1] / max_attractor_points)))
    attractor = x_nature[:, ::attractor_stride]

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0])

    ax3d = fig.add_subplot(gs[0, :], projection='3d')
    ax3d.scatter(
        attractor[0], attractor[1], attractor[2],
        s=0.7, color='0.55', alpha=0.16, linewidths=0, label='Lorenz attractor'
    )
    ax3d.scatter(
        selected_cse_start_points[0],
        selected_cse_start_points[1],
        selected_cse_start_points[2],
        s=70,
        c=colors_regime_points,
        edgecolors='black',
        linewidths=0.7,
        label='aligned CSE start',
        zorder=5,
    )
    ax3d.set_xlabel('x')
    ax3d.set_ylabel('y')
    ax3d.set_zlabel('z')
    ax3d.set_title(
        f'Lorenz-63 attractor and aligned CSE start points ({regime_points_label})'
    )
    ax3d.legend(loc='upper left')

    projections = [
        (0, 1, 'x', 'y'),
        (0, 2, 'x', 'z'),
        (1, 2, 'y', 'z'),
    ]
    projection_axes = [fig.add_subplot(gs[1, k]) for k in range(3)]
    for ax, (i, j, xlabel, ylabel) in zip(projection_axes, projections):
        ax.scatter(
            attractor[i],
            attractor[j],
            s=0.9,
            color='0.45',
            alpha=0.28,
            linewidths=0,
        )
        ax.scatter(
            selected_cse_start_points[i],
            selected_cse_start_points[j],
            s=48,
            c=colors_regime_points,
            edgecolors='black',
            linewidths=0.55,
            label='aligned CSE start',
            zorder=4,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{xlabel}-{ylabel} projection')
        ax.grid(True, alpha=0.18)

    fig.axes[1].legend(loc='best')
    output_path = (
        path_figures
        / f'qiwen_aligned_cse_start_points_lorenz_attractor_3d_2d_{regime_points_label}.png'
    )
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.show()
    print('Saved Lorenz attractor aligned-start diagnostic:', output_path)

    return fig, (ax3d, *projection_axes), output_path


def plot_filter_diagnostic_during_control(
    evo_history,
    Xa_history_control,
    xf_4d_box_history,
    yo_history,
    branch_history,
    analysis_rmse,
    N_COMPLETED,
    Ta,
    xdim,
    path_figures,
    output_filename='filter_diagnostic_during_control.png',
):
    cycle_axis = np.arange(N_COMPLETED)
    analysis_times = (cycle_axis + 1) * Ta
    filter_plot_steps = np.arange(N_COMPLETED * Ta)

    analysis_x_mean = np.asarray([
        np.asarray(Xa_history_control[i + 1]).mean(axis=1)[0]
        for i in range(N_COMPLETED)
    ])
    prior_members_at_analysis = np.asarray([
        xf_4d_box_history[i][:, :, Ta]
        for i in range(N_COMPLETED)
    ])
    prior_mean = prior_members_at_analysis.mean(axis=1)
    prior_spread = prior_members_at_analysis.std(axis=1).mean(axis=1)
    innovation = np.linalg.norm(np.asarray(yo_history) - prior_mean, axis=1) / np.sqrt(xdim)
    innovation_spread_ratio = np.divide(
        innovation,
        prior_spread,
        out=np.full_like(innovation, np.nan, dtype=float),
        where=prior_spread > 0,
    )

    analysis_rmse = np.asarray(analysis_rmse)
    branch_history_array = np.asarray(branch_history)
    free_mask = branch_history_array == 'free'
    control_mask = branch_history_array != 'free'

    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=False, constrained_layout=True)

    axes[0].plot(
        filter_plot_steps,
        evo_history[0, :N_COMPLETED * Ta],
        color='red',
        linewidth=2.0,
        label='NR controlled',
    )
    axes[0].scatter(
        analysis_times,
        analysis_x_mean,
        color='black',
        s=36,
        zorder=5,
        label='analysis mean x',
    )
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=1.0)
    axes[0].set_title('A - Filter tracking')
    axes[0].set_xlabel('Relative time step from t0')
    axes[0].set_ylabel('x')
    axes[0].legend(loc='best')

    axes[1].plot(cycle_axis, innovation, color='tab:orange', marker='o', label='innovation RMSE')
    axes[1].plot(cycle_axis, prior_spread, color='tab:purple', marker='s', label='prior spread')
    axes[1].set_yscale('log')
    axes[1].set_title('B - Innovation vs prior spread')
    axes[1].set_xlabel('Cycle')
    axes[1].set_ylabel('value')
    axes[1].legend(loc='best')

    control_start = None
    for i, is_control in enumerate(control_mask):
        if is_control and control_start is None:
            control_start = i
        if control_start is not None and (not is_control or i == len(control_mask) - 1):
            control_end = i if is_control else i - 1
            axes[2].axvspan(control_start - 0.5, control_end + 0.5, color='green', alpha=0.15)
            control_start = None

    axes[2].scatter(
        cycle_axis[free_mask],
        analysis_rmse[free_mask],
        color='tab:blue',
        s=42,
        label='free cycles',
        zorder=4,
    )
    axes[2].scatter(
        cycle_axis[control_mask],
        analysis_rmse[control_mask],
        color='green',
        s=42,
        label='control cycles',
        zorder=5,
    )
    axes[2].axhline(0.33, color='black', linestyle='--', linewidth=1.2, label='OSSE benchmark RMSE = 0.33')
    axes[2].set_title('C - Analysis RMSE by branch')
    axes[2].set_xlabel('Cycle')
    axes[2].set_ylabel('analysis RMSE')
    axes[2].legend(loc='best')

    output_path = path_figures / output_filename
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.show()

    def _mean_or_nan(values):
        values = np.asarray(values, dtype=float)
        return float(np.nanmean(values)) if values.size else np.nan

    diagnostics = {
        'free_rmse_mean': _mean_or_nan(analysis_rmse[free_mask]),
        'control_rmse_mean': _mean_or_nan(analysis_rmse[control_mask]),
        'free_innovation_spread_ratio_mean': _mean_or_nan(innovation_spread_ratio[free_mask]),
        'control_innovation_spread_ratio_mean': _mean_or_nan(innovation_spread_ratio[control_mask]),
        'innovation': innovation,
        'prior_spread': prior_spread,
        'innovation_spread_ratio': innovation_spread_ratio,
        'analysis_x_mean': analysis_x_mean,
        'output_path': output_path,
    }

    print('Mean analysis RMSE in free cycles:', diagnostics['free_rmse_mean'])
    print('Mean analysis RMSE in control cycles:', diagnostics['control_rmse_mean'])
    print('Mean innovation/spread ratio in free cycles:', diagnostics['free_innovation_spread_ratio_mean'])
    print('Mean innovation/spread ratio in control cycles:', diagnostics['control_innovation_spread_ratio_mean'])
    print('Saved filter diagnostic to:', output_path)

    return fig, axes, diagnostics


def plot_3d_nature_run_comparison(
    nature_nocontrol,
    nature_control,
    path_figures,
    Ta,
    num_cycles,
    output_filename='NR_3D_comparison_after_control.png',
):
    """Plot 3D free vs controlled Lorenz-63 trajectories."""
    nature_nocontrol = np.asarray(nature_nocontrol)
    nature_control = np.asarray(nature_control)

    if nature_nocontrol.shape[0] != 3 and nature_nocontrol.shape[1] == 3:
        nature_nocontrol = nature_nocontrol.T
    if nature_control.shape[0] == 3:
        nature_control_plot = nature_control.T
    else:
        nature_control_plot = nature_control

    n_steps = num_cycles * Ta + 1
    free_plot = nature_nocontrol[:, :n_steps]
    controlled_plot = nature_control_plot[:n_steps, :]

    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    fig = plt.figure(figsize=(14, 7))

    ax1 = fig.add_subplot(121, projection='3d', facecolor='white')
    ax1.scatter(
        free_plot[0],
        free_plot[1],
        free_plot[2],
        c=np.linspace(0, 1, free_plot.shape[1]),
        cmap='jet',
        s=3,
        alpha=0.9,
    )
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Lorenz-63 No Control', fontsize=16, pad=1)
    ax1.set_xlim(-25, 25)
    ax1.set_ylim(-20, 20)
    ax1.set_zlim(0, 40)
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(
        controlled_plot[:, 0],
        controlled_plot[:, 1],
        controlled_plot[:, 2],
        c=np.linspace(0, 1, controlled_plot.shape[0]),
        cmap='jet',
        s=3,
        alpha=0.9,
    )
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Lorenz-63 Under Control', fontsize=16, pad=1)
    ax2.set_xlim(-25, 25)
    ax2.set_ylim(-20, 20)
    ax2.set_zlim(0, 40)
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    plt.tight_layout(rect=[0, 0, 0.98, 0.9])
    plt.subplots_adjust(wspace=-0.1)
    plt.tight_layout()

    output_path = path_figures / output_filename
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print('Saved 3D nature-run comparison:', output_path)

    return fig, (ax1, ax2), output_path


def run_single_cse(
    event_index,
    D_run,
    seed,
    das,
    x_nature,
    t_start_series,
    osse_Xa_history,
    osse_summary,
    experiment_step,
    position,
    Ta,
    fcst_dt,
    path_figures=None,
    total_events=40,
    regime_reference='nature',
    candidate_pool='full',
    control_indices=None,
):
    if regime_reference not in {'nature', 'member0'}:
        raise ValueError(
            "regime_reference must be either 'nature' or 'member0', "
            f"got {regime_reference!r}"
        )
    if candidate_pool not in {'full', 'unchanged_bkp'}:
        raise ValueError(
            "candidate_pool must be either 'full' or 'unchanged_bkp', "
            f"got {candidate_pool!r}"
        )

    n_test_run = experiment_step
    rng_run = np.random.default_rng(seed)
    obs_error_std_run = np.sqrt(2.0)
    solver_params_run = {
        'model_func': lorenz63,
        'sigma': 10.0,
        'beta': 8.0 / 3.0,
        'rho': 28.0,
        'dt': fcst_dt,
    }

    t_start_run = t_start_series[event_index]
    t0_run = (t_start_run - position) - (t_start_run - position) % Ta
    osse_start_index_run = (t0_run - osse_summary['t0']) // Ta
    if osse_summary['t0'] + osse_start_index_run * Ta != t0_run:
        raise ValueError('t0 is not aligned with the OSSE grid')

    x_original_run = np.asarray(x_nature[:, t0_run]).ravel().copy()
    Xa_run = np.matrix(osse_Xa_history[osse_start_index_run])
    Xa_history_control_run = np.zeros((n_test_run + 1, das.xdim, das.edim))
    Xa_history_control_run[0] = osse_Xa_history[osse_start_index_run]
    evo_history_run = np.zeros((das.xdim, n_test_run * Ta + 1))
    evo_history_run[:, 0] = x_original_run

    analysis_rmse_run = []
    perturbed_time_run = []
    false_alarm_perturbed_time_run = []
    success_perturbed_time_run = []
    branch_counts_run = {'free': 0, 'current_safe_control': 0, 'past_safe_control': 0}
    safe_search_log_run = []
    unchanged_members_history = []

    cant_control_run = False
    failure_cycle_run = None
    evol_run = None
    hidden_forecast_needs_recompute_run = True
    initial_regime_value_run = evo_history_run[0, 0]

    for time_run in range(n_test_run):
        if not np.all(evo_history_run[0, 0:time_run * Ta] * initial_regime_value_run > 0):
            cant_control_run = True
            failure_cycle_run = time_run
            break

        if hidden_forecast_needs_recompute_run:
            evol_run = run_RK4(
                lorenz63,
                10.0,
                8.0 / 3.0,
                28.0,
                x_original_run,
                fcst_dt,
                fcst_dt * position,
            ).integrate()[:position + 1, :].T
        else:
            evol_extension_run = run_RK4(
                lorenz63,
                10.0,
                8.0 / 3.0,
                28.0,
                np.asarray(evol_run[:, -1]).ravel(),
                fcst_dt,
                fcst_dt * Ta,
            ).integrate()
            evol_run = np.concatenate(
                [evol_run[:, Ta:], evol_extension_run[1:Ta + 1, :].T],
                axis=1,
            )

        cycle_perturbed_run = False
        xf_4d_box_run = np.zeros((das.edim, das.xdim, position + 1))
        Xf_run = np.matrix(np.zeros((das.xdim, das.edim)))
        unchange_ens_run = []
        change_ens_run = []

        for k in range(das.edim):
            traj_run = run_RK4(
                lorenz63,
                10.0,
                8.0 / 3.0,
                28.0,
                np.ravel(Xa_run[:, k]),
                fcst_dt,
                fcst_dt * position,
            ).integrate()
            xf_4d_box_run[k] = traj_run[:position + 1, :].T

            if np.all(xf_4d_box_run[k, 0, :] > 0) or np.all(xf_4d_box_run[k, 0, :] < 0):
                unchange_ens_run.append(k)
            else:
                change_ens_run.append(k)

            Xf_run[:, k] = np.matrix(xf_4d_box_run[k, :, Ta]).T

        p_change_run = len(change_ens_run)
        unchanged_members_history.append([int(k) for k in unchange_ens_run])

        if p_change_run == 0:
            branch_run = 'free'
            branch_counts_run[branch_run] += 1
            nature_traj_run = run_RK4(
                lorenz63,
                10.0,
                8.0 / 3.0,
                28.0,
                x_original_run,
                fcst_dt,
                fcst_dt * Ta,
            ).integrate()
            evo_history_run[:, time_run * Ta + 1:(time_run + 1) * Ta + 1] = nature_traj_run[1:Ta + 1, :].T
            x_original_run = nature_traj_run[Ta, :].copy()

        elif p_change_run < das.edim:
            branch_run = 'current_safe_control'
            branch_counts_run[branch_run] += 1
            cycle_perturbed_run = True
            safe_member_run = int(rng_run.choice(unchange_ens_run))
            shift_member_run = int(rng_run.choice(change_ens_run))
            x_final_run, perturbation_history_run, perturbation_vector_run = build_and_apply_perturbation(
                x_original_run,
                xf_4d_box_run[safe_member_run],
                xf_4d_box_run[shift_member_run],
                D_run,
                Ta,
                solver_params_run,
                control_indices=control_indices,
            )
            evo_history_run[:, time_run * Ta:time_run * Ta + Ta] = perturbation_history_run.T
            evo_history_run[:, (time_run + 1) * Ta] = x_final_run
            perturbed_time_run.extend(range(time_run * Ta + 1, time_run * Ta + Ta))
            x_original_run = x_final_run.copy()

        else:
            branch_run = 'past_safe_control'
            branch_counts_run[branch_run] += 1
            cycle_perturbed_run = True
            if regime_reference == 'nature':
                regime_sign_run = np.sign(x_original_run[0])
            elif regime_reference == 'member0':
                regime_sign_run = np.sign(xf_4d_box_run[0, 0, 0])
            xf_safe_traj_run, safe_s_run, safe_phase_run, safe_member_run, aligned_margin_run, full_margin_run, safe_pool_size_run = find_past_safe_trajectory(
                time=time_run,
                Xa_history_control=Xa_history_control_run,
                osse_Xa_history=osse_Xa_history,
                osse_start_index=osse_start_index_run,
                position=position,
                Ta=Ta,
                regime_sign=regime_sign_run,
                solver_params=solver_params_run,
                max_depth=400,
                path_figures=path_figures,
                x_nature=x_nature,
                t0=t0_run,
                candidate_pool=candidate_pool,
                unchanged_members_history=unchanged_members_history,
            )
            shift_member_run = int(rng_run.choice(change_ens_run))
            x_final_run, perturbation_history_run, perturbation_vector_run = build_and_apply_perturbation(
                x_original_run,
                xf_safe_traj_run,
                xf_4d_box_run[shift_member_run],
                D_run,
                Ta,
                solver_params_run,
                control_indices=control_indices,
            )
            evo_history_run[:, time_run * Ta:time_run * Ta + Ta] = perturbation_history_run.T
            evo_history_run[:, (time_run + 1) * Ta] = x_final_run
            perturbed_time_run.extend(range(time_run * Ta + 1, time_run * Ta + Ta))
            x_original_run = x_final_run.copy()
            safe_search_log_run.append({
                'time': time_run,
                's': safe_s_run,
                'phase': safe_phase_run,
                'member': safe_member_run,
                'regime_reference': regime_reference,
                'regime_sign': float(regime_sign_run),
                'candidate_pool': candidate_pool,
                'pool_size': int(safe_pool_size_run),
                'aligned_margin': aligned_margin_run,
                'full_margin': full_margin_run,
            })

        if cycle_perturbed_run:
            cycle_perturbed_steps_run = range(time_run * Ta + 1, time_run * Ta + Ta)
            if np.all(evol_run[0, :] * initial_regime_value_run > 0):
                false_alarm_perturbed_time_run.extend(cycle_perturbed_steps_run)
            else:
                success_perturbed_time_run.extend(cycle_perturbed_steps_run)

        hidden_forecast_needs_recompute_run = cycle_perturbed_run

        yo_run = x_original_run + rng_run.normal(0, obs_error_std_run, das.xdim)
        Xa_run, KH_run = das.compute_analysis(Xf_run, yo_run)
        Xa_history_control_run[time_run + 1] = Xa_run

        xa_mean_run = np.asarray(Xa_run).mean(axis=1)
        analysis_time_index_run = (time_run + 1) * Ta
        controlled_truth_at_analysis_run = evo_history_run[:, analysis_time_index_run]
        analysis_rmse_run.append(
            np.linalg.norm(xa_mean_run - controlled_truth_at_analysis_run) / np.sqrt(das.xdim)
        )

    n_completed_run = sum(branch_counts_run.values())
    total_perturbed_cycles_run = branch_counts_run['current_safe_control'] + branch_counts_run['past_safe_control']
    total_perturbed_steps_run = len(perturbed_time_run)
    total_false_alarm_steps_run = len(false_alarm_perturbed_time_run)
    total_success_justified_steps_run = len(success_perturbed_time_run)
    total_experiment_steps_run = n_test_run * Ta

    success_run = not cant_control_run
    control_ratio_run = total_perturbed_cycles_run / n_test_run
    false_alarm_ratio_run = (
        total_false_alarm_steps_run / total_perturbed_steps_run
        if total_perturbed_steps_run else np.nan
    )
    success_control_ratio_run = total_success_justified_steps_run / total_experiment_steps_run
    analysis_rmse_run = np.asarray(analysis_rmse_run, dtype=float)
    max_s_used_run = max((row['s'] for row in safe_search_log_run), default=None)

    print(
        f"[{event_index + 1}/{total_events}] event={event_index} "
        f"success={success_run} ctrl={control_ratio_run:.4f} fa={false_alarm_ratio_run:.4f}"
    )

    return {
        'event_index': event_index,
        'D': D_run,
        'regime_reference': regime_reference,
        'candidate_pool': candidate_pool,
        'control_indices': (
            None if control_indices is None
            else [int(index) for index in np.asarray(control_indices).ravel()]
        ),
        't_start': t_start_run,
        't0': t0_run,
        'osse_start_index': osse_start_index_run,
        'success': success_run,
        'failure_cycle': failure_cycle_run,
        'control_ratio': control_ratio_run,
        'false_alarm_ratio': false_alarm_ratio_run,
        'success_control_ratio': success_control_ratio_run,
        'branch_counts': branch_counts_run,
        'mean_analysis_rmse': float(np.mean(analysis_rmse_run)) if analysis_rmse_run.size else np.nan,
        'max_analysis_rmse': float(np.max(analysis_rmse_run)) if analysis_rmse_run.size else np.nan,
        'n_safe_searches': len(safe_search_log_run),
        'max_s_used': max_s_used_run,
        'analysis_rmse': analysis_rmse_run,
        'safe_search_log': safe_search_log_run,
        'evo_history': evo_history_run,
    }


def run_single_cse_mini_test(
    run_single_cse_func,
    evo_history,
    Xa_history_control,
    x_original,
    interactive_success=None,
    interactive_control_ratio=None,
):
    evo_before = evo_history.copy()
    xa_hist_before = Xa_history_control.copy()
    x_orig_before = np.asarray(x_original).copy()

    r1 = run_single_cse_func(0, 0.05, seed=1)
    r2 = run_single_cse_func(0, 0.05, seed=1)

    scalar_metric_keys = ['success', 'control_ratio', 'false_alarm_ratio', 'failure_cycle']

    def _same_scalar(a, b):
        if a is None or b is None:
            return a is b
        if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
            return bool(a) == bool(b)
        return bool(np.isclose(a, b, equal_nan=True))

    repro_evo_ok = np.array_equal(r1['evo_history'], r2['evo_history'])
    repro_scalar_checks = {
        key: _same_scalar(r1[key], r2[key])
        for key in scalar_metric_keys
    }
    repro_scalars_ok = all(repro_scalar_checks.values())
    repro_ok = repro_evo_ok and repro_scalars_ok
    print('a) reproducibility evo_history:', repro_evo_ok)
    for key, ok in repro_scalar_checks.items():
        print(f'a) reproducibility {key}:', ok, '| r1 =', r1[key], '| r2 =', r2[key])
    print('a) reproducibility overall:', repro_ok)

    isolation_evo_ok = np.array_equal(evo_history, evo_before)
    isolation_xa_ok = np.array_equal(Xa_history_control, xa_hist_before)
    isolation_x_orig_ok = np.array_equal(np.asarray(x_original), x_orig_before)
    isolation_ok = isolation_evo_ok and isolation_xa_ok and isolation_x_orig_ok
    print('b) isolation evo_history unchanged:', isolation_evo_ok)
    print('b) isolation Xa_history_control unchanged:', isolation_xa_ok)
    print('b) isolation x_original unchanged:', isolation_x_orig_ok)
    print('b) isolation overall:', isolation_ok)

    r3 = run_single_cse_func(0, 0.05, seed=2)
    different_seed_ok = not np.array_equal(r1['evo_history'], r3['evo_history'])
    print('c) different seed gives different evo_history:', different_seed_ok)

    interactive_available = interactive_success is not None and interactive_control_ratio is not None
    print('d) interactive metrics available:', interactive_available)
    if interactive_available:
        success_match = _same_scalar(r1['success'], interactive_success)
        control_ratio_match = _same_scalar(r1['control_ratio'], interactive_control_ratio)
        print('d) success r1 vs interactive:', r1['success'], interactive_success, '| match:', success_match)
        print('d) control_ratio r1 vs interactive:', r1['control_ratio'], interactive_control_ratio, '| match:', control_ratio_match)
    else:
        success_match = None
        control_ratio_match = None
        print('d) skipped side-by-side match because interactive summary variables are missing.')

    if repro_ok and isolation_ok and different_seed_ok:
        print('MINI-TEST OK')
    else:
        raise AssertionError('MINI-TEST FAILED')

    return {
        'r1': r1,
        'r2': r2,
        'r3': r3,
        'repro_ok': repro_ok,
        'repro_evo_ok': repro_evo_ok,
        'repro_scalar_checks': repro_scalar_checks,
        'isolation_ok': isolation_ok,
        'isolation_evo_ok': isolation_evo_ok,
        'isolation_xa_ok': isolation_xa_ok,
        'isolation_x_orig_ok': isolation_x_orig_ok,
        'different_seed_ok': different_seed_ok,
        'interactive_available': interactive_available,
        'success_match': success_match,
        'control_ratio_match': control_ratio_match,
    }


def summarize_cse_batch_results(base_path, D_value, display_table=True):
    summary_file = base_path / f'cse_qiwen_40_results_D{D_value}.pkl'

    with open(summary_file, 'rb') as f:
        summary_results = pickle.load(f)

    summary_rows = [
        {
            'event': row['event_index'],
            'success': row['success'],
            'control_ratio': row['control_ratio'],
            'false_alarm_ratio': row['false_alarm_ratio'],
            'failure_cycle': row['failure_cycle'],
        }
        for row in summary_results
    ]

    summary_table_file = base_path / f'cse_qiwen_40_summary_D{D_value}.csv'

    try:
        import pandas as pd
        summary_table = pd.DataFrame(summary_rows)
        summary_table.to_csv(summary_table_file, index=False)
        if display_table:
            try:
                from IPython.display import display
                display(summary_table)
            except ImportError:
                print(summary_table)
    except ImportError:
        import csv
        summary_table = summary_rows
        with open(summary_table_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        for row in summary_rows:
            print(row)

    print('Saved summary table:', summary_table_file)

    summary_metrics = {
        'success_ratio_global': float(np.mean([row['success'] for row in summary_results])),
        'mean_control_ratio': float(np.nanmean([row['control_ratio'] for row in summary_results])),
        'mean_false_alarm_ratio': float(np.nanmean([row['false_alarm_ratio'] for row in summary_results])),
        'mean_success_control_ratio': float(np.nanmean([row['success_control_ratio'] for row in summary_results])),
    }

    print('Global success ratio:', summary_metrics['success_ratio_global'])
    print('Mean control ratio:', summary_metrics['mean_control_ratio'])
    print('Mean false alarm ratio:', summary_metrics['mean_false_alarm_ratio'])
    print('Mean success control ratio:', summary_metrics['mean_success_control_ratio'])

    return {
        'summary_results': summary_results,
        'summary_rows': summary_rows,
        'summary_table': summary_table,
        'summary_table_file': summary_table_file,
        **summary_metrics,
    }


def summarize_cse_batch_results_by_D(base_path, D_values, display_table=True):
    """Summarize all available 40-event CSE batches for a list of D values."""
    all_results = []
    event_rows = []
    aggregate_rows = []

    for D_value in D_values:
        summary_file = base_path / f'cse_qiwen_40_results_D{D_value}.pkl'
        if not summary_file.exists():
            print('Missing batch file, skipping:', summary_file)
            continue

        with open(summary_file, 'rb') as f:
            summary_results = pickle.load(f)

        all_results.extend(summary_results)
        n_events = len(summary_results)
        for row in summary_results:
            event_rows.append({
                'D': D_value,
                'event': row['event_index'],
                'success': row['success'],
                'control_ratio': row['control_ratio'],
                'false_alarm_ratio': row['false_alarm_ratio'],
                'success_control_ratio': row['success_control_ratio'],
                'failure_cycle': row['failure_cycle'],
            })

        aggregate_rows.append({
            'D': D_value,
            'n_events': n_events,
            'success_ratio': float(np.mean([row['success'] for row in summary_results])),
            'mean_control_ratio': float(np.nanmean([row['control_ratio'] for row in summary_results])),
            'mean_false_alarm_ratio': float(np.nanmean([row['false_alarm_ratio'] for row in summary_results])),
            'mean_success_control_ratio': float(np.nanmean([row['success_control_ratio'] for row in summary_results])),
            'n_failures': int(np.sum([not row['success'] for row in summary_results])),
        })

    if not event_rows:
        raise FileNotFoundError('No CSE batch result pickles were found for the requested D values.')

    event_table_file = base_path / 'cse_qiwen_40_summary_all_D_events.csv'
    aggregate_table_file = base_path / 'cse_qiwen_40_summary_all_D_aggregates.csv'

    try:
        import pandas as pd
        event_table = pd.DataFrame(event_rows)
        aggregate_table = pd.DataFrame(aggregate_rows)
        event_table.to_csv(event_table_file, index=False)
        aggregate_table.to_csv(aggregate_table_file, index=False)
        if display_table:
            try:
                from IPython.display import display
                print('Aggregate metrics by D:')
                display(aggregate_table)
                print('Event-level results:')
                display(event_table)
            except ImportError:
                print('Aggregate metrics by D:')
                print(aggregate_table)
                print('Event-level results:')
                print(event_table)
    except ImportError:
        import csv
        event_table = event_rows
        aggregate_table = aggregate_rows
        with open(event_table_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=event_rows[0].keys())
            writer.writeheader()
            writer.writerows(event_rows)
        with open(aggregate_table_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=aggregate_rows[0].keys())
            writer.writeheader()
            writer.writerows(aggregate_rows)
        print('Aggregate metrics by D:')
        for row in aggregate_rows:
            print(row)
        print('Event-level results:')
        for row in event_rows:
            print(row)

    print('Saved event-level summary table:', event_table_file)
    print('Saved aggregate summary table:', aggregate_table_file)

    return {
        'summary_results': all_results,
        'summary_rows': event_rows,
        'summary_table': event_table,
        'aggregate_rows': aggregate_rows,
        'aggregate_table': aggregate_table,
        'summary_table_file': event_table_file,
        'aggregate_table_file': aggregate_table_file,
    }


def plot_failed_event_diagnostic(
    base_path,
    D_value,
    event_index,
    x_nature,
    experiment_step,
    Ta,
    fcst_dt,
    path_figures,
    output_filename=None,
):
    diagnostic_file = base_path / f'cse_qiwen_40_results_D{D_value}.pkl'
    with open(diagnostic_file, 'rb') as f:
        diagnostic_results = pickle.load(f)

    event_result = next(row for row in diagnostic_results if row['event_index'] == event_index)
    event_evo = np.asarray(event_result['evo_history'])
    event_t0 = event_result['t0']
    event_failure_cycle = event_result['failure_cycle']
    event_failure_step = None if event_failure_cycle is None else event_failure_cycle * Ta

    print('failure_cycle:', event_failure_cycle)
    print('t_start:', event_result['t_start'])
    print('branch_counts:', event_result['branch_counts'])

    perturbed_steps = []
    last_step = min(event_evo.shape[1] - 1, experiment_step * Ta)
    for step in range(1, last_step + 1):
        if not np.all(np.isfinite(event_evo[:, step - 1:step + 1])):
            continue
        free_one_step = run_RK4(
            lorenz63,
            10.0,
            8.0 / 3.0,
            28.0,
            event_evo[:, step - 1],
            fcst_dt,
            fcst_dt,
        ).integrate()[1, :]
        if np.linalg.norm(event_evo[:, step] - free_one_step) > 1e-8:
            perturbed_steps.append(step)
    perturbed_steps = np.asarray(perturbed_steps, dtype=int)

    def _contiguous_segments(indices):
        if len(indices) == 0:
            return []
        breaks = np.where(np.diff(indices) > 1)[0]
        starts = np.r_[indices[0], indices[breaks + 1]]
        ends = np.r_[indices[breaks], indices[-1]]
        return list(zip(starts, ends))

    plot_stop = min(event_evo.shape[1], experiment_step * Ta + 1)
    plot_steps = np.arange(plot_stop)
    free_truth_x = x_nature[0, event_t0:event_t0 + plot_stop]

    fig, ax = plt.subplots(figsize=(18, 6), constrained_layout=True)
    ax.plot(plot_steps, event_evo[0, :plot_stop], color='red', linewidth=2.0, label='NR controlled')
    ax.plot(plot_steps, free_truth_x, color='black', linewidth=1.8, label='NR')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1.0)

    if event_failure_step is not None:
        ax.axvline(
            event_failure_step,
            color='tab:red',
            linestyle=':',
            linewidth=2.0,
            label=f'failure cycle {event_failure_cycle}',
        )

    for seg_idx, (start, end) in enumerate(_contiguous_segments(perturbed_steps)):
        ax.axvspan(
            start - 0.5,
            end + 0.5,
            color='tab:green',
            alpha=0.12,
            label='perturbed interval' if seg_idx == 0 else None,
        )
    if perturbed_steps.size:
        ax.scatter(
            perturbed_steps,
            event_evo[0, perturbed_steps],
            color='tab:green',
            s=10,
            zorder=5,
            label='perturbed steps',
        )

    ax.set_xlabel('Relative time step from t0')
    ax.set_ylabel('x')
    ax.set_title(
        'Failed event diagnostic: '
        f"event={event_index}, success={event_result['success']}, "
        f"ctrl={event_result['control_ratio']:.4f}, "
        f"fa={event_result['false_alarm_ratio']:.4f}, "
        f"success_ctrl={event_result['success_control_ratio']:.4f}"
    )
    ax.legend(loc='best')

    if output_filename is None:
        output_filename = f'failed_event_{event_index}_diagnostic.png'
    output_path = path_figures / output_filename
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.show()

    return {
        'fig': fig,
        'ax': ax,
        'output_path': output_path,
        'event_result': event_result,
        'perturbed_steps': perturbed_steps,
        'failure_cycle': event_failure_cycle,
        'failure_step': event_failure_step,
    }


def plot_cse_cycle_diagnostics(
    cycles_to_plot,
    cycle_t_now_history,
    Xa_cycle_initial_history,
    xf_4d_box_history,
    yo_history,
    change_ens_history,
    unchange_ens_history,
    x_nature,
    t0,
    t_start,
    Ta,
    position,
    edim,
    path_figures,
    show=True,
):
    figures = {}

    for cycle in cycles_to_plot:
        t_now = cycle_t_now_history[cycle]
        obs_time = t_now + Ta
        forecast_end_time = t_now + position
        forecast_steps = np.arange(t_now, forecast_end_time + 1)

        Xa0 = Xa_cycle_initial_history[cycle]
        xf_4d_box = xf_4d_box_history[cycle]
        yo = yo_history[cycle]
        change_ens = change_ens_history[cycle]
        unchange_ens = unchange_ens_history[cycle]

        xlim_start = max(0, min(t0, t_now) - 20)
        xlim_end = min(x_nature.shape[1] - 1, max(t_start, t_now + position) + 20)
        nature_steps = np.arange(xlim_start, xlim_end + 1)

        fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True, constrained_layout=True)

        def draw_base(ax, show_forecast_end=False):
            ax.plot(
                nature_steps,
                x_nature[0, xlim_start:xlim_end + 1],
                color='black',
                label='nature x',
            )
            ax.scatter(
                [obs_time],
                [yo[0]],
                marker='s',
                s=80,
                color='tab:purple',
                zorder=6,
                label=f'observation time ({obs_time})',
            )
            ax.scatter(
                np.full(edim, t_now),
                Xa0[0, :],
                marker='o',
                s=80,
                facecolors='none',
                edgecolors='tab:green',
                linewidths=2,
                zorder=7,
                label='initial ensemble',
            )
            ax.axhline(0, color='gray', linestyle='--', linewidth=1.0)
            ax.axvline(t0, color='blue', linewidth=2.0, label=f'control start time ({t0})')
            ax.axvline(t_start, color='green', linewidth=2.0, label=f'event time ({t_start})')
            ax.axvline(t_now, color='tab:green', linestyle=':', linewidth=1.5)
            ax.axvline(obs_time, color='tab:purple', linestyle=':', linewidth=1.5)
            if show_forecast_end:
                ax.axvline(
                    forecast_end_time,
                    color='green',
                    linestyle='--',
                    linewidth=1.8,
                    label=f'free forecast end ({forecast_end_time})',
                )
            ax.set_xlim(xlim_start, xlim_end)
            ax.set_ylabel('x')

        # Panel 1: nature + obs + initial ensemble
        draw_base(axes[0])
        axes[0].set_title(f'Cycle {cycle}: nature + obs + initial ensemble')
        axes[0].legend(loc='best')

        # Panel 2: free ensemble forecast, all members together
        draw_base(axes[1], show_forecast_end=True)
        for k in range(edim):
            axes[1].plot(
                forecast_steps,
                xf_4d_box[k, 0, :],
                color='tab:orange',
                alpha=0.85,
                linewidth=1.8,
                label='free forecast ensemble' if k == 0 else None,
            )
        axes[1].set_title('Nature + obs + initial ensemble + free forecast')
        axes[1].legend(loc='best')

        # Panel 3: free forecast separated by changed / unchanged
        draw_base(axes[2], show_forecast_end=True)
        for k in unchange_ens:
            axes[2].plot(
                forecast_steps,
                xf_4d_box[k, 0, :],
                color='tab:blue',
                alpha=0.9,
                linewidth=2.0,
                label='unchanged' if k == unchange_ens[0] else None,
            )
        for k in change_ens:
            axes[2].plot(
                forecast_steps,
                xf_4d_box[k, 0, :],
                color='tab:red',
                alpha=0.9,
                linewidth=2.0,
                label='changed' if k == change_ens[0] else None,
            )
        axes[2].set_title(
            f'Qiwen classification: changed={change_ens}, '
            f'unchanged={unchange_ens}, p_change={len(change_ens)}'
        )
        axes[2].set_xlabel('Absolute time step')
        axes[2].legend(loc='best')

        fig.savefig(path_figures / f'cse_cycle_{cycle:03d}_diagnostic.png', dpi=200, bbox_inches='tight')
        figures[cycle] = fig
        if show:
            plt.show()
        else:
            plt.close(fig)

    return figures
