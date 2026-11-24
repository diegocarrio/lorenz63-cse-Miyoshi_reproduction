import copy
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from class_Lorenz63 import run_RK4
from class_cse import cse


class osse:
    """
    Utilities for standard offline OSSE experiments.

    The class assumes the truth trajectory and synthetic observations already
    exist. The OSSE then slices a fixed window, runs DA cycles with those
    precomputed observations, and advances an ensemble forecast between cycles.
    """

    def __init__(
        self,
        das,
        x_nature,
        y_obs,
        model_func,
        sigma,
        beta,
        rho,
        fcst_dt,
        path_figures='unknown',
        state_labels=None,
    ):
        self.das = das
        self.x_nature = x_nature
        self.y_obs = y_obs
        self.model_func = model_func
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.fcst_dt = fcst_dt
        self.path_figures = path_figures
        self.state_labels = state_labels or ['x', 'y', 'z']
        if self.model_func is None:
            self.model_func = lambda state, t: self.lorenz63(
                state,
                t,
                sigma=self.sigma,
                rho=self.rho,
                beta=self.beta,
            )
        self.results = None
        self.diagnostics = None
        self.plot_data = None

    @staticmethod
    def lorenz63(state, t, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return np.array([dx, dy, dz])

    @staticmethod
    def make_lorenz63_model(sigma=10.0, rho=28.0, beta=8.0 / 3.0):
        return lambda state, t: osse.lorenz63(state, t, sigma=sigma, rho=rho, beta=beta)

    @staticmethod
    def compute_reference_trajectory(model_func, sigma, beta, rho, x0, fcst_dt, forecast_length=80):
        if model_func is None:
            model_func = osse.make_lorenz63_model(sigma=sigma, rho=rho, beta=beta)
        solver = run_RK4(model_func, sigma, beta, rho, x0, fcst_dt, forecast_length)
        return solver.integrate()

    def _prepare_run_state(
        self,
        t0,
        Ta,
        multiple_number,
        T_0,
        multiplicative_inflation,
        das=None,
        seed=None,
    ):
        das_run = self.das if das is None else das
        if seed is not None:
            np.random.seed(seed)

        forecast_position = int(np.ceil(T_0 * multiple_number))
        if forecast_position <= Ta:
            raise ValueError(
                f'forecast_position={forecast_position} must be greater than Ta={Ta} '
                'because the next prior is read at lead time Ta.'
            )
        das_run.multiplicative_inflation = multiplicative_inflation
        das_run.x0 = self.x_nature[:, t0]

        Xa0 = das_run.initEns(
            das_run.x0,
            mu=das_run.ens_bias_init,
            sigma=das_run.ens_sigma_init,
            edim=das_run.edim,
            separate='no',
        )
        return das_run, forecast_position, Xa0

    @staticmethod
    def _progress_settings(num_cycles, progress_every):
        if progress_every is None:
            progress_every = 1000 if num_cycles <= 100000 else 10000
        return int(progress_every)

    @staticmethod
    def _print_cycle_progress(progress_label, completed_cycles, num_cycles, start_time):
        elapsed = time.time() - start_time
        cycles_per_second = completed_cycles / elapsed if elapsed > 0 else float('nan')
        remaining_cycles = num_cycles - completed_cycles
        eta_seconds = remaining_cycles / cycles_per_second if cycles_per_second > 0 else float('nan')
        print(
            f'{progress_label}: cycle {completed_cycles:,}/{num_cycles:,} '
            f'({100.0 * completed_cycles / num_cycles:5.2f}%) | '
            f'{cycles_per_second:,.2f} cycles/s | '
            f'elapsed {elapsed / 60.0:,.1f} min | '
            f'ETA {eta_seconds / 60.0:,.1f} min',
            flush=True,
        )

    def run_standard_osse(
        self,
        t0,
        Ta,
        num_cycles,
        multiple_number,
        T_0,
        multiplicative_inflation,
        das=None,
        seed=None,
        progress_every=None,
        progress_label='OSSE',
    ):
        das_run, forecast_position, Xa0 = self._prepare_run_state(
            t0=t0,
            Ta=Ta,
            multiple_number=multiple_number,
            T_0=T_0,
            multiplicative_inflation=multiplicative_inflation,
            das=das,
            seed=seed,
        )

        truth_time_indices = t0 + np.arange(num_cycles * Ta + 1)
        da_time_indices = t0 + np.arange(num_cycles + 1) * Ta
        self._check_window(truth_time_indices, da_time_indices)

        truth_osse_full = self.x_nature[:, truth_time_indices].copy()
        truth_osse_da = self.x_nature[:, da_time_indices].copy()
        y_obs_osse = self.y_obs[:, da_time_indices].copy()

        Xf = Xa0.copy()
        xf_4d_history = []
        analysis_history = []
        prior_history = []
        osse_system = cse(str_figures_path=self.path_figures, dt=das_run.fcst_dt)
        osse_system.yo_history = [y_obs_osse[:, idx].copy() for idx in range(y_obs_osse.shape[1])]

        start_time = time.time()
        progress_every = self._progress_settings(num_cycles, progress_every)
        if progress_every > 0:
            print(
                f'{progress_label}: starting {num_cycles} DA cycles '
                f'(progress every {progress_every} cycles).',
                flush=True,
            )

        for loop_index in range(num_cycles):
            osse_system.t_index = loop_index * Ta
            prior_history.append(np.asarray(Xf, dtype=float).copy())

            yo_cycle = y_obs_osse[:, loop_index].copy()
            Xa, _ = osse_system.CSE_Step1(Xf, yo_cycle, das_run)
            analysis_history.append(np.asarray(Xa, dtype=float).copy())

            solver = run_RK4(
                self.model_func,
                self.sigma,
                self.beta,
                self.rho,
                np.ravel(Xf[:, 0]),
                self.fcst_dt,
                self.fcst_dt * forecast_position,
            )
            xf_4d = osse_system.CSE_Step2(multiple_number, T_0, solver, Xa, das_run)
            xf_4d_history.append(xf_4d[:, 0, :].copy())
            Xf = xf_4d[:, Ta, :].copy().T
            completed_cycles = loop_index + 1

            if (
                progress_every > 0
                and (completed_cycles == 1 or completed_cycles % progress_every == 0 or completed_cycles == num_cycles)
            ):
                self._print_cycle_progress(progress_label, completed_cycles, num_cycles, start_time)

        results = {
            'initial_ensemble': Xa0,
            'truth_osse_full': truth_osse_full,
            'truth_osse_da': truth_osse_da,
            'y_obs_osse': y_obs_osse,
            'truth_time_indices': truth_time_indices,
            'da_time_indices': da_time_indices,
            'xf_4d_history': xf_4d_history,
            'analysis_history': analysis_history,
            'prior_history': prior_history,
            'forecast_position': forecast_position,
            'multiplicative_inflation': das_run.multiplicative_inflation,
            't0': t0,
            'Ta': Ta,
            'num_cycles': num_cycles,
            'multiple_number': multiple_number,
            'T_0': T_0,
        }
        if das is None:
            self.results = results
            self.diagnostics = None
            self.plot_data = None
        return results

    def run_standard_osse_minimal(
        self,
        t0,
        Ta,
        num_cycles,
        multiple_number,
        T_0,
        multiplicative_inflation,
        analysis_file='osse_Xa_history.pkl',
        prior_file='osse_Xf_history.pkl',
        summary_file='osse_summary.pkl',
        checkpoint_file='osse_Xa_history_checkpoint.npy',
        prior_checkpoint_file='osse_Xf_history_checkpoint.npy',
        checkpoint_summary_file='osse_summary_checkpoint.pkl',
        das=None,
        seed=None,
        progress_every=None,
        checkpoint_every=50000,
        progress_label='OSSE minimal',
        extra_summary=None,
    ):
        """
        Run the OSSE over a long window while storing only analysis ensembles.

        This is the lightweight path needed by CSE_Step4b_previous_analysis:
        it writes Xa history with shape (num_cycles, xdim, edim) plus a small
        summary used to align CSE times with OSSE cycles.
        """
        das_run, forecast_position, Xa0 = self._prepare_run_state(
            t0=t0,
            Ta=Ta,
            multiple_number=multiple_number,
            T_0=T_0,
            multiplicative_inflation=multiplicative_inflation,
            das=das,
            seed=seed,
        )

        final_truth_index = int(t0 + num_cycles * Ta)
        if final_truth_index >= self.x_nature.shape[1]:
            raise ValueError(
                f'OSSE truth window exceeds x_nature length: last index {final_truth_index}, '
                f'available {self.x_nature.shape[1] - 1}.'
            )
        if final_truth_index >= self.y_obs.shape[1]:
            raise ValueError(
                f'OSSE observation window exceeds y_obs length: last index {final_truth_index}, '
                f'available {self.y_obs.shape[1] - 1}.'
            )

        if checkpoint_file:
            analysis_history = np.lib.format.open_memmap(
                checkpoint_file,
                mode='w+',
                dtype=float,
                shape=(num_cycles, das_run.xdim, das_run.edim),
            )
        else:
            analysis_history = np.empty((num_cycles, das_run.xdim, das_run.edim), dtype=float)
        if prior_checkpoint_file:
            prior_history = np.lib.format.open_memmap(
                prior_checkpoint_file,
                mode='w+',
                dtype=float,
                shape=(num_cycles, das_run.xdim, das_run.edim),
            )
        else:
            prior_history = np.empty((num_cycles, das_run.xdim, das_run.edim), dtype=float)

        summary = {
            't0': int(t0),
            'Ta': int(Ta),
            'num_cycles': int(num_cycles),
            'nature_run_last_index': int(self.x_nature.shape[1] - 1),
            'osse_end_index': int(final_truth_index),
            'multiplicative_inflation': float(multiplicative_inflation),
            'T_0': float(T_0),
            'multiple_number': int(multiple_number),
            'forecast_position': int(forecast_position),
            'analysis_history_file': str(analysis_file),
            'prior_history_file': str(prior_file),
            'checkpoint_file': str(checkpoint_file),
            'prior_checkpoint_file': str(prior_checkpoint_file),
            'completed_cycles': 0,
            'experiment_type': 'minimal_offline_OSSE_for_CSE_fallback',
        }
        if extra_summary:
            summary.update(extra_summary)

        Xf = Xa0.copy()
        osse_system = cse(str_figures_path=self.path_figures, dt=das_run.fcst_dt)

        start_time = time.time()
        progress_every = self._progress_settings(num_cycles, progress_every)
        checkpoint_every = int(checkpoint_every)
        if progress_every > 0:
            print(
                f'{progress_label}: starting {num_cycles:,} DA cycles '
                f'(progress every {progress_every:,} cycles).',
                flush=True,
            )

        for loop_index in range(num_cycles):
            osse_system.t_index = loop_index * Ta
            prior_history[loop_index] = np.asarray(Xf, dtype=float)

            yo_cycle = self.y_obs[:, t0 + loop_index * Ta].copy()
            Xa, _ = osse_system.CSE_Step1(Xf, yo_cycle, das_run)
            analysis_history[loop_index] = np.asarray(Xa, dtype=float)

            solver = run_RK4(
                self.model_func,
                self.sigma,
                self.beta,
                self.rho,
                np.ravel(Xf[:, 0]),
                self.fcst_dt,
                self.fcst_dt * Ta,
            )
            xf_4d = osse_system.CSE_Step2_n_steps(Ta + 1, solver, Xa, das_run)
            Xf = xf_4d[:, Ta, :].copy().T
            completed_cycles = loop_index + 1

            should_checkpoint = (
                checkpoint_every > 0
                and (completed_cycles % checkpoint_every == 0 or completed_cycles == num_cycles)
            )
            if should_checkpoint:
                if hasattr(analysis_history, 'flush'):
                    analysis_history.flush()
                if hasattr(prior_history, 'flush'):
                    prior_history.flush()
                summary['completed_cycles'] = int(completed_cycles)
                summary['last_completed_osse_index'] = int(t0 + (completed_cycles - 1) * Ta)
                with open(checkpoint_summary_file, 'wb') as fp:
                    pickle.dump(summary, fp)

            if (
                progress_every > 0
                and (completed_cycles == 1 or completed_cycles % progress_every == 0 or completed_cycles == num_cycles)
            ):
                self._print_cycle_progress(progress_label, completed_cycles, num_cycles, start_time)

        if hasattr(analysis_history, 'flush'):
            analysis_history.flush()
        if hasattr(prior_history, 'flush'):
            prior_history.flush()
        analysis_history_array = np.asarray(analysis_history, dtype=float)
        prior_history_array = np.asarray(prior_history, dtype=float)
        summary['completed_cycles'] = int(num_cycles)
        summary['last_completed_osse_index'] = int(t0 + (num_cycles - 1) * Ta)

        with open(analysis_file, 'wb') as fp:
            pickle.dump(analysis_history_array, fp)
        with open(prior_file, 'wb') as fp:
            pickle.dump(prior_history_array, fp)
        with open(summary_file, 'wb') as fp:
            pickle.dump(summary, fp)

        print('Saved OSSE analysis history to:', analysis_file, flush=True)
        print('Saved OSSE prior history to:', prior_file, flush=True)
        print('Saved OSSE summary to:', summary_file, flush=True)
        print('Analysis history shape:', analysis_history_array.shape, flush=True)
        print('Prior history shape:', prior_history_array.shape, flush=True)
        return {
            'analysis_history': analysis_history_array,
            'prior_history': prior_history_array,
            'summary': summary,
            'forecast_position': forecast_position,
        }

    def run_standard_osse_experiment(
        self,
        t0,
        Ta=8,
        default_num_cycles=1000,
        run_full_nature_period=True,
        run_minimal_output=True,
        progress_every=10000,
        checkpoint_every=50000,
        multiple_number=4,
        T_0=75.1,
        multiplicative_inflation=1.05,
        analysis_file='osse_Xa_history.pkl',
        prior_file='osse_Xf_history.pkl',
        summary_file='osse_summary.pkl',
        checkpoint_file='osse_Xa_history_checkpoint.npy',
        prior_checkpoint_file='osse_Xf_history_checkpoint.npy',
        checkpoint_summary_file='osse_summary_checkpoint.pkl',
        extra_summary=None,
    ):
        """
        Configure and run the standard OSSE notebook workflow.

        Returns a dictionary intended to update the notebook namespace.
        """
        extra_summary = {} if extra_summary is None else dict(extra_summary)

        print(' * * * Starting the Observing System Simulation Experiment * * *')
        print(f'       We choose t = {t0} as the starting time for the OSSE.')

        nature_run_last_index = int(self.x_nature.shape[1] - 1)
        full_nature_period_cycles_from_osse_t0 = int((nature_run_last_index - t0) // Ta)
        num_cycles = int(default_num_cycles)

        if run_full_nature_period:
            num_cycles = full_nature_period_cycles_from_osse_t0
            print('Nature Run last index:', nature_run_last_index)
            print('OSSE t0:', t0)
            print('OSSE cycles to cover Nature Run through its available end:', num_cycles)
            print('OSSE end index:', t0 + num_cycles * Ta)
        else:
            print('Using default OSSE num_cycles:', num_cycles)

        common = {
            'Ta': int(Ta),
            'DEFAULT_OSSE_NUM_CYCLES': int(default_num_cycles),
            'RUN_OSSE_FULL_NATURE_PERIOD': bool(run_full_nature_period),
            'RUN_OSSE_MINIMAL_OUTPUT': bool(run_minimal_output),
            'OSSE_PROGRESS_EVERY': int(progress_every),
            'OSSE_CHECKPOINT_EVERY': int(checkpoint_every),
            'multiple_number': int(multiple_number),
            'T_0': float(T_0),
            'MULTIPLICATIVE_INFLATION': float(multiplicative_inflation),
            'nature_run_last_index': int(nature_run_last_index),
            'full_nature_period_cycles_from_osse_t0': int(full_nature_period_cycles_from_osse_t0),
            'num_cycles': int(num_cycles),
        }

        if run_minimal_output:
            minimal_summary = {
                **extra_summary,
                'run_osse_full_nature_period': bool(run_full_nature_period),
                'full_nature_period_cycles_from_osse_t0': int(full_nature_period_cycles_from_osse_t0),
            }
            osse_results = self.run_standard_osse_minimal(
                t0=t0,
                Ta=Ta,
                num_cycles=num_cycles,
                multiple_number=multiple_number,
                T_0=T_0,
                multiplicative_inflation=multiplicative_inflation,
                analysis_file=analysis_file,
                prior_file=prior_file,
                summary_file=summary_file,
                checkpoint_file=checkpoint_file,
                prior_checkpoint_file=prior_checkpoint_file,
                checkpoint_summary_file=checkpoint_summary_file,
                progress_every=progress_every,
                checkpoint_every=checkpoint_every,
                extra_summary=minimal_summary,
            )
            analysis_history = osse_results['analysis_history']
            prior_history = osse_results['prior_history']
            forecast_position = osse_results['forecast_position']

            result = {
                **common,
                'osse_results': osse_results,
                'analysis_history': analysis_history,
                'prior_history': prior_history,
                'forecast_position': forecast_position,
                'diagnostics': {},
                'plot_data': {},
                'RUN_OSSE_PLOTS': False,
                'RUN_INFLATION_SWEEP': False,
                'Xa': analysis_history[0].T,
                'truth_osse_full': np.empty((self.das.xdim, 0)),
                'truth_osse_da': np.empty((self.das.xdim, 0)),
                'y_obs_osse': np.empty((self.das.ydim, 0)),
                'truth_time_indices': np.array([t0, t0 + num_cycles * Ta]),
                'da_time_indices': t0 + np.arange(num_cycles + 1) * Ta,
                'xf_4d_history': [],
            }
            print('Minimal OSSE complete. CSE needs osse_Xa_history.pkl and osse_summary.pkl.')
            print('Minimal OSSE also saved osse_Xf_history.pkl.')
            return result

        osse_results = self.run_standard_osse(
            t0=t0,
            Ta=Ta,
            num_cycles=num_cycles,
            multiple_number=multiple_number,
            T_0=T_0,
            multiplicative_inflation=multiplicative_inflation,
            progress_every=progress_every,
        )

        diagnostics, plot_data = self.compute_diagnostics(osse_results)
        result = {
            **common,
            'osse_results': osse_results,
            'Xa': osse_results['initial_ensemble'],
            'truth_osse_full': osse_results['truth_osse_full'],
            'truth_osse_da': osse_results['truth_osse_da'],
            'y_obs_osse': osse_results['y_obs_osse'],
            'truth_time_indices': osse_results['truth_time_indices'],
            'da_time_indices': osse_results['da_time_indices'],
            'xf_4d_history': osse_results['xf_4d_history'],
            'analysis_history': osse_results['analysis_history'],
            'prior_history': osse_results['prior_history'],
            'forecast_position': osse_results['forecast_position'],
            'diagnostics': diagnostics,
            'plot_data': plot_data,
            'rmse_analysis_total': diagnostics['rmse_analysis_total'],
            'analysis_spread_total': diagnostics['analysis_spread_total'],
            'osse_time_steps': plot_data['osse_time_steps'],
            'osse_obs_time_steps': plot_data['osse_obs_time_steps'],
            'analysis_time_steps': plot_data['analysis_time_steps'],
            'analysis_mean_array': plot_data['analysis_mean_array'],
            'prior_mean_array': plot_data['prior_mean_array'],
            'state_labels': self.state_labels,
        }

        print(
            f'ens_bias_init: {self.das.ens_bias_init}, sigma: {self.das.ens_sigma_init}, '
            f'ensemble members: {self.das.edim}'
        )
        print(f'Multiplicative inflation: {self.das.multiplicative_inflation}')
        print(f'OSSE starts at absolute t0 = {t0}')
        print(f'OSSE truth window: {result["truth_time_indices"][0]} to {result["truth_time_indices"][-1]}')
        print(f'OSSE DA observation times: {result["da_time_indices"][0]} to {result["da_time_indices"][-1]} every Ta={Ta}')
        print(f'truth_osse_full shape: {result["truth_osse_full"].shape}')
        print(f'y_obs_osse shape: {result["y_obs_osse"].shape}')
        print('Analysis total RMSE:', result['rmse_analysis_total'])
        print('Analysis total spread:', result['analysis_spread_total'])
        return result

    @staticmethod
    def publish_run_outputs(osse_run, namespace, include_full_diagnostics=None):
        """
        Publish selected OSSE run outputs into a notebook namespace.

        This keeps notebooks concise while avoiding an unrestricted
        globals().update(osse_run).
        """
        keys = [
            'num_cycles',
            'nature_run_last_index',
            'full_nature_period_cycles_from_osse_t0',
            'forecast_position',
            'osse_results',
            'analysis_history',
            'prior_history',
            'diagnostics',
            'plot_data',
            'Xa',
            'truth_osse_full',
            'truth_osse_da',
            'y_obs_osse',
            'truth_time_indices',
            'da_time_indices',
            'xf_4d_history',
        ]

        if include_full_diagnostics is None:
            include_full_diagnostics = 'rmse_analysis_total' in osse_run
        if include_full_diagnostics:
            keys.extend([
                'rmse_analysis_total',
                'analysis_spread_total',
                'osse_time_steps',
                'osse_obs_time_steps',
                'analysis_time_steps',
                'analysis_mean_array',
                'prior_mean_array',
                'state_labels',
            ])

        for key in keys:
            if key in osse_run:
                namespace[key] = osse_run[key]

        namespace['RUN_OSSE_PLOTS'] = osse_run.get(
            'RUN_OSSE_PLOTS',
            namespace.get('RUN_OSSE_PLOTS', osse_run['num_cycles'] <= 5000),
        )
        namespace['RUN_INFLATION_SWEEP'] = osse_run.get(
            'RUN_INFLATION_SWEEP',
            namespace.get('RUN_INFLATION_SWEEP', False),
        )
        return namespace

    def compute_diagnostics(self, results=None):
        results = self._get_results(results)
        analysis_mean_array, prior_mean_array = self.compute_means(results)

        truth_da_for_cycles = results['truth_osse_da'][:, :-1]
        analysis_error = analysis_mean_array - truth_da_for_cycles
        rmse_analysis_total = np.sqrt(np.mean(analysis_error**2))

        analysis_array = np.asarray(
            [np.asarray(xa_cycle, dtype=float) for xa_cycle in results['analysis_history']]
        )
        analysis_spread_by_cycle_state = np.std(analysis_array, axis=2, ddof=1)
        analysis_spread_total = np.sqrt(np.mean(analysis_spread_by_cycle_state**2))

        diagnostics = {
            'rmse_analysis_total': float(rmse_analysis_total),
            'analysis_spread_total': float(analysis_spread_total),
        }
        plot_data = {
            'osse_time_steps': np.arange(results['truth_osse_full'].shape[1]),
            'osse_obs_time_steps': np.arange(0, (results['num_cycles'] + 1) * results['Ta'], results['Ta']),
            'analysis_time_steps': np.arange(0, results['num_cycles'] * results['Ta'], results['Ta']),
            'analysis_mean_array': analysis_mean_array,
            'prior_mean_array': prior_mean_array,
        }

        if results is self.results:
            self.diagnostics = diagnostics
            self.plot_data = plot_data
        return diagnostics, plot_data

    def compute_inflation_sweep_diagnostics(self, results, spinup_exclude=0):
        results = self._get_results(results)
        spinup_exclude = int(spinup_exclude)
        if spinup_exclude < 0:
            raise ValueError('spinup_exclude must be non-negative.')

        analysis_array = np.asarray(
            [np.asarray(xa_cycle, dtype=float) for xa_cycle in results['analysis_history']]
        )
        n_cycles = analysis_array.shape[0]
        if spinup_exclude >= n_cycles:
            raise ValueError(
                f'spinup_exclude={spinup_exclude} must be smaller than n_cycles={n_cycles}.'
            )

        analysis_mean_array = analysis_array.mean(axis=2)
        truth_da_for_cycles = results['truth_osse_da'][:, :n_cycles].T
        post_analysis_mean = analysis_mean_array[spinup_exclude:]
        post_truth = truth_da_for_cycles[spinup_exclude:]

        instantaneous_rmse = np.sqrt(np.mean((post_analysis_mean - post_truth) ** 2, axis=1))
        rmse_analysis_total = float(np.sqrt(np.mean((post_analysis_mean - post_truth) ** 2)))

        post_analysis_spread_by_cycle_state = np.std(
            analysis_array[spinup_exclude:],
            axis=2,
            ddof=1,
        )
        analysis_spread_total = float(np.sqrt(np.mean(post_analysis_spread_by_cycle_state ** 2)))

        median_rmse = float(np.median(instantaneous_rmse))
        if median_rmse > 0:
            n_outliers = int(np.sum(instantaneous_rmse > 3.0 * median_rmse))
        else:
            n_outliers = int(np.sum(instantaneous_rmse > 0.0))
        spread_rmse_ratio = (
            float(analysis_spread_total / rmse_analysis_total)
            if rmse_analysis_total > 0
            else float('nan')
        )

        return {
            'rmse_analysis_total': rmse_analysis_total,
            'analysis_spread_total': analysis_spread_total,
            'spread_rmse_ratio': spread_rmse_ratio,
            'n_outliers': n_outliers,
            'spinup_exclude': spinup_exclude,
        }

    def compute_means(self, results=None):
        results = self._get_results(results)
        analysis_mean_array = np.asarray(
            [np.asarray(xa_cycle, dtype=float).mean(axis=1) for xa_cycle in results['analysis_history']]
        ).T
        prior_mean_array = np.asarray(
            [np.asarray(xf_cycle, dtype=float).mean(axis=1) for xf_cycle in results['prior_history']]
        ).T
        return analysis_mean_array, prior_mean_array

    def plot_truth_observations_means(
        self,
        results=None,
        diagnostics=None,
        plot_data=None,
        zoom_xlim=None,
        figsize=(12, 8),
        legend_outside=False,
        title=None,
        observation_size=11,
        analysis_linewidth=2.0,
        prior_linewidth=1.4,
        truth_linewidth=1.5,
        truth_color='black',
        observation_color='0.55',
        analysis_color='#0072B2',
        prior_color='#D55E00',
        analysis_linestyle='-',
        prior_linestyle='--',
        legend_fontsize=8,
        legend_loc='upper right',
        legend_mode='axes_reserved',
        legend_ncol=2,
        legend_reserved_fraction=0.35,
        highlight_mean_difference=False,
        mean_difference_alpha=0.18,
        ylim_from_means=False,
        mean_ylim_padding=0.12,
    ):
        results = self._get_results(results)
        if diagnostics is None or plot_data is None:
            diagnostics, plot_data = self.compute_diagnostics(results)

        fig, axes = plt.subplots(len(self.state_labels), 1, figsize=figsize, sharex=True)
        axes = np.atleast_1d(axes)
        for state_idx, ax in enumerate(axes):
            ax.plot(
                plot_data['osse_time_steps'],
                results['truth_osse_full'][state_idx, :],
                color=truth_color,
                linewidth=truth_linewidth,
                label='truth',
            )
            ax.scatter(
                plot_data['osse_obs_time_steps'],
                results['y_obs_osse'][state_idx, :],
                color=observation_color,
                s=observation_size,
                alpha=0.70,
                label='observations',
            )
            ax.plot(
                plot_data['analysis_time_steps'],
                plot_data['analysis_mean_array'][state_idx, :],
                color=analysis_color,
                linestyle=analysis_linestyle,
                linewidth=analysis_linewidth,
                alpha=0.95,
                label='analysis mean',
            )
            ax.plot(
                plot_data['analysis_time_steps'],
                plot_data['prior_mean_array'][state_idx, :],
                color=prior_color,
                linestyle=prior_linestyle,
                linewidth=prior_linewidth,
                alpha=0.90,
                label='prior mean',
            )
            if highlight_mean_difference:
                ax.fill_between(
                    plot_data['analysis_time_steps'],
                    plot_data['prior_mean_array'][state_idx, :],
                    plot_data['analysis_mean_array'][state_idx, :],
                    color='0.35',
                    alpha=mean_difference_alpha,
                    label='analysis-prior difference',
                    linewidth=0,
                )
            if zoom_xlim is not None:
                ax.set_xlim(*zoom_xlim)
            if ylim_from_means:
                xmask = np.ones_like(plot_data['analysis_time_steps'], dtype=bool)
                if zoom_xlim is not None:
                    xmask = (
                        (plot_data['analysis_time_steps'] >= zoom_xlim[0])
                        & (plot_data['analysis_time_steps'] <= zoom_xlim[1])
                    )
                mean_values = np.concatenate([
                    plot_data['prior_mean_array'][state_idx, xmask],
                    plot_data['analysis_mean_array'][state_idx, xmask],
                ])
                ymin = np.nanmin(mean_values)
                ymax = np.nanmax(mean_values)
                yrange = ymax - ymin
                if yrange == 0:
                    yrange = max(abs(ymax), 1.0)
                padding = mean_ylim_padding * yrange
                ax.set_ylim(ymin - padding, ymax + padding)
            ax.set_ylabel(f'state {self.state_labels[state_idx]}')
            ax.grid(True, alpha=0.25)
            if legend_mode == 'axes_reserved':
                ymin, ymax = ax.get_ylim()
                yrange = ymax - ymin
                if yrange == 0:
                    yrange = max(abs(ymax), 1.0)
                ax.set_ylim(ymin, ymax + legend_reserved_fraction * yrange)
                ax.legend(
                    loc=legend_loc,
                    fontsize=legend_fontsize,
                    framealpha=0.92,
                    ncol=legend_ncol,
                )
            if legend_mode == 'axes' and legend_outside:
                ax.legend(
                    loc='center left',
                    bbox_to_anchor=(1.01, 0.5),
                    fontsize=legend_fontsize,
                    framealpha=0.85,
                    ncol=legend_ncol,
                )
            elif legend_mode == 'axes':
                ax.legend(loc=legend_loc, fontsize=legend_fontsize, framealpha=0.80, ncol=legend_ncol)

        axes[-1].set_xlabel('Time step (1 step = 0.01)')
        axes[0].text(
            0.01,
            0.96,
            'Analysis total RMSE = {:.4f}\nAnalysis total spread = {:.4f}'.format(
                diagnostics['rmse_analysis_total'],
                diagnostics['analysis_spread_total'],
            ),
            transform=axes[0].transAxes,
            ha='left',
            va='top',
            fontsize=9,
            bbox={'facecolor': 'white', 'edgecolor': '0.75', 'alpha': 0.85, 'pad': 3},
        )
        if title is None:
            if zoom_xlim is None:
                title = f'OSSE truth, observations, prior mean, and analysis mean, start t0={results["t0"]}'
            else:
                title = (
                    'OSSE truth, observations, prior mean, and analysis mean; '
                    f'zoom {zoom_xlim[0]}-{zoom_xlim[1]} steps'
                )
        fig.suptitle(title, fontsize=13)
        if legend_mode == 'figure_right':
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc='center left',
                bbox_to_anchor=(0.86, 0.5),
                fontsize=legend_fontsize,
                framealpha=0.90,
                ncol=legend_ncol,
            )
            fig.tight_layout(rect=(0, 0, 0.84, 1))
        elif legend_outside:
            fig.tight_layout(rect=(0, 0, 0.84, 1))
        else:
            fig.tight_layout()
        return fig, axes

    def run_inflation_sweep(
        self,
        inflation_values,
        t0,
        Ta,
        num_cycles,
        multiple_number,
        T_0,
        seed=2,
        save_csv=True,
        csv_file='osse_inflation_sweep_table.csv',
        spinup_exclude=0,
    ):
        import pandas as pd

        diagnostics = []
        for inflation in inflation_values:
            das_run = copy.deepcopy(self.das)
            results = self.run_standard_osse(
                t0=t0,
                Ta=Ta,
                num_cycles=num_cycles,
                multiple_number=multiple_number,
                T_0=T_0,
                multiplicative_inflation=float(inflation),
                das=das_run,
                seed=seed,
            )
            run_diagnostics = self.compute_inflation_sweep_diagnostics(
                results,
                spinup_exclude=spinup_exclude,
            )
            diagnostics.append({
                'multiplicative_inflation': float(inflation),
                'analysis_spread_total': run_diagnostics['analysis_spread_total'],
                'rmse_analysis_total': run_diagnostics['rmse_analysis_total'],
                'spread_rmse_ratio': run_diagnostics['spread_rmse_ratio'],
                'n_outliers': run_diagnostics['n_outliers'],
            })

        diagnostics = sorted(diagnostics, key=lambda row: row['multiplicative_inflation'])
        columns = [f'case_{idx + 1}' for idx in range(len(diagnostics))]
        table = pd.DataFrame(
            [
                [row['analysis_spread_total'] for row in diagnostics],
                [row['rmse_analysis_total'] for row in diagnostics],
                [row['spread_rmse_ratio'] for row in diagnostics],
                [row['n_outliers'] for row in diagnostics],
                [row['multiplicative_inflation'] for row in diagnostics],
            ],
            index=[
                'ensemble spread',
                'RMSE',
                'spread/RMSE',
                'n_outliers',
                'multiplicative inflation',
            ],
            columns=columns,
        ).round(3)

        if save_csv:
            table.to_csv(csv_file, index=True)
            print('Saved OSSE inflation sweep table to:', csv_file)
        return table

    def plot_inflation_sweep(self, inflation_table, figsize=(8, 4.5)):
        inflation_plot_values = inflation_table.loc['multiplicative inflation'].to_numpy(dtype=float)
        spread_plot_values = inflation_table.loc['ensemble spread'].to_numpy(dtype=float)
        rmse_plot_values = inflation_table.loc['RMSE'].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(
            inflation_plot_values,
            rmse_plot_values,
            color='tab:blue',
            linewidth=1.4,
            marker='o',
            markersize=3,
            label='RMSE',
        )
        ax.plot(
            inflation_plot_values,
            spread_plot_values,
            color='tab:green',
            linewidth=1.4,
            marker='s',
            markersize=3,
            label='ensemble spread',
        )
        ax.set_xlabel('Multiplicative inflation')
        ax.set_ylabel('Value')
        ax.set_title('OSSE RMSE and ensemble spread vs multiplicative inflation')
        ax.grid(True, alpha=0.25)
        ax.legend(loc='best', fontsize=9)
        fig.tight_layout()
        return fig, ax

    def run_inflation_sweep_table(
        self,
        t0,
        Ta,
        num_cycles,
        multiple_number,
        T_0,
        inflation_values=None,
        seed=2,
        csv_file='osse_inflation_sweep_table.csv',
        table_figure_file=None,
        curve_figure_file=None,
        display_table=False,
        table_figsize=(12, 1.9),
        curve_figsize=(8, 4.5),
        spinup_exclude=0,
    ):
        """
        Run the OSSE inflation sweep and save the table and diagnostic figures.
        """
        if inflation_values is None:
            inflation_values = np.round(np.arange(1.01, 1.09 + 1e-12, 0.005), 3)

        inflation_table = self.run_inflation_sweep(
            inflation_values=inflation_values,
            t0=t0,
            Ta=Ta,
            num_cycles=num_cycles,
            multiple_number=multiple_number,
            T_0=T_0,
            seed=seed,
            save_csv=True,
            csv_file=csv_file,
            spinup_exclude=spinup_exclude,
        )

        if display_table:
            try:
                from IPython.display import display
                display(inflation_table)
            except ImportError:
                print(inflation_table)

        fig_table, ax_table = plt.subplots(figsize=table_figsize)
        ax_table.axis('off')
        table_artist = ax_table.table(
            cellText=inflation_table.values,
            rowLabels=inflation_table.index,
            colLabels=inflation_table.columns,
            loc='center',
            cellLoc='center',
            rowLoc='center',
        )
        table_artist.auto_set_font_size(False)
        table_artist.set_fontsize(7)
        table_artist.scale(1.0, 1.22)
        ax_table.set_title(
            f'OSSE inflation sweep diagnostics ({num_cycles} cycles, spin-up excluded={int(spinup_exclude)})',
            fontsize=11,
            pad=2,
        )
        fig_table.tight_layout()
        if table_figure_file:
            fig_table.savefig(table_figure_file, dpi=300, bbox_inches='tight')

        fig_curve, ax_curve = self.plot_inflation_sweep(inflation_table, figsize=curve_figsize)
        if curve_figure_file:
            fig_curve.savefig(curve_figure_file, dpi=300, bbox_inches='tight')

        return inflation_table, (fig_table, ax_table), (fig_curve, ax_curve)

    def save_standard_osse_outputs(
        self,
        results,
        diagnostics=None,
        extra_summary=None,
        truth_file='osse_truth_full.pkl',
        truth_da_file='osse_truth_da_times.pkl',
        obs_file='osse_yobs.pkl',
        summary_file='osse_summary.pkl',
        results_file='osse_results.pkl',
        analysis_file='osse_analysis_history.pkl',
        xa_history_file='osse_Xa_history.pkl',
        prior_file='osse_Xf_history.pkl',
    ):
        """
        Save the full standard OSSE output bundle used for diagnostics and plots.
        """
        diagnostics = {} if diagnostics is None else diagnostics
        extra_summary = {} if extra_summary is None else extra_summary

        truth_full_array = np.asarray(results['truth_osse_full'])
        truth_da_array = np.asarray(results['truth_osse_da'])
        yobs_array = np.asarray(results['y_obs_osse'])
        analysis_history = results['analysis_history']
        analysis_history_array = np.asarray(analysis_history)

        summary = {
            't0': int(results['t0']),
            'Ta': int(results['Ta']),
            'multiplicative_inflation': float(results['multiplicative_inflation']),
            'T_0': float(results['T_0']),
            'multiple_number': int(results['multiple_number']),
            'forecast_position': int(results['forecast_position']),
            'num_cycles': int(results['num_cycles']),
            'experiment_type': 'standard_offline_OSSE',
            'diagnostics': diagnostics,
            'truth_time_indices': results['truth_time_indices'],
            'da_time_indices': results['da_time_indices'],
        }
        summary.update(extra_summary)

        output_bundle = {
            'summary': summary,
            'truth_full': truth_full_array,
            'truth_da_times': truth_da_array,
            'yobs': yobs_array,
            'xf_4d_history': results['xf_4d_history'],
            'analysis_history': analysis_history,
            'prior_history': results['prior_history'],
            'diagnostics': diagnostics,
        }

        files_to_payloads = [
            (truth_file, truth_full_array),
            (truth_da_file, truth_da_array),
            (obs_file, yobs_array),
            (summary_file, summary),
            (results_file, output_bundle),
            (analysis_file, analysis_history_array),
            (xa_history_file, analysis_history_array),
            (prior_file, results['prior_history']),
        ]
        for output_file, payload in files_to_payloads:
            with open(output_file, 'wb') as fp:
                pickle.dump(payload, fp)

        print('Saved full OSSE truth to:', truth_file)
        print('Saved OSSE truth at DA times to:', truth_da_file)
        print('Saved precomputed OSSE observations to:', obs_file)
        print('Saved summary to:', summary_file)
        print('Saved consolidated results to:', results_file)
        print('Saved analysis history to:', analysis_file)
        print('Saved Xa history to:', xa_history_file)
        print('Saved prior history to:', prior_file)
        print('Full truth shape:', truth_full_array.shape)
        print('Truth at DA times shape:', truth_da_array.shape)
        print('Precomputed observations shape:', yobs_array.shape)
        print('Saved OSSE cycles:', len(analysis_history_array))
        print('Diagnostics:', diagnostics)

        return summary, output_bundle

    def plot_nature_observations_analysis_members(
        self,
        summary_file='osse_summary.pkl',
        analysis_history_file=None,
        prior_history_file=None,
        checkpoint_file=None,
        prior_checkpoint_file=None,
        plot_start=0,
        plot_window=350,
        obs_freq=None,
        output_dir=None,
        output_file=None,
        analysis_color='#7B2CBF',
        prior_color='#009E73',
        nature_color='blue',
        observation_edge_color='red',
        figsize=(12, 14),
        marker_size_analysis=35,
        marker_size_prior=30,
        marker_size_observation=25,
        show_prior_members=True,
        show=True,
        close=True,
    ):
        """
        Plot nature, observations, and OSSE analysis ensemble members.

        Analysis members are plotted as violet crosses at DA times. Prior
        members are plotted as green crosses when prior history is available.
        The method supports the minimal OSSE output files used by the CSE
        fallback.
        """
        summary_file = Path(summary_file)
        with open(summary_file, 'rb') as fp:
            osse_summary = pickle.load(fp)

        osse_t0 = int(osse_summary['t0'])
        osse_Ta = int(osse_summary['Ta'])
        osse_num_cycles = int(osse_summary.get('completed_cycles', osse_summary['num_cycles']))
        osse_end_index = int(osse_summary.get('osse_end_index', osse_t0 + osse_num_cycles * osse_Ta))

        plot_stop = min(plot_start + plot_window, osse_end_index - osse_t0)
        plot_t0 = osse_t0 + plot_start
        plot_tf = osse_t0 + plot_stop

        cycle_start = max(0, int(np.ceil((plot_t0 - osse_t0) / osse_Ta)))
        cycle_stop = min(osse_num_cycles, int(np.floor((plot_tf - 1 - osse_t0) / osse_Ta)) + 1)
        analysis_cycles = np.arange(cycle_start, cycle_stop)
        analysis_rel_times = osse_t0 + analysis_cycles * osse_Ta - plot_t0

        if checkpoint_file is None:
            checkpoint_file = osse_summary.get('checkpoint_file', 'osse_Xa_history_checkpoint.npy')
        if analysis_history_file is None:
            analysis_history_file = osse_summary.get('analysis_history_file', 'osse_Xa_history.pkl')
        if prior_checkpoint_file is None:
            prior_checkpoint_file = osse_summary.get('prior_checkpoint_file', 'osse_Xf_history_checkpoint.npy')
        if prior_history_file is None:
            prior_history_file = osse_summary.get('prior_history_file', 'osse_Xf_history.pkl')
        checkpoint_file = Path(checkpoint_file)
        analysis_history_file = Path(analysis_history_file)
        prior_checkpoint_file = Path(prior_checkpoint_file)
        prior_history_file = Path(prior_history_file)
        legacy_prior_history_file = Path('osse_prior_history.pkl')

        if checkpoint_file.exists():
            osse_Xa_history = np.load(checkpoint_file, mmap_mode='r')
        elif analysis_history_file.exists():
            with open(analysis_history_file, 'rb') as fp:
                osse_Xa_history = pickle.load(fp)
        else:
            raise FileNotFoundError(f'Could not find {checkpoint_file} or {analysis_history_file}.')

        prior_window = None
        if show_prior_members:
            if prior_checkpoint_file.exists():
                prior_history = np.load(prior_checkpoint_file, mmap_mode='r')
            elif prior_history_file.exists():
                with open(prior_history_file, 'rb') as fp:
                    prior_history = pickle.load(fp)
            elif legacy_prior_history_file.exists():
                with open(legacy_prior_history_file, 'rb') as fp:
                    prior_history = pickle.load(fp)
            else:
                prior_history = None
                print(
                    'Prior history not found. Skipping prior members: '
                    f'{prior_checkpoint_file}, {prior_history_file}, or {legacy_prior_history_file}.'
                )
            if prior_history is not None:
                prior_window = np.asarray(prior_history[cycle_start:cycle_stop])

        xa_window = np.asarray(osse_Xa_history[cycle_start:cycle_stop])
        obs_freq_plot = int(obs_freq if obs_freq is not None else osse_summary.get('obs_freq', osse_Ta))
        state_labels = self.state_labels

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=False)
        for state_idx, ax in enumerate(axes):
            ax.plot(
                self.x_nature[state_idx, plot_t0:plot_tf],
                linestyle='dashed',
                marker='.',
                markersize=3,
                color=nature_color,
                label='Nature Run',
            )
            ax.scatter(
                np.arange(0, plot_tf - plot_t0, obs_freq_plot),
                self.y_obs[state_idx, plot_t0:plot_tf:obs_freq_plot],
                marker='o',
                s=marker_size_observation,
                facecolors='none',
                edgecolors=observation_edge_color,
                label='Observations',
            )
            ax.scatter(
                np.repeat(analysis_rel_times, xa_window.shape[2]),
                xa_window[:, state_idx, :].reshape(-1),
                marker='x',
                s=marker_size_analysis,
                linewidths=1.0,
                color=analysis_color,
                label='Analysis members',
            )
            if prior_window is not None:
                ax.scatter(
                    np.repeat(analysis_rel_times, prior_window.shape[2]),
                    prior_window[:, state_idx, :].reshape(-1),
                    marker='x',
                    s=marker_size_prior,
                    linewidths=1.0,
                    color=prior_color,
                    label='Prior members',
                )
            ax.axhline(y=0, linestyle='dashed', color='grey')
            ax.axvline(x=0, linestyle='dashed', color='grey')
            ax.axvline(x=300, linestyle='dashed', color='grey')
            ax.set_ylabel(f'State {state_labels[state_idx]}', fontsize=14)
            ax.tick_params(axis='both', labelsize=14)
            ax.set_xlim(-5, (plot_tf - plot_t0) + 5)
            ax.legend(fontsize=12)

        axes[0].set_title('Lorenz-63 Nature Run, Observations, and OSSE Analysis Members', fontsize=18)
        axes[0].set_ylim(-18, 18)
        axes[-1].set_xlabel('Time step (1 step = 0.01)', fontsize=14)

        if output_dir is None:
            output_dir = self.path_figures
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_file is None:
            members_label = 'analysis_prior_members' if prior_window is not None else 'analysis_members'
            output_file = output_dir / f'osse_nature_observations_{members_label}_crosses_{plot_start}_{plot_stop}.png'
        else:
            output_file = Path(output_file)
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print('Saved:', output_file)

        if show:
            plt.show()
        if close:
            plt.close(fig)
        return fig, axes, output_file

    def _check_window(self, truth_time_indices, da_time_indices):
        if truth_time_indices[-1] >= self.x_nature.shape[1]:
            raise ValueError(
                f'OSSE truth window exceeds x_nature length: last index {truth_time_indices[-1]}, '
                f'available {self.x_nature.shape[1] - 1}.'
            )
        if da_time_indices[-1] >= self.y_obs.shape[1]:
            raise ValueError(
                f'OSSE observation window exceeds y_obs length: last index {da_time_indices[-1]}, '
                f'available {self.y_obs.shape[1] - 1}.'
            )

    def _get_results(self, results):
        if results is None:
            if self.results is None:
                raise ValueError('No OSSE results available. Run run_standard_osse first.')
            return self.results
        return results
