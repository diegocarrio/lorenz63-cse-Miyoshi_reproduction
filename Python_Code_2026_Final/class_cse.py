import numpy as np
import random
import pickle
import numpy.matlib
import matplotlib.pyplot as plt
from scipy import linalg
from class_da_sys import da_system
from class_Lorenz63 import run_RK4
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from matplotlib.colors import Normalize

class cse:
    
    def __init__(
        self,
        x0=[],
        yo=[],
        t0=0,
        dt=0,
        state_vector=[],
        obs_data=[],
        acyc_step=1,
        str_figures_path='unknown',
        seed=2,
        osse_Xa_history=None,
        osse_start_index=None,
        osse_Xa_history_orientation=None,
    ):
        self.xdim           = np.size(x0)
        self.ydim           = np.size(yo)
        self.edim           = 0
        self.yo             = yo
        self.yo_history     = []
        self.x0             = x0
        self.t0             = t0
        self.dt             = dt
        self.t              = t0
        self.acyc_step      = acyc_step       # Number of time steps between assimilation cycles
        self.dtau           = dt * acyc_step  # Time interval between assimilation cycles
        self.fcst_step      = acyc_step       # Number of time steps between forecasts
        self.fcst_dt        = dt
        self.maxit          = 0
        self.statedim       = 0
        self.paradim        = 0
        self.B              = np.matrix(np.identity(self.xdim))  # Background error covariance matrix
        self.R              = np.matrix(np.identity(self.ydim))  # Observation error covariance matrix
        self.H              = np.matrix(np.identity(self.xdim))  # Observation operator matrix
        self.SqrtB          = []
        self.state_vector   = state_vector
        self.obs_data       = obs_data
        self.method         = ''
        self.KH             = []
        self.khidx          = []
        self.das_bias_init  = 0
        self.das_sigma_init = 0.1
        self.history        = []
        self.history_final  = []
        self.multiple_number = 4
        self.T_0            = 75.1
        self.Ta             = 8
        self.t_index        = 0
        self.xf4d           = []
        self.target         = []
        self.xf_init        = []
        self.xf_unpert      = []
        self.xf_pert        = []
        self.norm_diff_ensemble = []
        self.intervention_history = []
        self.last_intervention = None
        self.figures_path   = str_figures_path
        self.rng            = np.random.default_rng(seed)
        self.regime_shift_criterion_log = []
        self.log_regime_shift_criterion_diff = False
        self.step4b_previous_analysis_max_search_depth = 20
        self.plot_step4b_osse_history_exhausted = True
        self.osse_Xa_history = osse_Xa_history
        self.osse_start_index = osse_start_index
        self.osse_Xa_history_orientation = osse_Xa_history_orientation
        self.last_step4b_search_attempts = []
    

    def __str__(self):
        print('CSE System Object')
        print('xdim: ', self.xdim)
        print('ydim: ', self.ydim)
        print('x0: ', self.x0)
        print('t0: ', self.t0)
        print('dt: ', self.dt)
        print('t: ', self.t)
        print('acyc_step: ', self.acyc_step)
        print('dtau: ', self.dtau)
        print('fcst_step: ', self.fcst_step)
        print('fcst_dt: ', self.fcst_dt)
        print('B: ')
        print(self.B)
        print('R: ')
        print(self.R)
        print('H: ')
        print(self.H)
        print('state_vector: ')
        print(self.state_vector)
        print('obs_data: ')
        print(self.obs_data)
        print('method: ')
        print(self.method)
        print('figures_path: ', self.figures_path)
        return 'type::da_system'
    

    # Get self.target:
    def getTarget(self):
        return self.target


    def _record_intervention(self, branch, perturbations, metadata=None):
        perturbations = np.asarray(perturbations, dtype=float)
        if perturbations.ndim == 1:
            perturbations = perturbations.reshape(1, -1)
        norms = np.linalg.norm(perturbations, axis=1)
        record = {
            'cycle_index': int(self.t_index // self.Ta) if self.Ta else None,
            'time_index': int(self.t_index),
            'branch': branch,
            'controlled': bool(np.any(norms > 0)),
            'perturbations': perturbations.copy(),
            'norms': norms.copy(),
            'metadata': metadata or {},
        }
        self.last_intervention = record
        self.intervention_history.append(record)
        return record

    def _warn_if_step4b_safe_opposite_regime(self, xf_safe, regime_sign):
        safe_x0 = float(np.asarray(xf_safe, dtype=float)[0, 0])
        regime_sign = float(regime_sign)
        if regime_sign != 0.0 and np.sign(safe_x0) != regime_sign:
            print(
                f"CRITICAL [cycle {self.t_index // self.Ta}]: "
                "xf_safe seleccionado pertenece al regimen contrario "
                f"(safe_x0={safe_x0}, regime_sign={regime_sign}). "
                "La perturbacion empujara la NR HACIA el cambio de regimen."
            )

    def _integrate_rk4_n_steps(self, solver, initial_state, dt, n_steps):
        temp_solver = run_RK4(
            solver.func,
            solver.sigma,
            solver.beta,
            solver.rho,
            initial_state,
            dt,
            dt * max(n_steps - 1, 0),
        )
        # run_RK4 uses int(t_final / h), which can lose one step through
        # floating-point roundoff. Force the exact requested grid length here.
        temp_solver.num_steps = max(n_steps - 1, 0)
        temp_solver.times = np.arange(n_steps) * dt
        temp_solver.trajectory = np.empty((n_steps, len(initial_state)))
        temp_solver.trajectory[0] = temp_solver.state
        return temp_solver.integrate()


    #
    # Defining the function to detect regime shifts:
    #
    def detect_regime_shift(self, forecast):
        forecast = np.asarray(forecast, dtype=float)
        return not (np.all(forecast > 0.0) or np.all(forecast < 0.0))
    
    
    #
    # Defining the function to classify ensemble members:
    #
    def classify_ensemble(self, Xf):
        predicted_changes = []
        safe_members = []
        for k in range(Xf.shape[0]):
            if self.detect_regime_shift(Xf[k, :, 0]):
                predicted_changes.append(k)
            else:
                safe_members.append(k)
        return predicted_changes, safe_members
    
    
    #
    # Defining the function to compute perturbation vector:
    #
    def compute_perturbation(self, xf_safe, xf_shift, D):
        delta = xf_safe - xf_shift
        amp = 1 / D
        perturbation_vector = delta / (np.linalg.norm(delta, axis=1, keepdims=True) * amp)
        # Check perturbations are normalized:
        #print(f' perturbation_vector shape: {perturbation_vector.shape}')
        #print('check perturbations are normalized: ', np.sqrt(perturbation_vector[0:10,0]**2 + perturbation_vector[0:10,1]**2 + perturbation_vector[0:10,2]**2))
        return perturbation_vector
    

    #
    # Defining the function to evolve nature with control:
    #
    def evolve_nature_with_control(self, x_init, perturb_vec, solver, cycle_steps):
        x = x_init.copy()
        history = []
        for q in range(cycle_steps):
            x_perturbed = x + perturb_vec[q, :]
            temp_solver = run_RK4(solver.func, solver.sigma, solver.beta, solver.rho, x_perturbed, self.dt, 1.5 * self.dt)
            x_next = temp_solver.integrate()[1, :]
            history.append(x_perturbed.tolist())
            x = x_next
        return x, history




    #
    # Defining function to perform the analysis step (Step 1 in CSE):
    #
    def CSE_Step1(self, Xf, yo, das):
        """
        STEP 1: PERFORM DA UPDATE:
        """
        #print('CSE System Step 1')
        method = das.getMethod()
        #print('Method: ', method)
        Xa, KH = das.compute_analysis(Xf,yo)
        return Xa, KH
    
    
    
    #
    # Defining function to perform the forecast step (Step 2 in CSE):
    #
    def CSE_Step2(self, multiple_number, T_0, solver, Xa, das):
        """
        STEP 2: RUN ENSEMBLE FORECAST:
        """
        #print('CSE System Step 2')
        position = int(np.ceil(T_0*multiple_number)) #multiple of To
        #print(f'Position: {position}')
        xf_4d = np.zeros((das.edim, position, das.xdim))
        #print(f'initial condition forecast: {Xa}')
        # Use das.fcst_dt instead of self.dt
        dt = das.fcst_dt  # Get the correct time step from das object
        # Run forecast for each ensemble member, using the initial condition from the posterior ensemble:
        for k in range(das.edim):
            # Integrate for (position-1) steps to get exactly position points (including initial condition)
            xf_ens = self._integrate_rk4_n_steps(solver, np.ravel(Xa[:, k]), dt, position)
            xf_4d[k,:,:] = xf_ens
        return xf_4d

    def CSE_Step2_n_steps(self, n_steps, solver, Xa, das):
        """
        STEP 2 helper: run an ensemble forecast with an explicit number of
        stored time steps, including the initial condition.
        """
        xf_4d = np.zeros((das.edim, n_steps, das.xdim))
        dt = das.fcst_dt
        for k in range(das.edim):
            xf_ens = self._integrate_rk4_n_steps(solver, np.ravel(Xa[:, k]), dt, n_steps)
            xf_4d[k, :, :] = xf_ens
        return xf_4d

    def _analysis_as_xdim_edim(self, analysis, das, source_label, preferred_orientation=None):
        """
        Return an analysis ensemble with shape (xdim, edim), accepting either
        DA-analysis storage (xdim, edim) or forecast-history storage (edim, xdim).
        """
        analysis = np.asarray(analysis, dtype=float)
        expected_xdim_edim = (int(das.xdim), int(das.edim))
        expected_edim_xdim = (int(das.edim), int(das.xdim))

        if preferred_orientation == 'xdim_edim':
            if analysis.shape != expected_xdim_edim:
                raise ValueError(
                    f'{source_label} has shape {analysis.shape}, expected {expected_xdim_edim}.'
                )
            return analysis
        if preferred_orientation == 'edim_xdim':
            if analysis.shape != expected_edim_xdim:
                raise ValueError(
                    f'{source_label} has shape {analysis.shape}, expected {expected_edim_xdim}.'
                )
            return analysis.T

        if analysis.shape == expected_xdim_edim:
            return analysis
        if analysis.shape == expected_edim_xdim:
            return analysis.T
        raise ValueError(
            f'{source_label} has shape {analysis.shape}, expected '
            f'{expected_xdim_edim} or {expected_edim_xdim}.'
        )

    def _plot_step4b_osse_history_exhausted(
        self,
        Xf,
        current_shift_members,
        current_time_steps,
        Ta,
        cycles_back,
        osse_index,
        osse_history_length,
        control_history_depth,
    ):
        """
        Diagnostic plot for Step 4b when the requested OSSE analysis index does
        not exist. In this case there is no past ensemble to draw, so the plot
        shows the current shift members and the missing OSSE-history metadata.
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        current_shift_members = np.asarray(current_shift_members, dtype=int)
        if current_shift_members.size == 0:
            current_shift_members = np.arange(Xf.shape[0], dtype=int)

        for member_idx in current_shift_members:
            ax.plot(
                current_time_steps,
                Xf[member_idx, :, 0],
                color='tab:blue',
                linestyle='-',
                linewidth=1.3,
                alpha=0.9,
                label='current cycle shift members' if member_idx == current_shift_members[0] else None,
            )
            ax.scatter(
                current_time_steps[0],
                Xf[member_idx, 0, 0],
                color='tab:blue',
                edgecolors='black',
                s=42,
                zorder=5,
            )
            ax.scatter(
                current_time_steps[-1],
                Xf[member_idx, -1, 0],
                color='tab:blue',
                edgecolors='black',
                s=42,
                zorder=5,
            )

        ax.axhline(0.0, color='red', linestyle=':', linewidth=1.6, label='x = 0')
        ax.axvline(0, color='tab:blue', linestyle='--', linewidth=1.4)
        ax.axvline(current_time_steps[-1], color='0.35', linestyle='--', linewidth=1.4)
        ax.set_xticks(np.unique(np.concatenate(([0], np.arange(0, current_time_steps[-1] + 1, 50), [current_time_steps[-1]]))).astype(int))
        ax.set_xlim(current_time_steps[0] - 5, current_time_steps[-1] + 5)

        ax.text(
            1.0,
            0.98,
            'current\nstart',
            color='tab:blue',
            fontsize=8,
            ha='left',
            va='top',
            transform=ax.get_xaxis_transform(),
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.75, 'pad': 1.5},
        )
        ax.text(
            current_time_steps[-1] - 1.0,
            0.98,
            'aligned\nfinal',
            color='0.35',
            fontsize=8,
            ha='right',
            va='top',
            transform=ax.get_xaxis_transform(),
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.75, 'pad': 1.5},
        )

        if osse_history_length > 0:
            available_range = f'0..{osse_history_length - 1}'
        else:
            available_range = 'empty'
        message = (
            'OSSE history exhausted\n'
            f'cycles_back={cycles_back}\n'
            f'osse_index={osse_index}\n'
            f'available={available_range}\n'
            f'control_history_depth={control_history_depth}'
        )
        ax.text(
            0.02,
            0.05,
            message,
            transform=ax.transAxes,
            fontsize=9,
            ha='left',
            va='bottom',
            bbox={'facecolor': 'white', 'edgecolor': '0.7', 'alpha': 0.9, 'pad': 6.0},
        )

        ax.set_xlabel('Time step (1 step = 0.01)')
        ax.set_ylabel('state x')
        ax.set_title(
            'OSSE history exhausted before past-analysis alignment\n'
            f'cycle={self.t_index // Ta}, cycles_back={cycles_back}, osse_index={osse_index}'
        )
        ax.grid(True, alpha=0.25)
        ax.legend(loc='upper right', fontsize=7, framealpha=0.82, handlelength=1.5)
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    #
    # Defining function to detect regime shift (Step 3 in CSE):
    #
    def CSE_Step3(self, Xf):
        """
        STEP 3: DETECT REGIME SHIFT:
        """
        #print('CSE System Step 3')
        predicted_changes, safe_members = self.classify_ensemble(Xf)
        old_changes = [
            k for k in range(Xf.shape[0])
            if np.asarray(Xf[k, :, 0], dtype=float)[-1] < 0.0
        ]
        changed_members = sorted(set(predicted_changes).symmetric_difference(old_changes))
        criterion_record = {
            't_index': int(self.t_index),
            'new_shift_count': int(len(predicted_changes)),
            'old_shift_count': int(len(old_changes)),
            'changed_count': int(len(changed_members)),
            'changed_members': [int(k) for k in changed_members],
        }
        self.regime_shift_criterion_log.append(criterion_record)
        if self.log_regime_shift_criterion_diff and changed_members:
            print(
                'Regime-shift criterion diff '
                f'(t_index={self.t_index}): '
                f'new={len(predicted_changes)}, old={len(old_changes)}, '
                f'changed={changed_members}'
            )
        #if len(predicted_changes) > 0:
        #    print(f'At least one ensemble member shows regime shift. Ensemble members showing regime shift: {predicted_changes}')
        return predicted_changes, safe_members
    
    
    
    
    #
    # Defining function to advance Nature Run if none of the members show regime shift (Step 3b in CSE):
    #
    def CSE_Step3b(self, safe_members, Xf, x_original, solver, Ta, D, das):
        """
        STEP 3B: ADVANCE NATURE RUN IF NONE OF THE MEMBERS SHOW REGIME SHIFT:
        """
        #print('CSE System Step 3b: Advance Nature Run to the next time step where observations are available')
        #
        #x_history = [x_original[0]]
        x_history = [x_original[:]]
        #print(f'x_original: {x_original}')
        #x_history = []
        #
        # Evolve NR forward
        #x_fcst = solver.run(x_original, np.arange(0, 7.5 * self.dt, self.dt), np.arange(0, 7.5 * self.dt, self.dt))[1, :]
        temp_solver1 = run_RK4(solver.func, solver.sigma, solver.beta, solver.rho, x_original, self.dt, 1.5 * self.dt)
        x_next = temp_solver1.integrate()[1, :]
        temp_solver2 = run_RK4(solver.func, solver.sigma, solver.beta, solver.rho, x_next, self.dt, 7.5 * self.dt)
        x_fcst = temp_solver2.integrate()
        #print('x_next shape: ', x_next.shape)
        #print('x_fcst shape: ', x_fcst.shape)
        #print('x_fcst[0,:]: ', x_fcst[0,:])
        x_final = x_fcst[-1,:]
        x_history.extend(x_fcst[0:-1,:].tolist())
        #
        safe_idx2  = self.rng.choice(safe_members)
        #print(f'safe_idx: {safe_idx2}')
        xf_safe   = Xf[safe_idx2, 0:8, :]
        self.target.append(xf_safe)
        zero_perturb = np.zeros((Ta, np.size(x_original)))
        self._record_intervention(
            'step3b_no_control',
            zero_perturb,
            metadata={'safe_member_index': int(safe_idx2)},
        )
        ####x_history.append(x_fcst[0:-1,:].tolist())
        #print('x_history shape: ', np.shape(x_history))
        #print('x_history: ', x_history)
        #x_first_component_history = x_history[0,:,:]
        #x_first_component_history = [x[0] for x in x_history]
        return x_final, x_history
    
    
    
    #
    # Defining function to perform the Control if at least one member shows a regime shift (Step 4 in CSE):
    #
    def CSE_Step4a(self, predicted_changes, safe_members, Xf, x_original, solver, Ta, D, das):
        """
        STEP 4a: CONTROL:
        """
        #print('CSE System Step 4a')
        x_final_history = []
        x_final_history.append(x_original[:].tolist())
        # Apply control only if some predict shift and others don't
        if 0 < len(predicted_changes) < Xf.shape[0]:
            #print(f'At least one ensemble member show regime shift. Activating control.')
            #print(f'initial condition shift Xf[:,0,:]: {Xf[:,0,:]}')
            # Select members
            shift_idx = self.rng.choice(predicted_changes)
            safe_idx  = self.rng.choice(safe_members)
            xf_shift  = Xf[shift_idx, :, :]
            xf_safe   = Xf[safe_idx, :, :]
            #print(f'safe_idx: {safe_idx}')
            #print(f'shape xf_safe:', xf_safe.shape)
            self.target.append(xf_safe[0:Ta,:])
            #
            # Compute perturbation vector
            perturb_vec = self.compute_perturbation(xf_safe, xf_shift, D)
            #print(f'perturb_vec: {perturb_vec}')
            #print(f' x_safe: {xf_safe}')
            #print(f' x_shift: {xf_shift}')
            #
            # Evolve NR forward with control
            temp_solver = run_RK4(solver.func, solver.sigma, solver.beta, solver.rho, x_original, self.dt, 1.5 * self.dt)
            x_next = temp_solver.integrate()[1, :]
            #print('x_next shape: ', x_next.shape)
            #print('x_next: ', x_next)
            x_final, x_history = self.evolve_nature_with_control(x_next, perturb_vec, solver, Ta - 1)
            x_final_history.extend(x_history)
            stored_perturb = np.vstack([np.zeros((1, perturb_vec.shape[1])), perturb_vec[:Ta-1, :]])
            self._record_intervention(
                'step4a_partial_control',
                stored_perturb,
                metadata={
                    'safe_member_index': int(safe_idx),
                    'shift_member_index': int(shift_idx),
                    'D': float(D),
                },
            )
            #print(f'x_history: {x_history}')
        return x_final, x_final_history
    
    
    
    #
    # Defining function to perform the Control (Step 4 in CSE):
    #
    def CSE_Step4b(self, predicted_changes, safe_members, Xf, x_original, solver, Ta, D, das):
        """
        STEP 4b: CONTROL:
        """
        print('CSE System Step 4b')
        x_final_history = []
        x_final_history.append(x_original[:].tolist())
        # Apply control only if some predict shift and others don't
        print(f'All ensemble members show regime shift. Activating control using members from previous initial times for an extended forecasting period.')
        #
        # Check if any of the ensemble members show a regime shift starting from the last time step in the history:
        #
        all_shifted    = True
        time_shift     = 1
        time_shift_max = 200
        t_index_incr   = Ta*time_shift
        regime_sign    = np.sign(x_original[0])
        while all_shifted:
            #print(f'self.history shape: {np.shape(self.history_final)}')
            history_arr = np.array(self.history[-(time_shift+1)])[:, :]
            #print(f'history_arr: {history_arr}')   
            history_current = np.array(self.history[-1])[:, :]
            #print(f'history_current: {history_current}')
            #print(f'history_arr shape: {history_arr.shape}')
            #
            # STEP 2: Run ensemble forecast with the last time step in the history:
            #
            #print(f'  Running ensemble forecast with time shift: {time_shift}')
            xf_4d = self.CSE_Step2(self.multiple_number, self.T_0, solver, history_arr.T, das)
            #print(f'  xf_4d shape: {xf_4d.shape}')
            #print(f'  initial condition xf_4d[:,8,:].T: {xf_4d[:,8,:].T}')
            #print(f'  initial condition xf_4d[:,8,:]: {xf_4d[:,8,:]}')
            t_index_shifted = self.t_index-(t_index_incr)
            #print(f'  t_index: {self.t_index}')
            #print(f'  t_index_shifted: {t_index_shifted}')
            #
            # STEP 3: Detect regime shift:
            #
            #print('  Detecting regime shift ensemble members...')
            predicted_changes, safe_members = self.CSE_Step3(xf_4d)
            #print(f'  Shift predicted by: {predicted_changes}')
            #print(f'  Safe ensemble members: {safe_members}')
            # Check if any safe members are found:
            if len(safe_members) > 0:
                all_shifted = False
                #print(f'  Found safe ensemble members after {time_shift} time steps.')
                t_index_shifted = self.t_index-(t_index_incr)
                #print('t_index: ', self.t_index)
                #print('t_index_incr: ', t_index_incr)
                #print('t_index_shifted: ', t_index_shifted)
                #
                # Step 4: Control
                #
                #print('  Applying control...')
                #print(f'  initial condition shift xf_4d[:,0,:]: {xf_4d[:,0,:]}')
                #
                # Select members
                #
                print(f'  Selecting ensemble members for control...')
                print(f' safe_members: {safe_members}')
                #shift_idx = self.rng.choice(predicted_changes)
                safe_idx  = self.rng.choice(safe_members)
                ttt = xf_4d.shape[1] - t_index_incr
                #print(f' ttt: {ttt}')
                xf_shift  = Xf[0, 0:ttt, :]
                #print(f' safe_idx: {safe_idx}')
                #print(f' shift_idx: {shift_idx}')
                print(f' t_index_incr: {t_index_incr}')
                print(f' ttt: {ttt}')
                print(f' ttt+t_index_incr: {ttt+t_index_incr}')
                #print(f'shape xf_4d:', xf_4d.shape)
                xf_safe   = xf_4d[safe_idx, t_index_incr:ttt+t_index_incr:1, :]
                self._warn_if_step4b_safe_opposite_regime(xf_safe, regime_sign)
                self.target.append(xf_safe[0:Ta,:])
                #print(f'  xf_shift shape: {xf_shift.shape}')
                #
                # Compute perturbation vector
                #print('Computing perturbation vector...')
                #print(f' x_safe: {xf_safe}')
                #print(f' x_shift: {xf_shift}')
                perturb_vec = self.compute_perturbation(xf_safe, xf_shift, D)
                #print(f'perturb_vec: {perturb_vec}')
                #
                #
                # Evolve NR forward with control
                temp_solver = run_RK4(solver.func, solver.sigma, solver.beta, solver.rho, x_original, self.dt, 1.5 * self.dt)
                x_next  = temp_solver.integrate()[1, :]
                #print('x_next shape: ', x_next.shape)
                #print('x_next: ', x_next)
                x_final, x_history = self.evolve_nature_with_control(x_next, perturb_vec, solver, Ta - 1)
                x_final_history.extend(x_history)
                #print(f'  x_history: {x_history}')
                #print(f'  x_original: {x_original}')
                #print(f'  x_final: {x_final}')
                #print('   x_final_history: ', x_final_history)
                x_history_only = [x[0] for x in x_final_history]
                #print('   x_history_only: ', x_history_only)
                #print(np.arange(self.t_index,Ta+self.t_index,1))
                #print('  Control applied successfully.')
            else:
                #print(f'  No safe ensemble members found after {time_shift} time steps. Continuing to search.')
                time_shift += 1
                if time_shift > time_shift_max:
                    # If no safe members are found after max time steps, exit loop:
                    Xf = xf_4d.copy()
                    #print('t_index: ', self.t_index)
                    t_index_incr = Ta*time_shift
                    t_index_shifted = self.t_index-(t_index_incr)
                    #print(f'  No safe ensemble members found after {time_shift_max} time steps. Exiting loop.')
                    #print('  EXITING CONTROL LOOP!!!!')
                    return
        return x_final, x_final_history
    
    
    def CSE_Step4b_fixed(self, predicted_changes, safe_members, Xf, x_original, solver, Ta, D, das):
        """
        STEP 4b (FIXED VERSION): CONTROL with robust dimension handling
        
        This version fixes the bug where temporal_offset was calculated outside the loop
        and adds robust handling for cases where the forecast doesn't have enough steps.
        
        KEY IMPROVEMENTS:
        1. temporal_offset recalculated INSIDE while loop (bug fix)
        2. Validation that available_steps >= Ta before extraction
        3. Forward projection when insufficient steps available
        """
        print('CSE System Step 4b (FIXED - ROBUST)')
        x_final_history = []
        x_final_history.append(x_original[:].tolist())
        
        print(f'All ensemble members show regime shift. Activating control using members from previous cycles.')
        
        # Special case: First cycle (no history available)
        if len(self.history) < 2:
            print(f'  ⚠️  WARNING: First cycle - no previous history available.')
            print(f'  Evolving without control for Ta={Ta} steps.')
            # Evolve nature run WITHOUT control for Ta steps
            x_temp = x_original.copy()
            for step in range(Ta):
                x_final_history.append(x_temp.tolist())
                temp_solver = run_RK4(solver.func, solver.sigma, solver.beta, solver.rho, x_temp, self.dt, 1.5 * self.dt)
                x_temp = temp_solver.integrate()[1, :]
            self._record_intervention('step4b_fixed_no_control', np.zeros((Ta, np.size(x_original))), metadata={'reason': 'insufficient_history'})
            return x_temp, x_final_history[1:]  # Return final state and Ta elements (skip initial)
        
        # Initialize temporal search variables
        searching = True
        cycles_back = 1
        max_search_depth = 200
        regime_sign = np.sign(x_original[0])
        
        # MAIN LOOP: Search backward through history until finding safe members
        while searching:
            # Check if we can access the required history
            if cycles_back + 1 > len(self.history):
                print(f'  ⚠️  Reached end of history (only {len(self.history)} cycles available).')
                print(f'  Cannot search {cycles_back} cycles back. Evolving without control for Ta={Ta} steps.')
                # Evolve nature run WITHOUT control for Ta steps
                x_temp = x_original.copy()
                for step in range(Ta):
                    x_final_history.append(x_temp.tolist())
                    temp_solver = run_RK4(solver.func, solver.sigma, solver.beta, solver.rho, x_temp, self.dt, 1.5 * self.dt)
                    x_temp = temp_solver.integrate()[1, :]
                self._record_intervention('step4b_fixed_no_control', np.zeros((Ta, np.size(x_original))), metadata={'reason': 'history_exhausted', 'cycles_back': int(cycles_back)})
                return x_temp, x_final_history[1:]  # Return final state and Ta elements (skip initial)
            
            # BUG FIX: Recalculate temporal_offset INSIDE loop
            temporal_offset = Ta * cycles_back
            
            # Retrieve ensemble initial conditions from (cycles_back+1) cycles ago
            history_arr = np.array(self.history[-(cycles_back+1)])[:, :]
            
            # STEP 2: Forecast forward from the past ensemble
            xf_4d = self.CSE_Step2(self.multiple_number, self.T_0, solver, history_arr.T, das)
            
            t_index_shifted = self.t_index - temporal_offset
            
            # STEP 3: Check which members maintain correct regime
            predicted_changes, safe_members = self.CSE_Step3(xf_4d)
            
            # CASE 1: Found safe members
            if len(safe_members) > 0:
                # Calculate available forecast length after temporal offset
                available_steps = xf_4d.shape[1] - temporal_offset
                
                # EDGE CASE: temporal_offset is beyond or at end of forecast
                if available_steps <= 0:
                    print(f'  ✗ temporal_offset ({temporal_offset}) >= forecast length ({xf_4d.shape[1]})')
                    print(f'    No forecast data available. Searching deeper...')
                    cycles_back += 1
                    continue
                
                # VALIDATION: Check if there are enough steps available
                if available_steps < Ta:
                    print(f'  ⚠️  WARNING: Not enough steps ({available_steps} < {Ta}).')
                    print(f'    Projecting forward to complete Ta={Ta} steps.')
                    
                    safe_idx = self.rng.choice(safe_members)
                    xf_safe_partial = xf_4d[safe_idx, temporal_offset:, :]
                    
                    x_last = xf_safe_partial[-1, :].copy()
                    remaining_steps = Ta - available_steps
                    
                    # Project forward - need to generate enough steps
                    projection_time = self.dt * (remaining_steps + 1)  # +1 to ensure we get enough points
                    proj_solver = run_RK4(solver.func, solver.sigma, solver.beta, solver.rho, x_last, self.dt, projection_time)
                    x_projected = proj_solver.integrate()
                    
                    # Combine available + projected (skip first point of x_projected as it's x_last)
                    xf_safe = np.vstack([xf_safe_partial, x_projected[1:remaining_steps+1, :]])
                    self._warn_if_step4b_safe_opposite_regime(xf_safe, regime_sign)
                    print(f'    ✓ Complete xf_safe: {xf_safe.shape} (partial: {xf_safe_partial.shape}, projected: {remaining_steps} steps)')
                    
                else:
                    # Normal case: enough steps available
                    safe_idx = self.rng.choice(safe_members)
                    xf_safe = xf_4d[safe_idx, temporal_offset:temporal_offset+Ta, :]
                    self._warn_if_step4b_safe_opposite_regime(xf_safe, regime_sign)
                
                searching = False
                print(f'  ✓ Found {len(safe_members)} safe members after {cycles_back} cycle(s).')
                print(f'    temporal_offset={temporal_offset}, available={available_steps}')
                
                # Extract SHIFT trajectory
                xf_shift = Xf[0, 0:Ta, :]
                
                # Store target
                self.target.append(xf_safe)
                
                # Compute perturbation and apply control
                perturb_vec = self.compute_perturbation(xf_safe, xf_shift, D)
                temp_solver = run_RK4(solver.func, solver.sigma, solver.beta, solver.rho, x_original, self.dt, 1.5 * self.dt)
                x_next = temp_solver.integrate()[1, :]
                x_final, x_history = self.evolve_nature_with_control(x_next, perturb_vec, solver, Ta - 1)
                x_final_history.extend(x_history)
                stored_perturb = np.vstack([np.zeros((1, perturb_vec.shape[1])), perturb_vec[:Ta-1, :]])
                self._record_intervention(
                    'step4b_fixed_control',
                    stored_perturb,
                    metadata={
                        'safe_member_index': int(safe_idx),
                        'cycles_back': int(cycles_back),
                        'temporal_offset': int(temporal_offset),
                        'D': float(D),
                    },
                )
                
            # CASE 2: No safe members
            else:
                print(f'  ✗ No safe members at cycles_back={cycles_back}')
                cycles_back += 1
                
                if cycles_back > max_search_depth:
                    print(f'  ⚠ Max search depth ({max_search_depth}). Evolving without control for Ta={Ta} steps.')
                    # Evolve nature run WITHOUT control for Ta steps
                    x_temp = x_original.copy()
                    for step in range(Ta):
                        temp_solver = run_RK4(solver.func, solver.sigma, solver.beta, solver.rho, x_temp, self.dt, 1.5 * self.dt)
                        x_temp = temp_solver.integrate()[1, :]
                    self._record_intervention('step4b_fixed_no_control', np.zeros((Ta, np.size(x_original))), metadata={'reason': 'max_search_depth', 'cycles_back': int(cycles_back)})
                    return x_temp, x_final_history[1:]  # Return final state and Ta elements (skip initial)
        
        return x_final, x_final_history
    
    
    
    
    #
    # Defining function to create new observations at t + Ta (Step 5 in CSE):
    #
    def CSE_Step5(self, nature, das):
        """
        STEP 5: CREATE NEW OBSERVATIONS AT t + Ta:
        """
        #print('CSE System Step 5')
        new_nr = nature.copy()
        yo_new = new_nr + self.rng.normal(0, np.sqrt(2.0), das.xdim)
        return yo_new
        
    
    
    def CSE_Step4b_optimized(self, predicted_changes, safe_members, Xf, x_original, solver, Ta, D, das):
        """
        STEP 4b: CONTROL (OPTIMIZED VERSION)
        
        This version is identical to CSE_Step4b but fixes the bug where t_index_incr 
        was calculated outside the loop and not updated when time_shift incremented.
        
        ONLY CHANGE: Line "t_index_incr = Ta*time_shift" moved INSIDE the while loop.
        """
        print('CSE System Step 4b (OPTIMIZED)')
        x_final_history = []
        x_final_history.append(x_original[:].tolist())
        
        print(f'All ensemble members show regime shift. Activating control using members from previous initial times for an extended forecasting period.')
        
        # Check if any of the ensemble members show a regime shift starting from the last time step in the history
        all_shifted    = True
        time_shift     = 1
        time_shift_max = 20  # Match Qiwen's code - search deep into history
        regime_sign    = np.sign(x_original[0])
        
        while all_shifted:
            # Check if we have enough history before accessing
            if (time_shift + 1) > len(self.history):
                print(f'  Insufficient history (need {time_shift+1}, have {len(self.history)}). Evolving without control.')
                x_temp = x_original.copy()
                for step in range(Ta):
                    temp_solver = run_RK4(solver.func, solver.sigma, solver.beta, solver.rho, x_temp, self.dt, 1.5 * self.dt)
                    x_temp = temp_solver.integrate()[1, :]
                    x_final_history.append(x_temp.tolist())
                self._record_intervention('step4b_optimized_no_control', np.zeros((Ta, np.size(x_original))), metadata={'reason': 'insufficient_history', 'time_shift': int(time_shift)})
                return x_temp, x_final_history[1:]
            
            # BUG FIX: Calculate t_index_incr INSIDE the loop (was outside in original)
            t_index_incr = Ta * time_shift
            
            history_arr = np.array(self.history[-(time_shift+1)])[:, :]
            history_current = np.array(self.history[-1])[:, :]
            
            # STEP 2: Run ensemble forecast with the last time step in the history
            xf_4d = self.CSE_Step2(self.multiple_number, self.T_0, solver, history_arr.T, das)
            t_index_shifted = self.t_index - (t_index_incr)
            
            # STEP 3: Detect regime shift
            predicted_changes, safe_members = self.CSE_Step3(xf_4d)
            
            # Check if any safe members are found:
            if len(safe_members) > 0:
                all_shifted = False
                t_index_shifted = self.t_index - (t_index_incr)
                
                # Check if we have enough data to extract perturbations
                ttt = xf_4d.shape[1] - t_index_incr
                
                # If ttt < Ta-1, we don't have enough steps for control, continue searching
                if ttt < (Ta - 1):
                    print(f'  Not enough steps available (ttt={ttt} < Ta-1={Ta-1}), continuing search...')
                    all_shifted = True  # Continue loop
                    time_shift += 1
                    if time_shift > time_shift_max:
                        Xf = xf_4d.copy()
                        return x_original, x_final_history
                    continue
                
                # Step 4: Control
                print(f'  Selecting ensemble members for control...')
                print(f' safe_members: {safe_members}')
                
                safe_idx  = self.rng.choice(safe_members)
                xf_shift  = Xf[0, 0:ttt, :]
                
                print(f' t_index_incr: {t_index_incr}')
                print(f' ttt: {ttt}')
                print(f' ttt+t_index_incr: {ttt+t_index_incr}')
                
                xf_safe = xf_4d[safe_idx, t_index_incr:ttt+t_index_incr:1, :]
                self._warn_if_step4b_safe_opposite_regime(xf_safe, regime_sign)
                self.target.append(xf_safe[0:Ta,:])
                
                # Compute perturbation vector
                perturb_vec = self.compute_perturbation(xf_safe, xf_shift, D)
                
                # Evolve NR forward with control
                temp_solver = run_RK4(solver.func, solver.sigma, solver.beta, solver.rho, x_original, self.dt, 1.5 * self.dt)
                x_next = temp_solver.integrate()[1, :]
                x_final, x_history = self.evolve_nature_with_control(x_next, perturb_vec, solver, Ta - 1)
                x_final_history.extend(x_history)
                stored_perturb = np.vstack([np.zeros((1, perturb_vec.shape[1])), perturb_vec[:Ta-1, :]])
                self._record_intervention(
                    'step4b_optimized_control',
                    stored_perturb,
                    metadata={
                        'safe_member_index': int(safe_idx),
                        'time_shift': int(time_shift),
                        't_index_incr': int(t_index_incr),
                        'D': float(D),
                    },
                )
                
                x_history_only = [x[0] for x in x_final_history]
            else:
                time_shift += 1
                if time_shift > time_shift_max:
                    # If no safe members are found after max time steps, evolve without control
                    print(f'  Max search depth ({time_shift_max}) reached. Evolving without control for Ta={Ta} steps.')
                    x_temp = x_original.copy()
                    for step in range(Ta):
                        temp_solver = run_RK4(solver.func, solver.sigma, solver.beta, solver.rho, x_temp, self.dt, 1.5 * self.dt)
                        x_temp = temp_solver.integrate()[1, :]
                        x_final_history.append(x_temp.tolist())
                    self._record_intervention('step4b_optimized_no_control', np.zeros((Ta, np.size(x_original))), metadata={'reason': 'max_search_depth', 'time_shift': int(time_shift)})
                    return x_temp, x_final_history[1:]  # Return final state (skip initial)
                    
        return x_final, x_final_history

    def CSE_Step4b_previous_analysis(self, predicted_changes, safe_members, Xf, x_original, solver, Ta, D, das):
        """
        STEP 4b alternative: all current members show a regime shift.

        Select the current member that most strongly moves away from the
        current regime as xf_shift.
        Then relaunch previous analysis ensembles, one cycle farther back at a
        time, until reaching the same final time as the current forecast. A
        member is safe only if x stays in the current regime over the full aligned
        trajectory.
        """
        print('CSE System Step 4b (PREVIOUS ANALYSIS)')
        x_final_history = []
        x_final_history.append(x_original[:].tolist())
        regime_sign = np.sign(x_original[0])
        self.last_step4b_search_attempts = []

        if len(self.history) < 2 and self.osse_Xa_history is None:
            print('  Insufficient history to use previous analysis ensemble. Evolving without control.')
            x_temp = x_original.copy()
            for step in range(Ta):
                temp_solver = run_RK4(solver.func, solver.sigma, solver.beta, solver.rho, x_temp, self.dt, 1.5 * self.dt)
                x_temp = temp_solver.integrate()[1, :]
                x_final_history.append(x_temp.tolist())
            self._record_intervention(
                'step4b_previous_analysis_no_control',
                np.zeros((Ta, np.size(x_original))),
                metadata={'reason': 'insufficient_history'},
            )
            return x_temp, x_final_history[:-1]
        if len(self.history) < 2:
            print('  Insufficient control history. Falling back to OSSE analysis history.')

        shifted_members = np.asarray(predicted_changes, dtype=int)
        if shifted_members.size == 0:
            shifted_members = np.arange(Xf.shape[0], dtype=int)
        shifted_final_x = np.asarray(Xf[shifted_members, -1, 0], dtype=float)
        shifted_final_score = regime_sign * shifted_final_x
        shift_idx = int(shifted_members[np.argmin(shifted_final_score)])
        xf_shift = Xf[shift_idx, :, :]

        max_search_depth = int(self.step4b_previous_analysis_max_search_depth)
        current_time_steps = np.arange(Xf.shape[1])
        current_shift_members = np.asarray(predicted_changes, dtype=int)
        if current_shift_members.size == 0:
            current_shift_members = np.arange(Xf.shape[0], dtype=int)

        found_safe = False
        search_phase = None
        control_history_depth = max(len(self.history) - 1, 0)
        osse_Xa_history = None if self.osse_Xa_history is None else np.asarray(self.osse_Xa_history, dtype=float)
        current_osse_index = None
        if self.osse_start_index is not None:
            current_osse_index = int(self.osse_start_index) + int(self.t_index // Ta)
        for cycles_back in range(1, max_search_depth + 1):
            osse_index = None
            if cycles_back <= control_history_depth:
                search_phase = 'control_history'
                previous_analysis = np.asarray(self.history[-(cycles_back + 1)], dtype=float)
                previous_analysis_for_forecast = self._analysis_as_xdim_edim(
                    previous_analysis,
                    das,
                    'control history previous analysis',
                    preferred_orientation='xdim_edim',
                )
            else:
                if osse_Xa_history is None:
                    print(f'  Insufficient history for cycles_back={cycles_back}. Stopping search.')
                    break
                if self.osse_start_index is None:
                    print('  OSSE analysis history is available, but osse_start_index is None. Stopping search.')
                    break
                osse_index = int(current_osse_index) - cycles_back
                if osse_index < 0 or osse_index >= osse_Xa_history.shape[0]:
                    print(f'  OSSE history exhausted for cycles_back={cycles_back} (osse_index={osse_index}). Stopping search.')
                    if self.plot_step4b_osse_history_exhausted:
                        self._plot_step4b_osse_history_exhausted(
                            Xf,
                            current_shift_members,
                            current_time_steps,
                            Ta,
                            cycles_back,
                            osse_index,
                            osse_Xa_history.shape[0],
                            control_history_depth,
                        )
                    break
                search_phase = 'osse_history'
                previous_analysis = np.asarray(osse_Xa_history[osse_index], dtype=float)
                previous_analysis_for_forecast = self._analysis_as_xdim_edim(
                    previous_analysis,
                    das,
                    'OSSE previous analysis',
                    preferred_orientation=self.osse_Xa_history_orientation or 'xdim_edim',
                )

            temporal_offset = Ta * cycles_back
            previous_forecast_steps = Xf.shape[1] + temporal_offset
            previous_forecast = self.CSE_Step2_n_steps(previous_forecast_steps, solver, previous_analysis_for_forecast, das)

            aligned_final_index = temporal_offset + Xf.shape[1] - 1
            previous_final_x = np.asarray(previous_forecast[:, aligned_final_index, 0], dtype=float)
            aligned_x = np.asarray(previous_forecast[:, temporal_offset:temporal_offset + Xf.shape[1], 0], dtype=float)
            signed_aligned_x = regime_sign * aligned_x
            previous_min_x = np.min(signed_aligned_x, axis=1)
            previous_safe_members = np.where(np.all(signed_aligned_x > 0.0, axis=1))[0]
            self.last_step4b_search_attempts.append({
                'cycles_back': int(cycles_back),
                'phase': search_phase,
                'osse_index': None if osse_index is None else int(osse_index),
                'safe_member_count': int(len(previous_safe_members)),
                'safe_members': [int(member) for member in previous_safe_members],
                'previous_start_x': np.asarray(previous_forecast[:, 0, 0], dtype=float).tolist(),
                'aligned_start_x': np.asarray(previous_forecast[:, temporal_offset, 0], dtype=float).tolist(),
                'aligned_final_x': np.asarray(previous_final_x, dtype=float).tolist(),
                'aligned_min_signed_x': np.asarray(previous_min_x, dtype=float).tolist(),
            })

            previous_plot = previous_forecast[:, :temporal_offset + Xf.shape[1], :]
            previous_time_steps = np.arange(previous_plot.shape[1]) - temporal_offset
            fig, ax = plt.subplots(figsize=(10, 5))
            for member_idx in range(previous_plot.shape[0]):
                ax.plot(
                    previous_time_steps,
                    previous_plot[member_idx, :, 0],
                    color='black',
                    linewidth=1.0,
                    alpha=0.75,
                    label='past cycle members' if member_idx == 0 else None,
                )
                ax.scatter(
                    previous_time_steps[0],
                    previous_plot[member_idx, 0, 0],
                    color='black',
                    s=28,
                    zorder=4,
                )
                ax.scatter(
                    previous_time_steps[-1],
                    previous_plot[member_idx, -1, 0],
                    color='black',
                    s=28,
                    zorder=4,
                )
            for member_idx in current_shift_members:
                ax.plot(
                    current_time_steps,
                    Xf[member_idx, :, 0],
                    color='tab:blue',
                    linestyle='-',
                    linewidth=1.3,
                    alpha=0.9,
                    label='current cycle shift members' if member_idx == current_shift_members[0] else None,
                )
                ax.scatter(
                    current_time_steps[0],
                    Xf[member_idx, 0, 0],
                    color='tab:blue',
                    edgecolors='black',
                    s=42,
                    zorder=5,
                )
                ax.scatter(
                    current_time_steps[-1],
                    Xf[member_idx, -1, 0],
                    color='tab:blue',
                    edgecolors='black',
                    s=42,
                    zorder=5,
                )
            ax.axhline(0.0, color='red', linestyle=':', linewidth=1.6, label='x = 0')
            ax.axvline(-temporal_offset, color='black', linestyle='--', linewidth=1.4)
            ax.axvline(0, color='tab:blue', linestyle='--', linewidth=1.4)
            ax.axvline(current_time_steps[-1], color='0.35', linestyle='--', linewidth=1.4)
            first_tick = int(np.floor((-temporal_offset) / 50.0) * 50)
            intermediate_ticks = np.arange(first_tick, current_time_steps[-1] + 1, 50)
            xticks = np.unique(np.concatenate(([-temporal_offset, 0], intermediate_ticks, [current_time_steps[-1]]))).astype(int)
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(tick) for tick in xticks])
            ax.set_xlim(-temporal_offset - 8, current_time_steps[-1] + 5)
            ax.text(
                -temporal_offset - 1.0,
                0.98,
                f'{cycles_back} cycle(s)\nback start',
                color='black',
                fontsize=8,
                ha='right',
                va='top',
                transform=ax.get_xaxis_transform(),
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.75, 'pad': 1.5},
            )
            ax.text(
                1.0,
                0.98,
                'current\nstart',
                color='tab:blue',
                fontsize=8,
                ha='left',
                va='top',
                transform=ax.get_xaxis_transform(),
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.75, 'pad': 1.5},
            )
            ax.text(
                current_time_steps[-1] - 1.0,
                0.98,
                'aligned\nfinal',
                color='0.35',
                fontsize=8,
                ha='right',
                va='top',
                transform=ax.get_xaxis_transform(),
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.75, 'pad': 1.5},
            )
            ax.set_xlabel('Time step (1 step = 0.01)')
            ax.set_ylabel('state x')
            safe_status = f'{len(previous_safe_members)} safe full trajectory member(s)' if len(previous_safe_members) > 0 else 'no safe full trajectories'
            ax.set_title(
                f'Past analysis ensemble aligned to current final time ({safe_status})\n'
                f'cycle={self.t_index // Ta}, cycles_back={cycles_back}, phase={search_phase}, shift_member={shift_idx}'
            )
            ax.grid(True, alpha=0.25)
            ax.legend(loc='upper right', fontsize=7, framealpha=0.82, handlelength=1.5)
            fig.tight_layout()
            plt.show()
            plt.close(fig)

            if len(previous_safe_members) > 0:
                safe_idx = int(previous_safe_members[np.argmax(previous_min_x[previous_safe_members])])
                xf_safe = previous_forecast[safe_idx, temporal_offset:temporal_offset + Xf.shape[1], :]
                self._warn_if_step4b_safe_opposite_regime(xf_safe, regime_sign)
                found_safe = True
                print(f'  Found safe member using phase={search_phase}, cycles_back={cycles_back}.')
                break

            print(f'  No safe member found for cycles_back={cycles_back}. Searching one more cycle back.')

        if not found_safe:
            raise RuntimeError(
                f'No safe member found after searching {max_search_depth} previous analysis ensembles.'
            )

        self.target.append(xf_safe[0:Ta, :])

        print(f'  shift_member_index: {shift_idx}, shift_final_x: {Xf[shift_idx, -1, 0]}')
        print(f'  safe_member_index: {safe_idx}, safe_final_x: {previous_final_x[safe_idx]}, safe_min_x: {previous_min_x[safe_idx]}')

        perturb_vec = self.compute_perturbation(xf_safe, xf_shift, D)

        temp_solver = run_RK4(solver.func, solver.sigma, solver.beta, solver.rho, x_original, self.dt, 1.5 * self.dt)
        x_next = temp_solver.integrate()[1, :]
        x_final, x_history = self.evolve_nature_with_control(x_next, perturb_vec, solver, Ta - 1)
        x_final_history.extend(x_history)

        stored_perturb = np.vstack([np.zeros((1, perturb_vec.shape[1])), perturb_vec[:Ta-1, :]])
        self._record_intervention(
            'step4b_previous_analysis_control',
            stored_perturb,
            metadata={
                'safe_member_index': int(safe_idx),
                'shift_member_index': int(shift_idx),
                'safe_final_x': float(previous_final_x[safe_idx]),
                'safe_min_x': float(previous_min_x[safe_idx]),
                'shift_final_x': float(Xf[shift_idx, -1, 0]),
                'aligned_final_index': int(aligned_final_index),
                'phase': search_phase,
                'cycles_back': int(cycles_back),
                'temporal_offset': int(temporal_offset),
                'D': float(D),
            },
        )

        return x_final, x_final_history

    #
    # Defining function to compute Lyapunov Exponents for Lorenz63 system:
    #
    def compute_LE_Lorenz63(self, x_nature, lorenz63, sigma, beta, rho, h, t_init, t_final, num_samples, FIT_END):
        #
        # Select initial condition from the nature run at a given time:
        #
        state_init = x_nature[:,t_init].copy()
        #
        # Create the RK4 solver:
        #
        solver = run_RK4(lorenz63, sigma, beta, rho, state_init, h, t_final, str_figures_path='')
        #
        # Run the solver and integrate the trajectory for the unperturbed case:
        #
        trajectory = solver.integrate()
        #
        # Now, perturb the initial condition slightly and run the solver again:
        # Generate random noise to add to the initial condition:
        noise = self.rng.normal(0, 0.0001, (num_samples, np.shape(state_init)[0]))
        #
        state_init_noisy = state_init + noise
        print('state_init_noisy shape: ', np.shape(state_init_noisy))
        # Create a new RK4 solver for each perturbed initial condition: 
        trajectory_noisy = np.zeros((num_samples, trajectory.shape[0], trajectory.shape[1]))
        for memb in range(num_samples):
            solver = run_RK4(lorenz63, sigma, beta, rho, state_init_noisy[memb,:], h, t_final, str_figures_path='')
            trajectory_noisy[memb,:] = solver.integrate()
        #
        print('trajectory_noisy shape: ', np.shape(trajectory_noisy))
        #
        # Compute distance between trajectory and trajectory_noisy at each time step for each member:
        #
        distances = np.zeros((num_samples, trajectory.shape[0]))
        for memb in range(num_samples):
            distances[memb,:] = np.linalg.norm(trajectory - trajectory_noisy[memb,:], axis=1)
        print('distances shape: ', np.shape(distances))
        #
        # Time axis:
        #
        time_axis = np.arange(distances.shape[1]) * h
        print(f'time_axis: {time_axis}')
        #
        # Compute the Lyapunov exponent as the slope of the logarithm of the distance over time for each member:
        #
        log_distances = np.zeros((num_samples, distances.shape[1]))
        for memb in range(num_samples):
            log_distances[memb,:] = np.log(distances[memb,:])
        print('log_distances shape: ', np.shape(log_distances))
        #
        # Choose a safe fit window:
        #
        Nfit = min(FIT_END, distances.shape[1])
        print(f'Nfit: {Nfit}')
        x_fit = time_axis[:Nfit]
        #
        # Compute Lyapunov exponent from ENSEMBLE MEAN distance
        # Step 1: Calculate mean distance across all trajectories
        distances_mean = np.mean(distances, axis=0)  # Shape: (time_steps,)
        #
        # Step 2: Take logarithm of mean distance
        #
        log_distances_mean = np.log(distances_mean)
        y_fit_mean = log_distances_mean[:Nfit]
        #
        # Step 3: Fit single regression line to get Lyapunov exponent
        #
        (p_mean, cov_mean) = np.polyfit(x_fit, y_fit_mean, 1, cov=True)
        lyapunov_from_mean = p_mean[0]  # Slope = Lyapunov exponent
        log_offset_mean = p_mean[1]     # Intercept
        lyapunov_std_from_mean = np.sqrt(cov_mean[0, 0])
        #
        print(f"\n✓ Lyapunov from ensemble mean distance:")
        print(f"  λ = {lyapunov_from_mean:.6f} ± {lyapunov_std_from_mean:.6f}")
        print(f"  Fit window: 0 to {Nfit} steps ({x_fit[-1]:.1f} seconds)")
        print(f"  Theoretical (Viswanath 1998): 0.9056")
        #
        # Use this value for further analysis
        #
        avg_lyapunov_exponent = lyapunov_from_mean
        print(f'Average Lyapunov Exponent: {avg_lyapunov_exponent}')
        return trajectory, trajectory_noisy, noise, distances, avg_lyapunov_exponent
        
    
    #
    # Defining function to compute the Leading LE for the Lorenz63 System:
    #
    def compute_Leading_LE_Lorenz63(self, x_nature, lorenz63, sigma, beta, rho, h, t_init, t_final, n_attractor_points, number_of_samples, n_perturbations, total_trajectories, FIT_END):
        """
        We want to reduce oscillations in the average error growth.
        Strategy: Sample 1000 distinct random points on the attractor, then for each point, 
        apply 100 different random perturbations. Total: 1000 x 100 = 100,000 trajectories.
        """
        print(f"{'='*70}")
        print(f"  Takemasa's Approach: Nested Sampling Strategy")
        print(f"  - Attractor points: {n_attractor_points}")
        print(f"  - Perturbations per point: {n_perturbations}")
        print(f"  - Total trajectories: {total_trajectories:,}")
        print(f"{'='*70}\n")
        #
        # STEP 1: Select 1000 distinct random time indices from the nature run (attractor points)
        # Make sure we have enough points in x_nature
        #
        available_times = x_nature.shape[1]
        t_indices = self.rng.choice(available_times, size=n_attractor_points, replace=False)
        print(f"Selected {len(t_indices)} random attractor points\n")
        #
        # STEP 2: For each attractor point, we'll compute 100 perturbed trajectories
        # Storage: distances[attractor_point_idx, perturbation_idx, time_step]
        #
        time_steps = int(t_final / h) + 1
        distances_all = np.zeros((n_attractor_points, n_perturbations, time_steps))
        
        # OPTIONAL: Store trajectories for plotting (only if requested to save memory)
        # We'll store a subset for visualization purposes
        store_trajectories = True  # Set to False to save memory
        if store_trajectories:
            trajectories_unpert = []  # List of unperturbed trajectories (one per attractor point)
            trajectories_pert = []    # List of lists of perturbed trajectories
        #
        # Time axis (same for all):
        #
        time_axis = np.arange(time_steps) * h
        print("Starting nested loop computation...")
        #
        # OUTER LOOP: Iterate over each attractor point
        #
        for i, t_idx in enumerate(t_indices):
            #
            # Get the initial condition from this attractor point
            #
            state_init = x_nature[:, t_idx].copy()  # Shape: (3,)
            #
            # Integrate the unperturbed trajectory from this initial condition
            #
            solver = run_RK4(lorenz63, sigma, beta, rho, state_init, h, t_final, str_figures_path='')
            trajectory_unpert = solver.integrate()  # Shape: (time_steps, 3)
            
            # Store unperturbed trajectory if requested
            if store_trajectories:
                trajectories_unpert.append(trajectory_unpert)
                pert_list = []
            #
            # INNER LOOP: Apply n_perturbations to this attractor point
            #
            for j in range(n_perturbations):
                #
                # Generate random perturbation for this initial condition
                #
                noise = self.rng.normal(0, 0.0001, size=3)
                state_init_pert = state_init + noise
                #
                # Integrate the perturbed trajectory
                #
                solver_pert = run_RK4(lorenz63, sigma, beta, rho, state_init_pert, h, t_final, str_figures_path='')
                trajectory_pert = solver_pert.integrate()  # Shape: (time_steps, 3)
                
                # Store perturbed trajectory if requested
                if store_trajectories:
                    pert_list.append(trajectory_pert)
                #
                # Compute distance at each time step
                #
                distances_all[i, j, :] = np.linalg.norm(trajectory_unpert - trajectory_pert, axis=1)
            
            # Store perturbations list for this attractor point
            if store_trajectories:
                trajectories_pert.append(pert_list)
        #
        # Computation complete message with timing
        #
        print(f"\n{'='*70}")
        print(f"✓ Computation complete!")
        print(f"  distances_all shape: {distances_all.shape}")
        print(f"  (attractor_points, perturbations, time_steps) = {distances_all.shape}")
        print(f"{'='*70}\n")
        #
        # STEP 3: Average over both dimensions (attractor points AND perturbations)
        # First reshape to combine both sampling dimensions: (1000 x 100, time_steps)
        #
        distances_flat = distances_all.reshape(-1, time_steps)  # Shape: (100000, time_steps)
        print(f"Flattened distances shape: {distances_flat.shape}")
        #
        # Compute ensemble mean distance (average over all 100,000 trajectories)
        #
        distances_mean = np.mean(distances_flat, axis=0)  # Shape: (time_steps,)
        print(f"Mean distance shape: {distances_mean.shape}")
        #
        # STEP 4: Compute Lyapunov exponent from ENSEMBLE MEAN distance
        # This is the correct approach: fit the averaged distance, not average individual fits
        #
        x_fit = time_axis[:FIT_END]
        print(f"\n{'='*70}")
        print(f"  STEP 4: Computing Lyapunov Exponent from Ensemble Mean")
        print(f"{'='*70}")
        #
        # Lyapunov exponent from ENSEMBLE MEAN (CORRECT METHOD)
        #
        log_distances_mean = np.log(distances_mean)
        y_fit_mean = log_distances_mean[:FIT_END]
        (p_mean, cov_mean) = np.polyfit(x_fit, y_fit_mean, 1, cov=True)
        lyapunov_from_mean = p_mean[0]
        lyapunov_std_from_mean = np.sqrt(cov_mean[0, 0])
        print(f"\n✓ Method 1 (CORRECT): Lyapunov from ensemble mean distance")
        print(f"  λ = {lyapunov_from_mean:.6f} ± {lyapunov_std_from_mean:.6f}")
        print(f"  Fit window: 0 to {FIT_END} steps ({x_fit[-1]:.1f} seconds)")
        #
        # Use the CORRECT method for further analysis
        #
        avg_lyapunov_exponent = lyapunov_from_mean
        print(f"\n{'='*70}")
        print(f"  FINAL RESULTS:")
        print(f"{'='*70}")
        print(f"  ✓ CORRECT: Lyapunov from ensemble mean = {lyapunov_from_mean:.6f}")
        print(f"  📖 Theoretical (Viswanath 1998)        = 0.9056")
        print(f"  ")
        print(f"  Error (correct method):  {abs(lyapunov_from_mean - 0.9056):.6f} ({abs(lyapunov_from_mean - 0.9056)/0.9056*100:.2f}%)")
        print(f"{'='*70}\n")
        #
        # Store key variables for plotting and future analysis
        #
        self.lyapunov_data = {
            'distances_all': distances_all,  # Shape: (n_attractor_points, n_perturbations, time_steps)
            't_indices': t_indices,          # Attractor point indices used
            'trajectory': x_nature,          # Reference trajectory
            'h': h,                          # Time step
            't_final': t_final,              # Total integration time
            'lyapunov_exponent': lyapunov_from_mean,
            'time_axis': time_axis
        }
        
        # Add trajectory data if it was stored
        if store_trajectories:
            self.lyapunov_data['trajectories_unpert'] = trajectories_unpert  # List of arrays
            self.lyapunov_data['trajectories_pert'] = trajectories_pert      # List of lists of arrays
            print(f"✓ Stored trajectories for plotting (memory: ~{n_attractor_points * n_perturbations * time_steps * 3 * 8 / 1e6:.1f} MB)\n")
        #
        distances = distances_flat  # For compatibility with plotting code below
        #
        return distances

    
    
    # Defining function to plot Nature Run for Lorenz63 system:
    def plot_nature_run_Lorenz63(self, x_nature, trajectory, trajectory_noisy, t_init, num_cycles, Ta, PathFigures):
        """
        Plotting Nature Run for Lorenz63 system:
        """
        #
        # Time axis for Nature Run:
        #
        time_axis = np.arange(0, num_cycles*Ta+1)
        #print(f'time_axis: {time_axis}')
        #
        # Time axis for trajectory:
        #
        time_axis_traj = np.arange(t_init, t_init + trajectory.shape[0])
        #print(f'time_axis_traj: {time_axis_traj}')
        # Choose a few members to show (to avoid spaghetti)
        M = trajectory_noisy.shape[0]
        show_k = min(4, M)        # show at most 5 members
        show_idx = np.linspace(0, M-1, show_k, dtype=int)
        #print(f'show_idx: {show_idx}')
        fig = plt.figure(figsize=(12,6))
        plt.plot(time_axis, x_nature[0,0:num_cycles*Ta+1], label='Nature Run (X)', color='blue')
        plt.plot(time_axis_traj, trajectory[:,0], label='Unperturbed Nature Run (X)', color='black', linestyle='dashed')
        for memb in show_idx:
            if memb == 0:
                plt.plot(time_axis_traj, trajectory_noisy[memb,:,0], label='Perturbed Trajectories (X)', color='green', linestyle='dashed')
            else:
                plt.plot(time_axis_traj, trajectory_noisy[memb,:,0], color='green', linestyle='dashed', alpha=0.35)
        # Highlight the starting point:
        plt.scatter(time_axis_traj[0], trajectory[0,0], color='orange', edgecolor = 'black', s=20, zorder=5, label=f'Starting Point (t={t_init})')
        plt.title('Nature Run of Lorenz-63 System (X Dimension) \n (Perturbed vs Unperturbed forecasts)', fontsize=16)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('X Value', fontsize=14)
        plt.xlim(0, num_cycles*Ta)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.show()
        #fig.savefig(PathFigures + 'NatureRun_Lorenz63_Z_Dimension_Perturbed_vs_Unperturbed.png', dpi=300)
        plt.close()




    #
    # Defining function to plot random initial points:
    #
    def plot_random_initial_points(self, field, num_rand_points, ylim_inf, ylim_sup, window, title_label, title_main, title_figure, PathFigures):
        time_axis = np.arange(0,window)
        available_times = field[0:window].shape[0]
        t_indices = self.rng.choice(available_times, size=num_rand_points, replace=False)
        print(f"Selected {len(t_indices)} random attractor points\n")
        fig = plt.figure(figsize=(14, 5))
        plt.plot(time_axis, field[0:window], label=title_label, color='green')
        # Highlight the starting point:
        plt.scatter(t_indices, field[t_indices], color='orange', edgecolor='black', s=20, zorder=5, label=f'Random Starting Points')
        plt.title(title_main, fontsize=16)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('X Value', fontsize=14)
        plt.xlim(0, window)
        plt.ylim(ylim_inf, ylim_sup)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()
        fig.savefig(self.figures_path + title_figure, dpi=300)
        plt.close()



    # Defining function to plot norm distances for Lorenz63 system in semilog scale:
    def plot_norm_distances_Lorenz63(self, distances, FIT_END, avg_lyapunov_exponent, PathFigures):
        """
        Plotting norm distances for Lorenz63 system in semilog scale:
        """
        fig = plt.figure(figsize=(12,6))
        num_trajectories = len(distances)
        sample_step = max(1, num_trajectories // 20)  # Show ~20 sample trajectories
        print(f'Total trajectories: {num_trajectories}, showing every {sample_step}th trajectory')
        for memb in range(0, num_trajectories, sample_step):
            if memb == 0:
                plt.semilogy(distances[memb,:], color='wheat', label='Individual Members', alpha=0.35)
            else:
                plt.semilogy(distances[memb,:], color='wheat', alpha=0.35)
        # plotting ensemble mean distance:
        plt.semilogy(np.mean(distances, axis=0), label='Ensemble Mean Distance', color='orange', linewidth = 2)
        # Vertical line marking end of fit window:
        plt.vlines(x=FIT_END, ymin=1e-9, ymax=1e+2, colors='gray', linestyles='dashed', color = 'gray', label='End of Fit Window')
        # Display average Lyapunov exponent:
        print(f'Average Lyapunov Exponent: {avg_lyapunov_exponent:.6f}')
        print(f'Number of trajectories: {num_trajectories}')
        print(f'FIT_END: {FIT_END}')
        plt.title('Distance between Unperturbed and Perturbed Trajectories over Time\n (Perturbations from same initial conditions)', fontsize=16)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Distance norm (Log Scale)', fontsize=14)
        #plt.xlim(t_init, t_init + num_cycles*Ta)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.show()
        #fig.savefig(PathFigures + 'Lyapunov_Exponent_Lorenz63.png', dpi=300)
        plt.close()


    #
    # Defining function to plot norm distances and Leading Lyapunov Exponent Lorenz63 System:
    #
    def plot_norm_distances_LLE_Lorenz63(self, distances, h, FIT_END, PathFigures, trajectory_type='nature'):
        """
        Plot mean distance evolution and compute Lyapunov exponent.
        
        Parameters:
        -----------
        distances : np.ndarray
            Distance array, shape (total_trajectories, time_steps)
        h : float
            Integration step size
        FIT_END : int
            Time step index for end of regression fit window
        PathFigures : str
            Path to save figures
        trajectory_type : str
            Type of trajectory for filename ('nature', 'target', or custom label)
        """
        distance_mean = np.mean(distances, axis=0)
        print(f'distance_mean shape: {np.shape(distance_mean)}')
        # Recreate time_axis to match distance_mean size (using time steps, not seconds)
        time_axis_plot = np.arange(len(distance_mean))
        print(f'time_axis_plot shape: {np.shape(time_axis_plot)}')
        # Compute Lyapunov exponent from ensemble mean distance
        print(f'Fit window end at time index: {FIT_END}')
        # Take logarithm of mean distances
        log_distance_mean = np.log(distance_mean)
        print(f'log_distance_mean shape: {np.shape(log_distance_mean)}')
        # Prepare data for linear regression (first FIT_END points)
        # For regression, we need actual time in seconds
        x_fit_seconds = np.arange(FIT_END) * h
        x_fit_indices = np.arange(FIT_END)
        y_fit = log_distance_mean[:FIT_END]
        # Perform linear regression: log(distance) = λ * time + c
        # Use time in seconds for the regression to get correct Lyapunov exponent
        (p, cov) = np.polyfit(x_fit_seconds, y_fit, 1, cov=True)
        lyapunov_exponent = p[0]  # Slope = Lyapunov exponent
        log_offset = p[1]         # Intercept
        print(f'Lyapunov Exponent from mean distance: {lyapunov_exponent:.6f}')
        print(f'Intercept (log scale): {log_offset:.6f}')
        #
        # Generate fitted line in original scale (exponential)
        #
        fitted_line_regression = np.exp(log_offset + lyapunov_exponent * x_fit_seconds)
        #
        # Create the plot
        #
        fig = plt.figure(figsize=(12,6))
        #
        # Plot ensemble mean distance in log scale (using time step indices)
        #
        plt.semilogy(time_axis_plot, distance_mean, label='Ensemble Mean Distance', color='orange', linewidth=2)
        #
        # Plot the fitted regression line (only over fit window, using time step indices)
        #
        plt.semilogy(x_fit_indices, fitted_line_regression, label=f'Linear Regression Fit (LE = {lyapunov_exponent:.4f})', 
                    color='red', linestyle='--', linewidth=2)
        #
        # Mark the end of the fit window
        #
        plt.vlines(x=x_fit_indices[-1], ymin=1e-9, ymax=1e+2, colors='gray', linestyles='dashed', 
                linewidth=1.5, label=f'End of Fit Window (step={FIT_END})')
        #
        # Labels and formatting
        #
        plt.title(f'Mean Distance between Unperturbed and Perturbed Trajectories\n ({trajectory_type})', fontsize=16)
        plt.xlabel('Model Time Step', fontsize=14)
        plt.ylabel('Distance norm (Log Scale)', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.show()
        #
        # Save figure with parameters in filename
        #
        total_trajectories = distances.shape[0]
        filename = f'Lyapunov_Exponent_Mean_Distance_{trajectory_type}_h_{h}_traj_{total_trajectories}.png'
        fig.savefig(PathFigures + filename, dpi=300)
        print(f'Figure saved to: {PathFigures}{filename}')
        plt.close()
        


    #
    # Defining function to plot perturbed trajectories from multiple attractor points:
    #
    def Plot_Perturbed_Trajectories_From_Attractor_Points(self, trajectory, lorenz63, sigma, beta, rho, h, 
                                                          t_final, n_points_to_plot=5, n_perturbations=10, 
                                                          plot_3d=True, PathFigures=''):
        """
        Visualize the evolution of perturbed trajectories from multiple attractor points.
        
        Parameters:
        -----------
        trajectory : np.ndarray
            Reference trajectory, shape (3, time_steps) - can be x_nature or target
        lorenz63 : function
            Lorenz63 dynamics function
        sigma, beta, rho : float
            Lorenz63 parameters
        h : float
            Integration time step
        t_final : float
            Total integration time for each perturbed trajectory
        n_points_to_plot : int
            Number of attractor points to sample and visualize
        n_perturbations : int
            Number of perturbations to apply at each attractor point
        plot_3d : bool
            If True, creates 3D phase space plot. If False, creates time series plots
        PathFigures : str
            Path to save figures
        """
        print(f"{'='*70}")
        print(f"  Plotting Perturbed Trajectories from Attractor Points")
        print(f"{'='*70}\n")
        
        # Sample random attractor points
        available_times = trajectory.shape[1]
        t_indices = self.rng.choice(available_times, size=n_points_to_plot, replace=False)
        t_indices = np.sort(t_indices)  # Sort for easier visualization
        
        print(f"Sampling {n_points_to_plot} attractor points:")
        for i, t_idx in enumerate(t_indices):
            print(f"  Point {i+1}: time index {t_idx}, state = [{trajectory[0,t_idx]:.3f}, {trajectory[1,t_idx]:.3f}, {trajectory[2,t_idx]:.3f}]")
        
        # Integration parameters
        time_steps = int(t_final / h) + 1
        time_axis = np.arange(time_steps) * h
        
        # Store all trajectories for plotting
        all_trajectories = []
        attractor_points = []
        
        print(f"\nComputing {n_perturbations} perturbed trajectories from each point...")
        
        for i, t_idx in enumerate(t_indices):
            # Get initial condition from attractor point
            state_init = trajectory[:, t_idx].copy()
            attractor_points.append(state_init)
            
            # Compute unperturbed trajectory
            solver_unpert = run_RK4(lorenz63, sigma, beta, rho, state_init, h, t_final, str_figures_path='')
            traj_unpert = solver_unpert.integrate()  # Shape: (time_steps, 3)
            
            # Store unperturbed trajectory
            trajs_from_point = {'unperturbed': traj_unpert, 'perturbed': []}
            
            # Compute perturbed trajectories
            for j in range(n_perturbations):
                # Add small random perturbation
                perturbation = self.rng.normal(0, 0.0001, size=3)
                state_pert = state_init + perturbation
                
                # Integrate perturbed trajectory
                solver_pert = run_RK4(lorenz63, sigma, beta, rho, state_pert, h, t_final, str_figures_path='')
                traj_pert = solver_pert.integrate()
                
                trajs_from_point['perturbed'].append(traj_pert)
            
            all_trajectories.append(trajs_from_point)
            print(f"  ✓ Point {i+1}/{n_points_to_plot} complete")
        
        print(f"\n{'='*70}")
        print(f"  Creating Visualizations")
        print(f"{'='*70}\n")
        
        if plot_3d:
            # Create 3D phase space plot
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot reference trajectory (attractor) - subsample for visualization
            # Use every Nth point to avoid memory overflow with matplotlib
            subsample = max(1, trajectory.shape[1] // 5000)  # Limit to ~5000 points
            ax.plot(trajectory[0,::subsample], trajectory[1,::subsample], trajectory[2,::subsample], 
                   color='lightgray', linewidth=0.5, alpha=0.3, label='Reference Trajectory (subsampled)')
            
            # Color map for different attractor points
            colors = plt.cm.rainbow(np.linspace(0, 1, n_points_to_plot))
            
            for i, (trajs, color) in enumerate(zip(all_trajectories, colors)):
                # Plot attractor point
                ax.scatter(attractor_points[i][0], attractor_points[i][1], attractor_points[i][2],
                          color='black', s=100, marker='o', edgecolors=color, linewidths=3, 
                          zorder=10, label=f'Attractor Point {i+1}' if i < 3 else '')
                
                # Plot unperturbed trajectory
                traj_u = trajs['unperturbed']
                ax.plot(traj_u[:, 0], traj_u[:, 1], traj_u[:, 2],
                       color=color, linewidth=2.5, alpha=0.9, zorder=5)
                
                # Plot perturbed trajectories
                for j, traj_p in enumerate(trajs['perturbed']):
                    ax.plot(traj_p[:, 0], traj_p[:, 1], traj_p[:, 2],
                           color=color, linewidth=1.0, alpha=0.3, zorder=3)
            
            ax.set_xlabel('X', fontsize=14)
            ax.set_ylabel('Y', fontsize=14)
            ax.set_zlabel('Z', fontsize=14)
            ax.set_title(f'Perturbed Trajectories from {n_points_to_plot} Attractor Points\n' + 
                        f'({n_perturbations} perturbations per point, t_final={t_final}s, h={h})',
                        fontsize=16, pad=20)
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            if PathFigures:
                fig.savefig(PathFigures + f'Perturbed_Trajectories_3D_n{n_points_to_plot}_pert{n_perturbations}.png', 
                           dpi=300, bbox_inches='tight')
                print(f"✓ 3D plot saved: Perturbed_Trajectories_3D_n{n_points_to_plot}_pert{n_perturbations}.png")
            plt.show()
            plt.close()
        
        else:
            # Create time series plots (X, Y, Z components)
            fig, axes = plt.subplots(3, 1, figsize=(14, 12))
            colors = plt.cm.rainbow(np.linspace(0, 1, n_points_to_plot))
            
            component_names = ['X', 'Y', 'Z']
            
            for comp_idx, ax in enumerate(axes):
                for i, (trajs, color) in enumerate(zip(all_trajectories, colors)):
                    # Plot unperturbed trajectory
                    traj_u = trajs['unperturbed']
                    ax.plot(time_axis, traj_u[:, comp_idx], color=color, linewidth=2.5, 
                           alpha=0.9, label=f'Point {i+1} (unperturbed)' if comp_idx == 0 else '')
                    
                    # Plot perturbed trajectories
                    for j, traj_p in enumerate(trajs['perturbed']):
                        ax.plot(time_axis, traj_p[:, comp_idx], color=color, linewidth=1.0, 
                               alpha=0.2)
                
                ax.set_ylabel(f'{component_names[comp_idx]} component', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
                
                if comp_idx == 0:
                    ax.legend(loc='upper right', fontsize=10, ncol=2)
                    ax.set_title(f'Perturbed Trajectories from {n_points_to_plot} Attractor Points\n' +
                               f'({n_perturbations} perturbations per point, solid=unperturbed)',
                               fontsize=14)
                
                if comp_idx == 2:
                    ax.set_xlabel('Time (seconds)', fontsize=12)
            
            plt.tight_layout()
            if PathFigures:
                fig.savefig(PathFigures + f'Perturbed_Trajectories_TimeSeries_n{n_points_to_plot}_pert{n_perturbations}.png',
                           dpi=300, bbox_inches='tight')
                print(f"✓ Time series plot saved: Perturbed_Trajectories_TimeSeries_n{n_points_to_plot}_pert{n_perturbations}.png")
            plt.show()
            plt.close()
        
        print(f"\n{'='*70}")
        print(f"  Plotting Complete!")
        print(f"{'='*70}\n")
        
        return all_trajectories, attractor_points
    
    
    #
    # Plotting perturbed trajectories from Lyapunov calculation data (NO recalculation)
    #
    def Plot_Lyapunov_Perturbations_From_Data(self, attractor_point_index=None, 
                                               n_perturbations_to_plot=50,
                                               window_margin=100, PathFigures=''):
        """
        Plot perturbed trajectories using data already computed by compute_Leading_LE_Lorenz63.
        Does NOT recalculate anything - just visualizes existing data.
        
        Parameters:
        -----------
        attractor_point_index : int or None
            If specified, plot trajectories from this specific attractor point index (0 to n_attractor_points-1).
            If None, plots from first 5 attractor points.
        n_perturbations_to_plot : int
            Number of perturbations to show per attractor point (max: n_perturbations from calculation)
        window_margin : int
            Number of time steps to show before the initial point for context
        PathFigures : str
            Path to save figures
        """
        # Check if Lyapunov data exists
        if not hasattr(self, 'lyapunov_data'):
            raise ValueError("No Lyapunov data found! Run compute_Leading_LE_Lorenz63() first.")
        
        if 'trajectories_unpert' not in self.lyapunov_data:
            raise ValueError("Trajectory data not stored during computation! " + 
                           "Set store_trajectories=True in compute_Leading_LE_Lorenz63().")
        
        print(f"{'='*70}")
        print(f"  Plotting Perturbed Trajectories from Lyapunov Data (NO Recalculation)")
        print(f"{'='*70}\n")
        
        # Extract stored data
        t_indices = self.lyapunov_data['t_indices']
        trajectory = self.lyapunov_data['trajectory']
        h = self.lyapunov_data['h']
        trajectories_unpert = self.lyapunov_data['trajectories_unpert']
        trajectories_pert = self.lyapunov_data['trajectories_pert']
        
        n_attractor_points = len(trajectories_unpert)
        n_perturbations = len(trajectories_pert[0])
        time_steps = trajectories_unpert[0].shape[0]
        
        print(f"Available data:")
        print(f"  - Attractor points computed: {n_attractor_points}")
        print(f"  - Perturbations per point: {n_perturbations}")
        print(f"  - Time steps: {time_steps}")
        print(f"  - Integration time: {(time_steps-1) * h:.1f} seconds\n")
        
        # Determine which attractor point(s) to plot
        if attractor_point_index is not None:
            if attractor_point_index < 0 or attractor_point_index >= n_attractor_points:
                raise ValueError(f"attractor_point_index must be between 0 and {n_attractor_points-1}")
            point_indices = [attractor_point_index]
            print(f"Plotting single attractor point: index {attractor_point_index}")
            print(f"  Time index in trajectory: t = {t_indices[attractor_point_index]}")
        else:
            # Plot first 5 attractor points (or fewer if not available)
            n_to_plot = min(5, n_attractor_points)
            point_indices = list(range(n_to_plot))
            print(f"Plotting {n_to_plot} attractor points")
        
        # Limit perturbations to plot
        n_pert_plot = min(n_perturbations_to_plot, n_perturbations)
        print(f"Showing {n_pert_plot} perturbations per point\n")
        
        # Create figure
        fig = plt.figure(figsize=(14, 6))
        
        # Determine plot window
        n_points = len(point_indices)
        t_plot_indices = [t_indices[i] for i in point_indices]
        
        if n_points == 1:
            t_min = max(0, t_plot_indices[0] - window_margin)
            t_max = min(trajectory.shape[1], t_plot_indices[0] + time_steps + 50)
        else:
            t_min = max(0, min(t_plot_indices) - window_margin)
            t_max = min(trajectory.shape[1], max(t_plot_indices) + time_steps + 50)
        
        # Plot reference trajectory
        time_window = np.arange(t_min, t_max)
        plt.plot(time_window, trajectory[0, t_min:t_max], color='green', linewidth=1.5,
                alpha=0.6, label='Reference Trajectory (X component)', zorder=1)
        
        # Color map
        if n_points == 1:
            colors = ['deeppink']
        else:
            colors = plt.cm.rainbow(np.linspace(0, 1, n_points))
        
        # Plot each attractor point and its perturbations
        for idx, (point_idx, color) in enumerate(zip(point_indices, colors)):
            t_idx = t_indices[point_idx]
            state_init = trajectory[:, t_idx]
            
            # Mark initial point
            plt.scatter(t_idx, state_init[0], color='orange', s=100,
                       edgecolors='black', linewidths=1, zorder=10,
                       label=f'Initial Point (t={t_idx})' if n_points == 1 else f'Point {idx+1}')
            
            # Get unperturbed trajectory
            traj_unpert = trajectories_unpert[point_idx]
            time_forecast = np.arange(t_idx, t_idx + time_steps)
            
            # Plot unperturbed trajectory
            plt.plot(time_forecast, traj_unpert[:, 0], color='blue', linewidth=1.2,
                    alpha=0.9, zorder=5, label='Unperturbed Forecast' if idx == 0 else '')
            
            # Plot perturbed trajectories
            for j in range(n_pert_plot):
                traj_pert = trajectories_pert[point_idx][j]
                
                if idx == 0 and j == 0:
                    plt.plot(time_forecast, traj_pert[:, 0], color=color, linewidth=1.0,
                            alpha=0.3, zorder=3,
                            label=f'Perturbed Forecasts ({n_pert_plot} members)')
                else:
                    plt.plot(time_forecast, traj_pert[:, 0], color=color, linewidth=1.0,
                            alpha=0.3, zorder=3)
            
            print(f"  ✓ Point {idx+1}/{n_points} plotted (t={t_idx})")
        
        # Formatting
        plt.axhline(0, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('X component', fontsize=12)
        
        if n_points == 1:
            plt.title(f'Perturbed Trajectories from Lyapunov Data (Point t={t_plot_indices[0]})\n' +
                     f'({n_pert_plot} perturbations)',
                     fontsize=14)
        else:
            plt.title(f'Perturbed Trajectories from Lyapunov Data ({n_points} Points)\n' +
                     f'({n_pert_plot} perturbations per point)',
                     fontsize=14)
        
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', fontsize=9)
        plt.xlim(t_min, t_max)
        plt.tight_layout()
        
        if PathFigures:
            if n_points == 1:
                filename = f'Lyapunov_Perturbations_Point_{attractor_point_index}_t{t_plot_indices[0]}.png'
            else:
                filename = f'Lyapunov_Perturbations_{n_points}points.png'
            fig.savefig(PathFigures + filename, dpi=300, bbox_inches='tight')
            print(f"\n✓ Figure saved: {filename}")
        
        plt.show()
        plt.close()
        
        print(f"\n{'='*70}")
        print(f"  Plotting Complete!")
        print(f"{'='*70}\n")


    #
    # Compare Lyapunov exponents: TARGET vs NATURE (or TARGET only)
    #
    def Compare_Lyapunov_Exponents(self, 
                                    # NATURE data (always optional - ignored if plot_mode='target_only')
                                    distances_nature=None, lambda_nature=None, p_nature=None, 
                                    h_nature=None, FIT_END_nature=None,
                                    # TARGET data
                                    h_target=None, FIT_END_target=None,
                                    # Theoretical and output
                                    lambda_theoretical=0.9056,
                                    PathFigures='', 
                                    plot_mode='comparison'):
        """
        Calculate and visualize Lyapunov exponents for TARGET (and optionally compare with NATURE).
        
        SUPER SIMPLE USAGE - Just change plot_mode!
        ============================================
        You can ALWAYS pass the same parameters. The function automatically:
        - Uses NATURE data if plot_mode='comparison'
        - Ignores NATURE data if plot_mode='target_only'
        
        Example - ONE CALL WORKS FOR BOTH MODES:
        ----------------------------------------
        # Define parameters once
        params = dict(
            distances_nature=distances_LLE,
            lambda_nature=lambda_nature,
            p_nature=p_nature,
            h_nature=h,
            FIT_END_nature=FIT_END,
            h_target=h_test,
            FIT_END_target=FIT_END_test,
            lambda_theoretical=0.9056,
            PathFigures=PathFigures
        )
        
        # Comparison mode - uses NATURE
        result = cse_system.Compare_Lyapunov_Exponents(**params, plot_mode='comparison')
        
        # Target only - ignores NATURE (same params!)
        result = cse_system.Compare_Lyapunov_Exponents(**params, plot_mode='target_only')
        
        Parameters:
        -----------
        distances_nature : ndarray, optional
            Flattened distances from NATURE run. Used only if plot_mode='comparison'
        lambda_nature : float, optional
            Computed Lyapunov exponent for NATURE. Used only if plot_mode='comparison'
        p_nature : ndarray, optional
            Polynomial fit coefficients for NATURE. Used only if plot_mode='comparison'
        h_nature : float, optional
            Integration step size for NATURE. Used only if plot_mode='comparison'
        FIT_END_nature : int, optional
            Fit window end for NATURE. Used only if plot_mode='comparison'
        h_target : float, optional
            Integration step size for TARGET (default: from lyapunov_data)
        FIT_END_target : int, optional
            Fit window end for TARGET (default: 100 or FIT_END_nature)
        lambda_theoretical : float
            Theoretical Lyapunov exponent (default: 0.9056 from Viswanath 1998)
        PathFigures : str
            Path to save figures
        plot_mode : str
            'comparison' : Plot NATURE vs TARGET (requires NATURE parameters)
            'target_only' : Plot only TARGET (ignores NATURE parameters)
            
        Returns:
        --------
        dict : Dictionary containing TARGET results
        """
        
        # Check if Lyapunov data exists
        if not hasattr(self, 'lyapunov_data'):
            raise ValueError("No Lyapunov data found! Run compute_Leading_LE_Lorenz63() first.")
        
        if 'distances_all' not in self.lyapunov_data:
            raise ValueError("distances_all not found in lyapunov_data!")
        
        # Validate plot_mode
        if plot_mode not in ['comparison', 'target_only']:
            raise ValueError("plot_mode must be 'comparison' or 'target_only'")
        
        # ===================================================================
        # VALIDATE NATURE DATA only if comparison mode
        # ===================================================================
        if plot_mode == 'comparison':
            # Check if NATURE data is provided
            if distances_nature is None or lambda_nature is None or p_nature is None:
                raise ValueError(
                    "comparison mode requires NATURE data!\n"
                    "Must provide: distances_nature, lambda_nature, p_nature, h_nature, FIT_END_nature"
                )
            if h_nature is None or FIT_END_nature is None:
                raise ValueError("comparison mode requires: h_nature and FIT_END_nature")
        
        # ===================================================================
        # TARGET PARAMETERS
        # ===================================================================
        if h_target is None:
            h_target = self.lyapunov_data['h']
        if FIT_END_target is None:
            if plot_mode == 'comparison' and FIT_END_nature is not None:
                FIT_END_target = FIT_END_nature
            else:
                FIT_END_target = 100
        
        # Print header
        print("="*70)
        if plot_mode == 'comparison':
            print("  Calculating Lyapunov Exponents: TARGET vs NATURE")
        else:
            print("  Calculating Lyapunov Exponent: TARGET Only")
        print("="*70)
        
        # Extract TARGET data
        distances_all_target = self.lyapunov_data['distances_all']
        
        print(f"\nDATA SOURCES:")
        if plot_mode == 'comparison':
            print(f"  NATURE: distances array provided")
        print(f"  TARGET: from self.lyapunov_data['distances_all']")
        print(f"    - Shape: {distances_all_target.shape}")
        
        n_attractor_points_target = distances_all_target.shape[0]
        n_perturbations_target = distances_all_target.shape[1]
        time_steps_target = distances_all_target.shape[2]
        
        # Flatten TARGET distances
        distances_flat_target = distances_all_target.reshape(-1, time_steps_target)
        
        print(f"\nPARAMETERS:")
        print(f"  TARGET:")
        print(f"    - Attractor points: {n_attractor_points_target}")
        print(f"    - Perturbations per point: {n_perturbations_target}")
        print(f"    - Time steps: {time_steps_target}")
        print(f"    - Total trajectories: {n_attractor_points_target * n_perturbations_target}")
        
        if plot_mode == 'comparison':
            print(f"  NATURE:")
            print(f"    - Total trajectories: {distances_nature.shape[0]}")
            print(f"    - Time steps: {distances_nature.shape[1]}")
        
        # Calculate ensemble mean for TARGET
        distance_mean_target = np.mean(distances_flat_target, axis=0)
        log_distance_target = np.log(distance_mean_target)
        
        # Calculate mean for NATURE (if comparison mode)
        if plot_mode == 'comparison':
            distance_mean_nature = np.mean(distances_nature, axis=0)
            log_distance_nature = np.log(distance_mean_nature)
        
        # Linear fit for TARGET
        x_fit_seconds_target = np.arange(0, FIT_END_target) * h_target
        y_fit_target = log_distance_target[:FIT_END_target]
        p_target = np.polyfit(x_fit_seconds_target, y_fit_target, 1)
        lambda_target = p_target[0]
        
        print(f"\nLYAPUNOV EXPONENT CALCULATION:")
        print(f"  TARGET:")
        print(f"    - Fit window: 0 to {FIT_END_target} steps ({FIT_END_target * h_target:.1f} seconds)")
        print(f"    - Slope (λ): {lambda_target:.6f}")
        print(f"    - Intercept: {p_target[1]:.6f}")
        
        if plot_mode == 'comparison':
            print(f"  NATURE:")
            print(f"    - Fit window: 0 to {FIT_END_nature} steps ({FIT_END_nature * h_nature:.1f} seconds)")
            print(f"    - Slope (λ): {lambda_nature:.6f}")
            print(f"    - Intercept: {p_nature[1]:.6f}")
        
        print(f"\n{'='*70}")
        print(f"  FINAL RESULTS")
        print(f"{'='*70}")
        print(f"  λ_TARGET        = {lambda_target:.6f}")
        
        if plot_mode == 'comparison':
            print(f"  λ_NATURE        = {lambda_nature:.6f}")
            print(f"  λ_theoretical   = {lambda_theoretical:.4f}")
            print(f"\n  Difference (TARGET - NATURE): {lambda_target - lambda_nature:.6f}")
            print(f"  Ratio (TARGET/NATURE):        {lambda_target/lambda_nature:.3f}")
            print(f"  TARGET is {(1 - lambda_target/lambda_nature)*100:.1f}% lower than NATURE")
        else:
            print(f"  λ_theoretical   = {lambda_theoretical:.4f}")
            print(f"\n  Difference (TARGET - theoretical): {lambda_target - lambda_theoretical:.6f}")
            print(f"  Ratio (TARGET/theoretical):        {lambda_target/lambda_theoretical:.3f}")
        
        print(f"{'='*70}\n")
        
        # ========== VISUALIZATION ==========
        if plot_mode == 'comparison':
            # TWO-PANEL COMPARISON: NATURE vs TARGET
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Panel 1: Log(distance) comparison
            ax1 = axes[0]
            
            time_steps_nature = np.arange(len(log_distance_nature))
            time_steps_target = np.arange(len(log_distance_target))
            
            # Convert log distances back to actual distances for semilogy
            distance_mean_nature_actual = np.exp(log_distance_nature)
            distance_mean_target_actual = np.exp(log_distance_target)
            
            # Data - use semilogy for log scale y-axis
            ax1.semilogy(time_steps_nature, distance_mean_nature_actual, 
                     color='blue', linewidth=2, alpha=0.7, label='NATURE (uncontrolled)')
            ax1.semilogy(time_steps_target, distance_mean_target_actual, 
                     color='red', linewidth=2, alpha=0.7, label='TARGET (CSE controlled)')
            
            # Fit lines - convert to model steps for plotting but keep slope calculation in seconds
            time_seconds_fit_nature = np.arange(FIT_END_nature) * h_nature
            time_seconds_fit_target = np.arange(FIT_END_target) * h_target
            y_fit_nature_full = p_nature[0] * time_seconds_fit_nature + p_nature[1]
            y_fit_target_full = p_target[0] * time_seconds_fit_target + p_target[1]
            
            # Convert log fit lines back to actual distances
            y_fit_nature_actual = np.exp(y_fit_nature_full)
            y_fit_target_actual = np.exp(y_fit_target_full)
            
            ax1.semilogy(np.arange(FIT_END_nature), y_fit_nature_actual,
                     'b--', linewidth=2.5, alpha=0.9,
                     label=f'Fit NATURE: λ={lambda_nature:.4f}')
            
            ax1.semilogy(np.arange(FIT_END_target), y_fit_target_actual,
                     'r--', linewidth=2.5, alpha=0.9,
                     label=f'Fit TARGET: λ={lambda_target:.4f}')
            
            # Fit windows using model steps
            # NATURE: 0 to FIT_END_nature steps 
            # TARGET: 0 to FIT_END_target steps
            ax1.axvspan(0, FIT_END_nature, alpha=0.15, color='blue', 
                        label=f'NATURE fit window (0-{FIT_END_nature} steps)')
            ax1.axvspan(0, FIT_END_target, alpha=0.15, color='red', 
                        label=f'TARGET fit window (0-{FIT_END_target} steps)')
            
            ax1.set_xlabel('Model Time Step', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Distance (log scale)', fontsize=12, fontweight='bold')
            # Add ensemble size info to title
            title_comparison = f'Perturbation Growth: NATURE vs TARGET\n'
            title_comparison += f'Attractor points: {n_attractor_points_target} | Perturbations: {n_perturbations_target}'
            ax1.set_title(title_comparison, fontsize=14, fontweight='bold')
            ax1.legend(loc='lower right', fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Panel 2: Bar comparison
            ax2 = axes[1]
            
            lambdas = [lambda_nature, lambda_target, lambda_theoretical]
            labels = ['NATURE\n(uncontrolled)', 'TARGET\n(CSE controlled)', 
                      'Theoretical\n(Viswanath 1998)']
            colors_bar = ['blue', 'red', 'green']
            
            bars = ax2.bar(labels, lambdas, color=colors_bar, alpha=0.7, 
                           edgecolor='black', linewidth=2)
            
            # Values on bars
            for i, (bar, val) in enumerate(zip(bars, lambdas)):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                         f'{val:.4f}',
                         ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            ax2.set_ylabel('Leading Lyapunov Exponent (λ)', fontsize=12, fontweight='bold')
            
            # Title with trajectory count
            n_traj = n_attractor_points_target * n_perturbations_target
            ax2.set_title(f'Lyapunov Exponent Comparison\n({n_traj} trajectories)', 
                          fontsize=14, fontweight='bold')
            ax2.grid(True, axis='y', alpha=0.3)
            ax2.set_ylim(0, max(lambdas) * 1.2)
            
            # Theoretical line
            ax2.axhline(lambda_theoretical, color='green', linestyle='--', 
                        linewidth=2, alpha=0.5, label='Theoretical')
            
            plt.tight_layout()
            
            if PathFigures:
                filename = f'Lyapunov_Comparison_NATURE_vs_TARGET_{n_traj}traj.png'
                fig.savefig(PathFigures + filename, dpi=300, bbox_inches='tight')
                print(f"✓ Figure saved: {filename}")
            
            plt.show()
            plt.close()
            
        else:  # plot_mode == 'target_only'
            # TWO-PANEL TARGET-ONLY: Growth + Bar comparison
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Panel 1: Log(distance) for TARGET only
            ax1 = axes[0]
            
            time_steps_target = np.arange(len(log_distance_target))
            
            # Convert log distances back to actual distances for semilogy
            distance_mean_target_actual = np.exp(log_distance_target)
            
            # Data - use semilogy for log scale y-axis
            ax1.semilogy(time_steps_target, distance_mean_target_actual, 
                     color='red', linewidth=2, alpha=0.7, label='TARGET (CSE controlled)')
            
            # Fit line - use time in seconds for calculation, steps for plotting
            time_seconds_fit = np.arange(FIT_END_target) * h_target
            y_fit_target_full = p_target[0] * time_seconds_fit + p_target[1]
            
            # Convert log fit line back to actual distances
            y_fit_target_actual = np.exp(y_fit_target_full)
            
            ax1.semilogy(np.arange(FIT_END_target), y_fit_target_actual,
                     'r--', linewidth=2.5, alpha=0.9,
                     label=f'Fit TARGET: λ={lambda_target:.4f}')
            
            # Fit window
            ax1.axvspan(0, FIT_END_target, alpha=0.1, color='red', 
                        label=f'Fit window (0-{FIT_END_target} steps)')
            
            ax1.set_xlabel('Model Time Step', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Distance (log scale)', fontsize=12, fontweight='bold')
            # Add ensemble size info to title
            title_target = f'Perturbation Growth: TARGET (CSE controlled)\n'
            title_target += f'Attractor points: {n_attractor_points_target} | Perturbations: {n_perturbations_target}'
            ax1.set_title(title_target, fontsize=14, fontweight='bold')
            ax1.legend(loc='lower right', fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Panel 2: Bar comparison with theoretical
            ax2 = axes[1]
            
            lambdas = [lambda_target, lambda_theoretical]
            labels = ['TARGET\n(CSE controlled)', 'Theoretical\n(Viswanath 1998)']
            colors_bar = ['red', 'green']
            
            bars = ax2.bar(labels, lambdas, color=colors_bar, alpha=0.7, 
                           edgecolor='black', linewidth=2)
            
            # Values on bars
            for i, (bar, val) in enumerate(zip(bars, lambdas)):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                         f'{val:.4f}',
                         ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            ax2.set_ylabel('Leading Lyapunov Exponent (λ)', fontsize=12, fontweight='bold')
            
            # Title with trajectory count
            n_traj = n_attractor_points_target * n_perturbations_target
            ax2.set_title(f'Lyapunov Exponent: TARGET vs Theoretical\n({n_traj} trajectories)', 
                          fontsize=14, fontweight='bold')
            ax2.grid(True, axis='y', alpha=0.3)
            ax2.set_ylim(0, max(lambdas) * 1.2)
            
            # Theoretical line
            ax2.axhline(lambda_theoretical, color='green', linestyle='--', 
                        linewidth=2, alpha=0.5, label='Theoretical')
            
            plt.tight_layout()
            
            if PathFigures:
                filename = f'Lyapunov_TARGET_only_{n_traj}traj.png'
                fig.savefig(PathFigures + filename, dpi=300, bbox_inches='tight')
                print(f"✓ Figure saved: {filename}")
            
            plt.show()
            plt.close()
        
        # Return results
        return {
            'lambda_target': lambda_target,
            'p_target': p_target,
            'distances_flat_target': distances_flat_target,
            'log_distance_target': log_distance_target,
            'distance_mean_target': distance_mean_target,
            'n_attractor_points': n_attractor_points_target,
            'n_perturbations': n_perturbations_target,
            'time_steps': time_steps_target
        }


    #
    # Defining function to plot distribution of noise for Lorenz63 system:
    #
    def plot_distribution_histogram_Lorenz63(self, state, num_bins, colorstring, PathFigures):
        """
        Plotting distribution of state for Lorenz63 system:
        """
        fig = plt.figure(figsize=(10, 6))
        plt.hist(state[:,0], bins=num_bins, alpha=0.7, color=colorstring, edgecolor='black')
        plt.title('Distribution of Initial Condition Perturbations (x-component)')
        plt.xlabel('Perturbation Value')
        plt.ylabel('Frequency')
        plt.grid()
        fig.savefig(PathFigures + 'Distribution_of_State.png')
        plt.show()
        plt.close()


    #
    # Defining function to run perturbed forecast for Lorenz63 system:
    #
    def run_perturbed_fcst_L63(self, length_forecast, number_of_samples, initial_index, solver):
        """
        Run perturbed forecast for Lorenz63 system:
        """
        print('Run Perturbed Forecast for Lorenz63')
        target = np.asarray(self.target)
        xf_init = target[initial_index][0,:]
        # save xf_init to self:
        self.xf_init = xf_init
        #
        print(f'xf_init: {xf_init}')
        # Run forecast with unperturbed initial conditions:
        xf_unpert = solver.run(xf_init, np.arange(0, self.dt * length_forecast, self.dt), np.arange(0, self.dt * length_forecast, self.dt))
        #
        # Add random noise to xf_init and run forecast with noise:
        #
        # Generate random noise
        noise = self.rng.normal(0, 1.0, (number_of_samples, *xf_init.shape))
        #
        # Create the perturbed initial conditions:
        #
        xf_init_noisy = xf_init + noise
        #
        # Run forecasts with each perturbed initial conditions:
        #
        xf_pert = []
        for i in range(number_of_samples):
            xf_pert.append(solver.run(xf_init_noisy[i], np.arange(0, self.dt * length_forecast, self.dt), np.arange(0, self.dt * length_forecast, self.dt)))
        xf_pert = np.array(xf_pert)
        # save xf_unpert and xf_pert to self:
        self.xf_unpert = xf_unpert
        self.xf_pert   = xf_pert
        return xf_unpert, xf_pert
    
    
    
    
    
    
    
    
    #
    # Defining function to run perturbed forecast:
    #
    def run_perturbed_fcst(self, length_forecast, number_of_samples, initial_index, solver):
        """
        Run perturbed forecast:
        """
        #print('CSE System Run Perturbed Forecast')
        target = np.asarray(self.target)
        xf_init = target[initial_index][0,:]
        # save xf_init to self:
        self.xf_init = xf_init
        #
        print(f'xf_init: {xf_init}')
        # Run forecast with unperturbed initial conditions:
        xf_unpert = solver.run(xf_init, np.arange(0, self.dt * length_forecast, self.dt), np.arange(0, self.dt * length_forecast, self.dt))
        #
        # Add random noise to xf_init and run forecast with noise:
        #
        # Generate random noise
        noise = self.rng.normal(0, 0.4, (number_of_samples, *xf_init.shape))
        #
        # Create the perturbed initial conditions:
        #
        xf_init_noisy = xf_init + noise
        #
        # Run forecasts with each perturbed initial conditions:
        #
        xf_pert = []
        for i in range(number_of_samples):
            xf_pert.append(solver.run(xf_init_noisy[i], np.arange(0, self.dt * length_forecast, self.dt), np.arange(0, self.dt * length_forecast, self.dt)))
        xf_pert = np.array(xf_pert)
        # save xf_unpert and xf_pert to self:
        self.xf_unpert = xf_unpert
        self.xf_pert   = xf_pert
        return xf_unpert, xf_pert
    
    
    #
    # Defining function to compute norm difference between perturbations:
    #
    def Compute_Perturbation_Norm_Diff(self, number_of_samples):
        """
        Compute norm difference between perturbations:
        """
        print('Compute Perturbation Norm Difference')
        norm_diff_ensemble = np.zeros((number_of_samples, 1))
        for i in range(number_of_samples):
            norm_diff_i = np.linalg.norm(self.xf_pert[i, :, 0] - self.xf_unpert[:,0], axis=0) # This gives a 1D array of norms at each time step
            norm_diff_ensemble[i] = norm_diff_i
        # Save to self:
        self.norm_diff_ensemble = norm_diff_ensemble
        return norm_diff_ensemble


    #
    # Defining function to plot norm difference between perturbed and unperturbed forecasts:
    #
    def Plot_Norm_Difference_Perturbed_Unperturbed_Forecasts(self,length_forecast, number_of_samples):
        """
        Plot the norm difference between perturbed and unperturbed forecasts:
        """
        print('Plot Norm Difference Perturbed-Unperturbed Forecasts')
        fig = plt.figure(figsize = (10,5))
        plt.bar(np.arange(1, number_of_samples + 1), self.norm_diff_ensemble.flatten(), color='skyblue')
        plt.xlabel('Ensemble Member')
        plt.ylabel('Norm of Difference')
        plt.title(f'Norm of Difference between Perturbed and Unperturbed Forecasts for Each Ensemble Member \n (Forecast length = {length_forecast} steps)')
        plt.xticks(np.arange(1, number_of_samples+1, 2))
        plt.xlim(0, number_of_samples + 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        fig.savefig(self.figures_path + 'Norm_Difference_Perturbed_Unperturbed_Forecasts.png', dpi=300, bbox_inches = 'tight')
        plt.close()
        print('Plotting complete')


    #
    # Defining function to plot histogram of norm difference between perturbed and unperturbed forecasts:
    #
    def Plot_Histogram_Norm_Difference_Perturbed_Unperturbed_Forecasts(self, bins, length_forecast):
        """
        Plot the histogram of the norm difference between perturbed and unperturbed forecasts:
        """
        # Ensure norm_diff_ensemble is 1D array
        norm_diff_ensemble = np.ravel(self.norm_diff_ensemble)
        print('Plot Histogram Norm Difference Perturbed-Unperturbed Forecasts')
        fig = plt.figure(figsize = (10,5))
        counts, bins, _ = plt.hist(norm_diff_ensemble, bins=bins,
                                 color='lightgreen', edgecolor='black', alpha=0.7)
        # Axis labels and title
        plt.xlabel('Norm of Difference')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Norm of Difference between Perturbed and Unperturbed Forecasts\n(Forecast length = {length_forecast} steps)')
        plt.grid(True, alpha=0.3)

        # X and Y limits / ticks
        plt.xticks(bins)
        plt.yticks(np.arange(0, counts.max() + 1, 2))
        plt.xlim(0, np.max(norm_diff_ensemble) * 1.05)
        plt.ylim(0, counts.max() * 1.1)

        # Optional: mean line
        mean_val = np.mean(norm_diff_ensemble)
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean = {mean_val:.2f}')
        plt.legend()
        plt.tight_layout()
        plt.show()
        fig.savefig(self.figures_path + 'Histogram_Norm_Difference_Perturbed_Unperturbed_Forecasts.png', dpi=300, bbox_inches = 'tight')
        plt.close()
        print('Plotting complete')



    
    
    
    
    #
    # Defining function to plot perturbed forecasts:
    #
    def Plot_Perturbed_Forecasts(self, length_forecast, number_of_samples, initial_index):
        """
        Plot the Perturbed Forecasts:
        """
        print('CSE System Plot Perturbed Forecasts')
        # Creating x_values for plotting:
        L        = 8      # Length of each target trajectory
        x_values = self.overlapped_values(np.array(self.target).shape[0], L)
        # Flattening target trajectory (variable X) for plotting:
        y        = np.asarray(self.target)[:, :, 0].ravel()
        # Plotting:
        fig = plt.figure(figsize = (14,5))
        plt.plot(x_values, y, color='green', linewidth=1.5, label='Target Trajectory (X component)')
        plt.scatter(x_values[::L], y[::L], color='black', s=10, label='Target Points (every 8 steps)', alpha=0.6)
        plt.scatter(x_values[initial_index*L-1] , self.xf_init[0], color='orange', s=100, label='Initial Point for Unperturbed Forecast', edgecolors='black', zorder=5)
        # Plotting ensemble perturbed forecasts:
        for i in range(number_of_samples):
            if i == 0:
                plt.plot(np.arange(initial_index*(L-1), initial_index*(L-1)+length_forecast), self.xf_pert[i,0:length_forecast,0], color='deeppink', linewidth=1.5, label='Perturbed Forecasts (50 members)', zorder=4, alpha=0.4)
            else:
                plt.plot(np.arange(initial_index*(L-1), initial_index*(L-1)+length_forecast), self.xf_pert[i,0:length_forecast,0], color='deeppink', linewidth=1.5, label='', zorder=4, alpha=0.4)
        # Plotting unperturbed forecast:
        plt.plot(np.arange(initial_index*(L-1), initial_index*(L-1)+length_forecast), self.xf_unpert[0:length_forecast,0], color='blue', linewidth=1.5, label='Unperturbed Forecast', zorder=4)
        plt.axhline(0, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
        plt.axvline(0, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
        plt.xlabel('Time Step')
        plt.ylabel('Target Trajectory (X component)')
        plt.title('Target Trajectory with 50 Perturbed Forecasts and 1 Unperturbed Forecast')
        plt.grid(True, alpha=0.3)
        plt.xticks(np.arange(0, max(x_values)+1, 100))
        plt.xlim(-1, 800)
        plt.legend()
        plt.tight_layout()
        plt.show()
        fig.savefig(self.figures_path + 'Target_Trajectory_with_Forecasts_t_index_'+str(initial_index)+'.png', dpi=300, bbox_inches = 'tight')
        plt.close()
        print('Plotting complete')
    
    
    #
    # Plotting perturbed trajectories from multiple attractor points (2D time series style)
    #
    def Plot_Perturbed_Trajectories_Attractor_TimeSeries(self, trajectory, lorenz63, sigma, beta, rho, h, 
                                                         length_forecast, n_attractor_points=5, 
                                                         n_perturbations=50, attractor_point_index=None,
                                                         window_margin=100, PathFigures=''):
        """
        Visualize perturbed forecasts from attractor points in a 2D time series plot.
        Similar to Plot_Perturbed_Forecasts but can show one or multiple initial points.
        
        Parameters:
        -----------
        trajectory : np.ndarray
            Reference trajectory, shape (3, time_steps) - can be x_nature or target_trajectory
        lorenz63 : function
            Lorenz63 dynamics function
        sigma, beta, rho : float
            Lorenz63 parameters
        h : float
            Integration time step
        length_forecast : int
            Number of time steps to forecast
        n_attractor_points : int
            Number of attractor points to sample (ignored if attractor_point_index is specified)
        n_perturbations : int
            Number of perturbations per point
        attractor_point_index : int or None
            If specified, plot only this specific attractor point (time index in trajectory).
            If None, sample n_attractor_points random points.
        window_margin : int
            Number of time steps to show before the initial point for context.
            Default is 100. Use larger values to see more context, smaller to zoom in.
        PathFigures : str
            Path to save figures
        """
        print(f"{'='*70}")
        print(f"  Plotting Perturbed Trajectories - Time Series Style")
        print(f"{'='*70}\n")
        
        # Determine which attractor points to use
        available_times = trajectory.shape[1]
        
        if attractor_point_index is not None:
            # Use specific point
            if attractor_point_index < 0 or attractor_point_index >= available_times:
                raise ValueError(f"attractor_point_index must be between 0 and {available_times-1}")
            t_indices = np.array([attractor_point_index])
            print(f"Using specific attractor point: time index {attractor_point_index}")
        else:
            # Sample random points
            t_indices = self.rng.choice(available_times, size=n_attractor_points, replace=False)
            t_indices = np.sort(t_indices)
            print(f"Sampling {n_attractor_points} random attractor points")
        
        print(f"Computing {n_perturbations} perturbed trajectories from each point...")
        print(f"Forecast length: {length_forecast} steps\n")
        
        # Determine number of points
        n_points = len(t_indices)
        
        # Create figure
        fig = plt.figure(figsize=(14, 6))
        
        # Determine plot window
        if n_points == 1:
            # For single point, focus on the region around initial point and forecast
            t_min = max(0, t_indices[0] - window_margin)
            t_max = min(trajectory.shape[1], t_indices[0] + length_forecast + 50)
        else:
            # For multiple points, show wider view
            t_min = max(0, min(t_indices) - window_margin)
            t_max = min(trajectory.shape[1], max(t_indices) + length_forecast + 50)
        
        # Plot reference trajectory within the window (no subsampling needed for focused view)
        time_window = np.arange(t_min, t_max)
        plt.plot(time_window, trajectory[0, t_min:t_max], color='green', linewidth=1.5, 
                alpha=0.6, label='Reference Trajectory (X component)', zorder=1)
        
        # Color map for different attractor points
        if n_points == 1:
            # Use a single distinctive color for one point
            colors = ['deeppink']
        else:
            colors = plt.cm.rainbow(np.linspace(0, 1, n_points))
        
        # Process each attractor point
        for i, (t_idx, color) in enumerate(zip(t_indices, colors)):
            # Get initial condition
            state_init = trajectory[:, t_idx].copy()
            
            # Mark the initial point on the trajectory
            plt.scatter(t_idx, state_init[0], color='orange', s=100, 
                       edgecolors='black', linewidths=1, zorder=10,
                       label=f'Initial Point (t={t_idx})' if n_points == 1 else f'Attractor Point {i+1}')
            
            # Compute unperturbed trajectory
            solver_unpert = run_RK4(lorenz63, sigma, beta, rho, state_init, h, 
                                   length_forecast * h, str_figures_path='')
            traj_unpert = solver_unpert.integrate()
            
            # Time axis for forecast
            time_forecast = np.arange(t_idx, t_idx + length_forecast)
            
            # Plot unperturbed forecast
            plt.plot(time_forecast, traj_unpert[:length_forecast, 0], 
                    color='blue', linewidth=1.2, alpha=0.9, zorder=5,
                    label='Unperturbed Forecast')
            
            # Compute and plot perturbed trajectories
            for j in range(n_perturbations):
                # Add small perturbation
                perturbation = self.rng.normal(0, 0.0001, size=3)
                state_pert = state_init + perturbation
                
                # Integrate
                solver_pert = run_RK4(lorenz63, sigma, beta, rho, state_pert, h,
                                     length_forecast * h, str_figures_path='')
                traj_pert = solver_pert.integrate()
                
                # Plot (only label first one)
                if j == 0:
                    plt.plot(time_forecast, traj_pert[:length_forecast, 0],
                            color=color, linewidth=1.0, alpha=0.3, zorder=3,
                            label=f'Perturbed Forecasts ({n_perturbations} members)')
                else:
                    plt.plot(time_forecast, traj_pert[:length_forecast, 0],
                            color=color, linewidth=1.0, alpha=0.3, zorder=3)
            
            if n_points == 1:
                print(f"  ✓ Complete: t={t_idx}, state=({state_init[0]:.3f}, {state_init[1]:.3f}, {state_init[2]:.3f})")
            else:
                print(f"  ✓ Point {i+1}/{n_points} complete (t={t_idx})")
        
        # Formatting
        plt.axhline(0, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('X component', fontsize=12)
        
        if n_points == 1:
            plt.title(f'Perturbed Trajectories from Attractor Point (t={t_indices[0]})\n' +
                     f'({n_perturbations} perturbations, forecast length={length_forecast} steps)',
                     fontsize=14)
        else:
            plt.title(f'Perturbed Trajectories from {n_points} Attractor Points\n' +
                     f'({n_perturbations} perturbations per point, forecast length={length_forecast} steps)',
                     fontsize=14)
        
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', fontsize=9, ncol=2)
        
        # Set focused x-axis limits
        plt.xlim(t_min, t_max)
        
        plt.tight_layout()
        
        if PathFigures:
            if n_points == 1:
                filename = f'Perturbed_Trajectories_Single_Point_t{t_indices[0]}_pert{n_perturbations}_len{length_forecast}.png'
            else:
                filename = f'Perturbed_Trajectories_TimeSeries_n{n_points}_pert{n_perturbations}_len{length_forecast}.png'
            fig.savefig(PathFigures + filename, dpi=300, bbox_inches='tight')
            print(f"\n✓ Figure saved: {filename}")
        
        plt.show()
        plt.close()
        
        print(f"\n{'='*70}")
        print(f"  Plotting Complete!")
        print(f"{'='*70}\n")
                
               
    
    
    #
    # Plotting function for the ensemble analysis (Step 1):
    #
    def Plot_Analysis_Ensemble_Step1(self, x_nature, Xf, Xa, obs, t_index, t0, tf):
        """
        Plot the ensemble analysis:
        """
        print('CSE System Plot Analysis Ensemble Step 1')
        #
        #T = int(np.ceil(T_0*multiple_number))                  # multiple of To
        #
        fig = plt.figure(figsize = (12,12))
        ax  = fig.add_subplot(211)
        #
        # Plot nature run:
        #
        plt.plot(x_nature[0,t0:tf], linestyle = 'solid', linewidth=3, marker='', markersize=3, color='black', label="Nature Run")
        #
        # Plot analysis and forecast ensemble members:
        #
        for member in range(np.shape(Xf)[0]):
            if member == 0:
                plt.scatter(t_index, Xa[0, member], marker='o', s=45, color = 'cyan', edgecolors='black', linewidths=0.7, label = "Analysis Ensemble")
                plt.scatter(t_index, Xf[0, member], marker='o', s=45, color = 'gray', alpha=0.3, edgecolors='black', linewidths=0.7, label = "Forecast Ensemble")
            else:
                plt.scatter(t_index, Xa[0, member], marker='o', s=45, color = 'cyan', edgecolors='black', linewidths=0.7)
                plt.scatter(t_index, Xf[0, member], marker='o', s=45, color = 'gray', alpha=0.3, edgecolors='black', linewidths=0.7)
        #
        # Plot observations:
        #
        plt.scatter(t_index, obs[0], marker='o', s=50, facecolors='none', edgecolors='red', linewidths=0.7, label="Observations")
        #
        plt.axhline(y = 0, linestyle='dashed', color = 'grey')   
        plt.axhline(y = 0, linestyle='dashed', color = 'grey')
        plt.axvline(x = 0, linestyle='dashed', color = 'grey')
        plt.axvline(x = t_index, linestyle='dashed', color = 'cyan')
        plt.axvline(x = 300, linestyle='dashed', color = 'grey')
        plt.xlabel(f'Time Step (1 step = {self.dt})', fontsize = 16)
        plt.ylabel('State x', fontsize = 16)
        plt.title('Lorenz-63 Nature Run and Ensemble Analysis at $t = $'+str(t_index), fontsize = 18)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.ylim(-18,18)
        ax.legend(fontsize=10)
        plt.show()
        fig.savefig(self.figures_path + 'CSE_Step1_Time_Step_'+str(t_index)+'.png', dpi=300, bbox_inches = 'tight')
        plt.close(fig)
        print('Plotting complete')
    
    
    
    
    #
    # Plotting function for the ensemble forecast (Step 2):
    #
    def Plot_Ensemble_Fcst_Step2(self, x_nature, Xf, Xa, t_index, t0, tf, T_0, multiple_number):
        """
        Plot the ensemble forecast:
        """
        print('CSE System Plot Ensemble Forecast Step 2')
        #
        T = int(np.ceil(T_0*multiple_number))                  # multiple of To
        #
        fig = plt.figure(figsize = (12,12))
        ax  = fig.add_subplot(211)
        #
        # Plot nature run:
        #
        plt.plot(x_nature[0,t0:tf], linestyle = 'solid', linewidth=3, marker='', markersize = 3, color = 'black', label="Nature Run")
        #
        # Plot forecast ensemble members:
        #
        for member in range(np.shape(Xf)[0]):
            plt.scatter(t_index, Xa[0, member], marker='o', s=45, color = 'cyan', edgecolors='black', linewidths=0.7)
            if member == 0:
                plt.plot(np.arange(t_index,T+t_index,1), Xf[member,0:T,0], linestyle = '-', marker='', markersize = 2, color = 'cyan', label = f"Forecast Ensemble")
            else:
                plt.plot(np.arange(t_index,T+t_index,1), Xf[member,0:T,0], linestyle = '-', marker='', markersize = 2, color = 'cyan')  
        #  
        plt.axhline(y = 0, linestyle='dashed', color = 'grey')   
        plt.axhline(y = 0, linestyle='dashed', color = 'grey')
        plt.axvline(x = 0, linestyle='dashed', color = 'grey')
        plt.axvline(x = t_index, linestyle='dashed', color = 'cyan')
        plt.axvline(x = 300, linestyle='dashed', color = 'grey')
        plt.xlabel(f'Time Step (1 step = {self.dt})', fontsize = 16)
        plt.ylabel('State x', fontsize = 16)
        plt.title(r'Lorenz-63 Nature Run and $T = \lceil 4T_o \rceil$ Ensemble Forecasts initiated at $t = {}$'.format(t_index), fontsize=18)
        plt.yticks(fontsize = 14)   
        plt.ylim(-18,18)
        ax.legend(fontsize=10)
        #plt.show()
        fig.savefig(self.figures_path + 'CSE_Step2_Time_Step_'+str(t_index)+'.png', dpi=300, bbox_inches = 'tight')
        plt.close(fig)
        print('Plotting complete')
        
        
    
    #
    # Plotting function for the ensemble forecast (Step 2):
    #
    def Plot_Ensemble_Fcst_Step4(self, Xf, Xa, yo, t_index, T_0, multiple_number, title):
        """
        Plot the ensemble forecast:
        """
        print('CSE System Plot Ensemble Forecast Step 2')
        #
        T = int(np.ceil(T_0*multiple_number))                  # multiple of To
        #
        fig = plt.figure(figsize = (12,12))
        ax  = fig.add_subplot(211)
        for member in range(np.shape(Xf)[0]):
            if member == 0:
                plt.plot(np.arange(t_index,T+t_index,1), Xf[member,0:T,0], linestyle = '-', marker='', markersize = 2, color = 'cyan', label = f"Ensemble Forecast")
                plt.scatter(t_index, Xa[0, member], marker='x', s=65, color = 'blue', edgecolors='black', linewidths=0.7, label="Ensemble Analysis")
            else:
                plt.plot(np.arange(t_index,T+t_index,1), Xf[member,0:T,0], linestyle = '-', marker='', markersize = 2, color = 'cyan')   
                plt.scatter(t_index, Xa[0, member], marker='x', s=65, color = 'blue', edgecolors='black', linewidths=0.7) 
        plt.scatter(t_index, yo[0], marker='o', s = 30, facecolors='red', edgecolors='red', label="Observations")
        plt.axhline(y = 0, linestyle='dashed', color = 'grey')   
        plt.axhline(y = 0, linestyle='dashed', color = 'grey')
        plt.axvline(x = 0, linestyle='dashed', color = 'grey')
        plt.axvline(x = t_index, linestyle='dashed', color = 'cyan')
        plt.axvline(x = t_index+self.Ta, linestyle='dashed', color = 'cyan')
        plt.axvline(x = 300, linestyle='dashed', color = 'grey')
        plt.xlabel(f'time step (1 step = {self.dt})', fontsize = 16)
        plt.ylabel('state x', fontsize = 16)
        plt.title(title, fontsize = 18)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.ylim(-18,18)
        plt.xlim(-2,self.t_index+350)
        ax.legend(fontsize=10)
        plt.show()
        #fig.savefig('Forecasts_Step2_Time_Step_'+str(t_index)+'.png', dpi=300, bbox_inches = 'tight')
        plt.close(fig)
        print('Plotting complete')
    
    
    
    
    
    #
    # Plotting function for the ensemble forecast (Step 2):
    #
    def Plot_All_Regime_Shift(self, Xf, X_orig, X_history, t_index_shift, t_index, T_0, Ta, multiple_number):
        """
        Plot the ensemble forecast:
        """
        print('CSE System Plot Ensemble Forecast Step 2')
        #
        T = int(np.ceil(T_0*multiple_number))                  # multiple of To
        #
        fig = plt.figure(figsize = (12,12))
        ax  = fig.add_subplot(211)
        for member in range(np.shape(Xf)[0]):
            if member == 0:
                plt.plot(np.arange(t_index_shift,T+t_index_shift,1), Xf[member,0:T,0], linestyle = '-', marker='', markersize = 2, color = 'cyan', label = f"Forecast Ensemble")
            else:
                plt.plot(np.arange(t_index_shift,T+t_index_shift,1), Xf[member,0:T,0], linestyle = '-', marker='', markersize = 2, color = 'cyan')
        plt.plot(np.arange(t_index,Ta+t_index,1), X_history[0:T], linestyle = '-', marker='', markersize = 2, color = 'red', label = f"Forecast Ensemble")  
        plt.axhline(y = 0, linestyle='dashed', color = 'grey')   
        plt.axhline(y = 0, linestyle='dashed', color = 'grey')
        plt.axvline(x = 0, linestyle='dashed', color = 'grey')
        plt.axvline(x = t_index_shift, linestyle='dashed', color = 'cyan')
        plt.axvline(x = t_index, linestyle='dashed', color = 'red')
        plt.axvline(x = 300, linestyle='dashed', color = 'grey')
        plt.xlabel(f'time step (1 step = {self.dt})', fontsize = 16)
        plt.ylabel('state x', fontsize = 16)
        plt.title('Lorenz-63 Nature Run and Initial Ensemble Forecasts at $t = $'+str(t_index), fontsize = 18)
#        xticks = list(plt.xticks()[0]) + [t_index]
#        xticks = sorted(set(xticks))
#        ax.set_xticks(xticks)
#        labels = [r"Ta" if int(tick) == t_index else str(int(tick)) for tick in xticks]
#        ax.set_xticklabels(labels, fontsize=14)
#        ax.get_xticklabels()[labels.index("Ta")].set_color('cyan')
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.ylim(-18,18)
        #plt.xlim(-10,350)
        ax.legend(fontsize=14)
        plt.show()
        #fig.savefig('Forecasts_Step2_Time_Step_'+str(t_index)+'.png', dpi=300, bbox_inches = 'tight')
        plt.close(fig)
        print('Plotting complete')
    
    
    

    #
    # Plotting function for the ensemble forecast (Step 2):
    #
    def Plot_Ensemble_Fcst_Step5(self, x_nature, x_nature_control, Xf, Xa, yo, t_index, T_0, multiple_number):
        """
        Plot the ensemble forecast:
        """
        print('CSE System Plot Ensemble Forecast Step 2')
        #
        T = int(np.ceil(T_0*multiple_number))     # multiple of To
        x_nature_control_3d = np.asarray(x_nature_control)
        #
        fig = plt.figure(figsize = (12,12))
        ax  = fig.add_subplot(211)
        plt.plot(x_nature[0,105768:105768+3*350], linestyle = 'solid', linewidth=3, marker='', markersize = 3, color = 'black', label="Nature Run")
        plt.plot(np.arange(t_index-8,t_index,1), x_nature_control_3d[:,0], linestyle = 'solid', linewidth=2, marker='', markersize = 3, color = 'red', label="Nature Run (Control)")
        print('Xf shape: ', np.shape(Xf))
        print('Xf[:,0, 0]: ', Xf[:, 8, 0])
        #print('Xf[:,t_index-8, 0]: ', Xf[:,t_index-8, 0])
        print('Xa[0, :]: ', Xa[0, :])
        for member in range(np.shape(Xf)[0]):
            if member == 0:
                plt.scatter(t_index, Xa[0, member], marker='o', s=45, color = 'deeppink', edgecolors='black', linewidths=0.7, label="Ensemble Analysis at t")
                plt.scatter(t_index, Xf[member,8, 0], marker='x', s=45, color = 'black', edgecolors='black', linewidths=0.7, label="Ensemble Forecast at t")
                plt.scatter(t_index-8, Xf[member,0, 0], marker='o', s=45, color = 'cyan', edgecolors='black', linewidths=0.7, label="Ensemble Forecast at t-Ta")
                plt.plot(np.arange(t_index-8,t_index+1,1), Xf[member,0:8+1,0], linestyle = '-', marker='', markersize = 2, color = 'cyan')
            else:
                plt.scatter(t_index, Xa[0, member], marker='o', s=45, color = 'deeppink', edgecolors='black', linewidths=0.7)
                plt.scatter(t_index, Xf[member,8, 0], marker='x', s=45, color = 'black', edgecolors='black', linewidths=0.7)
                plt.scatter(t_index-8, Xf[member,0, 0], marker='o', s=45, color = 'cyan', edgecolors='black', linewidths=0.7)
                plt.plot(np.arange(t_index-8,t_index+1,1), Xf[member,0:8+1,0], linestyle = '-', marker='', markersize = 2, color = 'cyan')
        plt.scatter(t_index, yo[0], marker='o', s = 30, facecolors='red', edgecolors='red', label="Observations")
        plt.axhline(y = 0, linestyle='dashed', color = 'grey')   
        plt.axhline(y = 0, linestyle='dashed', color = 'grey')
        plt.axvline(x = t_index-8, linestyle='dashed', color = 'grey')
        plt.axvline(x = t_index, linestyle='dashed', color = 'cyan')
        plt.axvline(x = 300, linestyle='dashed', color = 'grey')
        plt.xlabel(f'time step (1 step = {self.dt})', fontsize = 16)
        plt.ylabel('state x', fontsize = 16)
        plt.title('Lorenz-63 Nature Run and Ensemble Forecasts at $t = $'+str(t_index), fontsize = 18)
        #xticks = list(plt.xticks()[0]) + [t_index]
        #xticks = sorted(set(xticks))
        ax.set_xticks(np.arange(t_index-10, t_index+10, 1))
        #labels = [r"Ta" if int(tick) == t_index else str(int(tick)) for tick in xticks]
        #ax.set_xticklabels(labels, fontsize=14)
        #ax.get_xticklabels()[labels.index("Ta")].set_color('cyan')
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.ylim(-18,18)
        plt.xlim(t_index-10,t_index+10)
        ax.legend(fontsize=14)
        plt.show()
        fig.savefig('Forecasts_Step2_Time_Step_'+str(t_index)+'.png', dpi=300, bbox_inches = 'tight')
        plt.close(fig)
        print('Plotting complete')
        
        
     
    
    def Plot_New_NR_and_Obs(self, x_init, x_nature, yo_new, yo_old, t_index, Ta, solver, das):
        """
        Plot the new nature run and observations:
        """
        print('Evolve Nature Run 350 - t time steps forward')
        print('t_index: ', t_index)
        print('350 - t_index: ', 350 - t_index)
        print('shape np.arange(t_index,350): ', np.shape(np.arange(t_index,350)))
        print('shape np.arange(t_index,350-t_index): ', np.shape(np.arange(t_index,350-t_index)))   
        print(f'shape np.arange(0, {self.dt} * (350 - t_index), {self.dt}): ', np.shape(np.arange(0, self.dt * (350 - t_index), self.dt)))    
        x_nature_evolved = solver.run(x_init, np.arange(0, self.dt * (350 - t_index - self.dt), self.dt), np.arange(0, self.dt * (350 - t_index - self.dt), self.dt))[:,0]
        print('x_nature_evolved shape: ', x_nature_evolved.shape)
        print('x_nature_evolved: ', x_nature_evolved)
        print('CSE System Plot New Nature Run and Observations')
        fig = plt.figure(figsize = (12,12))
        ax  = fig.add_subplot(211)
        plt.plot(x_nature[0,105768:105768+350], linestyle = 'solid', linewidth=3, marker='', markersize = 3, color = 'black', label="Nature Run before control")
        plt.plot(np.arange(t_index,350),x_nature_evolved, linestyle = 'solid', linewidth=3, marker='', markersize = 3, color = 'red', label="Nature Run after control")
        plt.scatter(t_index,yo_new[0], marker='o', s = 30, facecolors='deeppink', edgecolors='deeppink', label="Observations")
        plt.scatter(t_index,yo_old[0,t_index], marker='o', s = 30, facecolors='green', edgecolors='green', label="NR Observations")
        plt.axhline(y = 0, linestyle='dashed', color = 'grey')
        plt.axvline(x = 0, linestyle='dashed', color = 'grey')
        plt.axvline(x = t_index, linestyle='dashed', color = 'red')
        plt.axvline(x = 300, linestyle='dashed', color = 'grey')
        plt.xlabel(f'time step (1 step = {self.dt})', fontsize = 16)
        plt.ylabel('state x', fontsize = 16)
        plt.title('Lorenz-63 Nature Run after control at time $t = $'+str(t_index), fontsize = 18)
        #xticks = list(plt.xticks()[0]) + [t_index]
        #xticks = sorted(set(xticks))
        #ax.set_xticks(xticks)
        #labels = [r"Ta" if int(tick) == t_index else str(int(tick)) for tick in xticks]
        #ax.set_xticklabels(labels, fontsize=14)
        #ax.get_xticklabels()[labels.index("Ta")].set_color('red')
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.ylim(-18,18)
        plt.xlim(-10,350)
        ax.legend(fontsize=14)
        plt.show()
        #fig.savefig('Forecasts_Step2_Time_Step_'+str(t_index)+'.png', dpi=300, bbox_inches = 'tight')
        plt.close(fig)
        print('Plotting complete')
        
        
        
    
    def Plot_Nature_Run_Comparison(self, nature_nocontrol, nature_control, yobs, t0, tf, Ta, num_cycles):
        """
        Plot the new nature run and observations:
        """
        fig = plt.figure(figsize = (14,14))
        ax  = fig.add_subplot(311)
        plt.plot(nature_nocontrol[0,0:num_cycles*Ta+1], linestyle = 'solid', linewidth=2, marker='', markersize = 3, color = 'black', label="Nature Run (No Control)")
        plt.plot(nature_control[:,0], linestyle = 'solid', linewidth=2, marker='', markersize = 3, color = 'red', label="Nature Run (Control)")
        plt.scatter(np.arange(Ta,Ta*num_cycles+Ta,Ta), yobs[:,0], marker='o', s=25, facecolors='none', edgecolors='blue', label="Observations")
        plt.axhline(y = 0, linestyle='dashed', color = 'grey')
        plt.axhline(y = 0, linestyle='dashed', color = 'grey')
        plt.axvline(x = 0, linestyle='dashed', color = 'grey')
        plt.axvline(x = 300, linestyle='dashed', color = 'grey')
        plt.ylabel('State x', fontsize = 16)
        plt.title('Lorenz-63 Nature Run', fontsize = 18)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.ylim(-18,18)
        ax.legend(fontsize=10)
        #
        ax  = fig.add_subplot(312)
        plt.plot(nature_nocontrol[1,0:num_cycles*Ta+1], linestyle = 'solid', linewidth=2, marker='', markersize = 3, color = 'black', label="Nature Run (No Control)")
        plt.plot(nature_control[:,1], linestyle = 'solid', linewidth=2, marker='', markersize = 3, color = 'red', label="Nature Run (Control)")
        plt.scatter(np.arange(Ta,Ta*num_cycles+Ta,Ta), yobs[:,1], marker='o', s=25, facecolors='none', edgecolors='blue', label="Observations")
        plt.axhline(y = 0, linestyle='dashed', color = 'grey')
        plt.axhline(y = 0, linestyle='dashed', color = 'grey')
        plt.axvline(x = 0, linestyle='dashed', color = 'grey')
        plt.axvline(x = 300, linestyle='dashed', color = 'grey')
        plt.ylabel('State y', fontsize = 16)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        ax.legend(fontsize=10)
        #
        ax  = fig.add_subplot(313)
        plt.plot(nature_nocontrol[2,0:num_cycles*Ta+1], linestyle = 'solid', linewidth=2, marker='', markersize = 3, color = 'black', label="Nature Run (No Control)")
        plt.plot(nature_control[:,2], linestyle = 'solid', linewidth=2, marker='', markersize = 3, color = 'red', label="Nature Run (Control)")
        plt.scatter(np.arange(Ta,Ta*num_cycles+Ta,Ta), yobs[:,2], marker='o', s=25, facecolors='none', edgecolors='blue', label="Observations")
        plt.axhline(y = 0, linestyle='dashed', color = 'grey')
        plt.axhline(y = 0, linestyle='dashed', color = 'grey')
        plt.axvline(x = 0, linestyle='dashed', color = 'grey')
        plt.axvline(x = 300, linestyle='dashed', color = 'grey')
        plt.xlabel(f'Time Step (1 step = {self.dt})', fontsize = 16)
        plt.ylabel('State z', fontsize = 16)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        ax.legend(fontsize=10)
        plt.show()
        fig.savefig(self.figures_path+'NR_comparison_after_control.png', dpi=300, bbox_inches = 'tight')
        plt.close(fig)
        print('Plotting complete')
        



    def Plot_3D_Nature_Run_Comparison(self, nature_nocontrol, nature_control, yobs, t0, tf, Ta, num_cycles):
        """
        Plot the new nature run and observations:
        """
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        fig = plt.figure(figsize = (14,7))
        ax1  = fig.add_subplot(121, projection='3d', facecolor='white')
        ax1.scatter(
            nature_nocontrol[0,0:num_cycles*Ta+1],
            nature_nocontrol[1,0:num_cycles*Ta+1],
            nature_nocontrol[2,0:num_cycles*Ta+1],
            c=np.linspace(0, 1, len(nature_nocontrol[0,0:num_cycles*Ta+1])),
            cmap='jet',
            s=3,
            alpha = 0.9
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
        #ax1.set_axis_off()
        #ax1.view_init(elev=30, azim=45)
        #
        ax2  = fig.add_subplot(122, projection='3d')
        ax2.scatter(
            nature_control[:,0],
            nature_control[:,1],
            nature_control[:,2],
            c=np.linspace(0, 1, np.shape(nature_control)[0]),
            cmap='jet',
            s=3,
            alpha = 0.9
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
        #plt.subplots_adjust(left=0.05, right=0.6, bottom=0.05, top=0.55, wspace=0.15)
        plt.subplots_adjust(wspace = -0.1)
        plt.tight_layout()
        plt.show()
        fig.savefig(self.figures_path+'NR_3D_comparison_after_control.png', dpi=300, bbox_inches = 'tight')
        plt.close(fig)
        print('Plotting complete')


    def Plot_3D_Nature_Run_Comparison_With_Interventions(self, nature_nocontrol, nature_control, yobs, t0, tf, Ta, num_cycles, intervention_history):
        """
        Plot the controlled 3D nature run with intervention points emphasized.
        """
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d', facecolor='white')

        ax.plot(
            nature_control[:, 0],
            nature_control[:, 1],
            nature_control[:, 2],
            color='#294A8D',
            linewidth=0.9,
            alpha=0.45,
            label='Controlled trajectory'
        )
        ax.scatter(
            nature_control[:, 0],
            nature_control[:, 1],
            nature_control[:, 2],
            color='#4F7FD8',
            s=6,
            alpha=0.18,
            depthshade=False
        )

        intervention_indices = []
        for fallback_cycle, item in enumerate(intervention_history):
            norms = np.asarray(item.get('norms', []), dtype=float)
            if not item.get('controlled', bool(np.any(norms > 0))):
                continue

            cycle_index = item.get('cycle_index', fallback_cycle)
            if cycle_index is None or cycle_index < 0 or cycle_index >= num_cycles:
                continue

            for substep in np.where(norms > 0)[0]:
                if substep < Ta:
                    flat_index = int(cycle_index) * Ta + int(substep)
                    if flat_index < np.shape(nature_control)[0]:
                        intervention_indices.append(flat_index)

        intervention_indices = np.unique(intervention_indices)
        total_points = int(np.shape(nature_control)[0])
        intervention_points = int(len(intervention_indices))
        non_intervention_points = total_points - intervention_points
        controlled_cycles = int(np.sum([bool(item.get('controlled', False)) for item in intervention_history]))

        if intervention_indices.size:
            ax.scatter(
                nature_control[intervention_indices, 0],
                nature_control[intervention_indices, 1],
                nature_control[intervention_indices, 2],
                color='#B7F34A',
                marker='o',
                s=185,
                alpha=0.35,
                depthshade=False,
                zorder=4
            )
            ax.scatter(
                nature_control[intervention_indices, 0],
                nature_control[intervention_indices, 1],
                nature_control[intervention_indices, 2],
                color='#00A651',
                edgecolors='#102A12',
                linewidths=0.8,
                marker='o',
                s=54,
                alpha=0.95,
                label='Intervention applied',
                depthshade=False,
                zorder=5
            )
            ax.legend(loc='upper right', fontsize=10)

        info_text = (
            f'Total points: {total_points}\n'
            f'With intervention: {intervention_points}\n'
            f'Without intervention: {non_intervention_points}\n'
            f'Controlled cycles: {controlled_cycles}/{num_cycles}'
        )
        ax.text2D(
            0.02, 0.94, info_text,
            transform=ax.transAxes,
            va='top',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.5', alpha=0.92)
        )

        ax.set_xlabel('X', labelpad=8)
        ax.set_ylabel('Y', labelpad=8)
        ax.set_zlabel('Z', labelpad=8)
        ax.set_title('Lorenz-63 Controlled Trajectory and Intervention Points', fontsize=16, pad=14)
        x_min, x_max = np.min(nature_control[:, 0]), np.max(nature_control[:, 0])
        y_min, y_max = np.min(nature_control[:, 1]), np.max(nature_control[:, 1])
        z_min, z_max = np.min(nature_control[:, 2]), np.max(nature_control[:, 2])
        x_pad = max(0.7, 0.08 * (x_max - x_min))
        y_pad = max(0.7, 0.08 * (y_max - y_min))
        z_pad = max(0.7, 0.08 * (z_max - z_min))
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_zlim(z_min - z_pad, z_max + z_pad)
        ax.set_box_aspect((x_max - x_min + 2 * x_pad, y_max - y_min + 2 * y_pad, z_max - z_min + 2 * z_pad))
        ax.view_init(elev=18, azim=-35)
        ax.grid(alpha=0.22)
        ax.xaxis.set_pane_color((0.98, 0.98, 0.98, 1.0))
        ax.yaxis.set_pane_color((0.98, 0.98, 0.98, 1.0))
        ax.zaxis.set_pane_color((0.98, 0.98, 0.98, 1.0))

        plt.tight_layout()
        plt.show()
        fig.savefig(self.figures_path + 'NR_3D_comparison_after_control_interventions.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Intervention points plotted in 3D: {intervention_points}')
        print(f'Points without intervention in 3D: {non_intervention_points}')
        print('Plotting complete')




    def Plot_3D_Target_Trajectory(self):
        target_array = np.array(self.target).T 
        # Reorganizar para obtener secuencia temporal continua
        if len(target_array.shape) == 3:
            # Aplanar las dimensiones de tiempo: (3, Ta, num_cycles) -> (3, Ta*num_cycles)
            target_flat = target_array.reshape(3, -1)
            print(f'Target reshaped de {target_array.shape} a {target_flat.shape}')
        else:
            target_flat = target_array
            
        fig = plt.figure(figsize=(6, 6))
        ax1 = fig.add_subplot(111, projection='3d', facecolor='white')
        target_length = target_flat.shape[1]
        print(f'Target length: {target_length}')
        
        ax1.scatter(
            target_flat[0, :],
            target_flat[1, :],
            target_flat[2, :],
            c=np.linspace(0, 1, target_length),
            cmap='viridis',  # Yellow-green-blue gradient
            s=3,
            alpha=0.9,
            edgecolors='none'
        )
        ax1.set_xlabel('X', fontsize=12)
        ax1.set_ylabel('Y', fontsize=12)
        ax1.set_zlabel('Z', fontsize=12)
        ax1.set_title('Target Trajectory', fontsize=16, pad=1)
        ax1.set_xlim(-25, 25)
        ax1.set_ylim(-20, 20)
        ax1.set_zlim(0, 40)
        ax1.grid(alpha=0.3)
        ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        plt.tight_layout()
        plt.savefig(self.figures_path + 'Target_Trajectory_3D.png', dpi=300, bbox_inches='tight')
        print(f'✓ Figura guardada en: {self.figures_path}Target_Trajectory_3D.png')
        plt.show()
        plt.close(fig)




    def Gif_comparison_evolution_Lorenz_Attractor(self, nature_nocontrol, nature_control, tf):
        """
        Create gif comparing the evolution of the Lorenz attractor with and without control:
        """
        solution = nature_nocontrol.copy().T[0:tf, :]
        solution_control = nature_control.copy()[0:tf, :]
        # Time setup
        t0, t_end, dt = 0.0, 80.0, 0.01
        t_points = np.arange(t0, t_end, dt)
        n_steps = min(len(t_points), solution.shape[0])
        # Normalize for color mapping
        norm = Normalize(vmin=0, vmax=n_steps)
        cmap = cm.get_cmap('jet')
        # Create figure
        fig = plt.figure(figsize=(13, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        plt.subplots_adjust(wspace=0.1)
        # Shared axis limits
        for ax in [ax1, ax2]:
            ax.set_xlim((-20, 20))
            ax.set_ylim((-30, 30))
            ax.set_zlim((0, 50))
            ax.set_xlabel("X", fontsize=12)
            ax.set_ylabel("Y", fontsize=12)
            ax.set_zlabel("Z", fontsize=12)
        ax1.set_title("Lorenz-63 No Control", fontsize=14)
        ax2.set_title("Lorenz-63 Under Control", fontsize=14)
        # Initialize lines/points
        line1, = ax1.plot([], [], [], lw=2)
        point1, = ax1.plot([], [], [], 'o', markeredgecolor='black', markeredgewidth = 0.8, markersize=6)
        line2, = ax2.plot([], [], [], lw=2)
        point2, = ax2.plot([], [], [], 'o', markeredgecolor='black', markeredgewidth = 0.8, markersize=6)
        window_size = 80
        def init():
            for line, point in [(line1, point1), (line2, point2)]:
                line.set_data([], [])
                line.set_3d_properties([])
                point.set_data([], [])
                point.set_3d_properties([])
            return line1, point1, line2, point2
        def update(frame):
            start = max(0, frame - window_size)
            end = frame
            color = cmap(norm(end))
            # No-control plot
            line1.set_data(solution[start:end+1, 0], solution[start:end+1, 1])
            line1.set_3d_properties(solution[start:end+1, 2])
            line1.set_color(color)
            point1.set_data([solution[end, 0]], [solution[end, 1]])
            point1.set_3d_properties([solution[end, 2]])
            point1.set_color(color)
            # Control plot
            line2.set_data(solution_control[start:end+1, 0], solution_control[start:end+1, 1])
            line2.set_3d_properties(solution_control[start:end+1, 2])
            line2.set_color(color)
            point2.set_data([solution_control[end, 0]], [solution_control[end, 1]])
            point2.set_3d_properties([solution_control[end, 2]])
            point2.set_color(color)
            return line1, point1, line2, point2
        # Create animation
        anim = FuncAnimation(fig, update, frames=range(0, n_steps, 4), init_func=init, blit=True)
        # Save or display
        anim.save(self.figures_path + "L63_comparison_after_control.gif", fps=20, writer="pillow")
        plt.show()
        print('Plotting complete')




    def overlapped_values(self, K, L):
        """
        Generate a list of overlapped values for K blocks of length L.
        
        Parameters:
        K: Number of blocks
        L: Length of each block
        
        Returns:
        A concatenated numpy array of overlapped values.
        """
        xs = []
        for i in range(0, K):
            xs.append(np.arange((L-1)*i, L+(L-1)*i))
        return np.concatenate(xs)



    def Plot_Target_Trajectory(self, zoom, zoom_xlim):
        #
        # Target trajectory
        target = np.array(self.target)
        print('target trajectory shape:', np.array(target).shape)
        # Computing x_values for plotting:
        L = 8      # Length of each target trajectory
        x_values = self.overlapped_values(np.array(target).shape[0], L)
        # Flattening target trajectory for all components:
        x_comp = np.asarray(target)[:, :, 0].ravel()
        y_comp = np.asarray(target)[:, :, 1].ravel()
        z_comp = np.asarray(target)[:, :, 2].ravel()
        
        # Plotting all three components:
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # X component
        axes[0].plot(x_values, x_comp, color='green', linewidth=1.5, label='Target Trajectory (X component)')
        axes[0].axhline(0, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
        axes[0].axvline(0, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
        axes[0].set_ylabel('X Component', fontsize=12)
        axes[0].set_title('Target Trajectory - All Components', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        if zoom:
            axes[0].set_xlim(-1, zoom_xlim)
        
        # Y component
        axes[1].plot(x_values, y_comp, color='blue', linewidth=1.5, label='Target Trajectory (Y component)')
        axes[1].axhline(0, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
        axes[1].axvline(0, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
        axes[1].set_ylabel('Y Component', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        if zoom:
            axes[1].set_xlim(-1, zoom_xlim)
        
        # Z component
        axes[2].plot(x_values, z_comp, color='purple', linewidth=1.5, label='Target Trajectory (Z component)')
        axes[2].axhline(0, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
        axes[2].axvline(0, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
        axes[2].set_xlabel('Time Step', fontsize=12)
        axes[2].set_ylabel('Z Component', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        if zoom:
            axes[2].set_xlim(-1, zoom_xlim)
            print('Zooming in on x-axis to: ', zoom_xlim)
        else:
            print('No zoom on x-axis')
        
        # Set x-ticks for all axes (every 500 steps for better readability)
        for ax in axes:
            ax.set_xticks(np.arange(0, max(x_values)+1, 500))
        
        plt.tight_layout()
        plt.show()
        if zoom:
            fig.savefig(self.figures_path + f'Target_Trajectory_zoomed_to_{zoom_xlim}.png', dpi=300)
        else:
            fig.savefig(self.figures_path + 'Target_Trajectory.png', dpi=300)
        plt.close()
        print('Plotting complete')
    
    
    def Plot_Target_Trajectory_test(self, zoom, zoom_xlim):
        #
        # Target trajectory
        target = np.array(self.target)
        print('target trajectory shape:', np.array(target).shape)
        # Computing x_values for plotting:
        L = 8      # Length of each target trajectory
        x_values = self.overlapped_values(np.array(target).shape[0], L)
        # Flattening target trajectory (variable X) for plotting:
        y = np.asarray(target)[:, :, 0].ravel()
        # Plotting:
        fig = plt.figure(figsize=(14, 5))
        plt.plot(x_values, y, color='green', linewidth=1.5, label='Target Trajectory (X component)')
        plt.axhline(0, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
        plt.axvline(0, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
        plt.xlabel('Time Step')
        plt.ylabel('Target Trajectory (X component)')
        plt.title('Target Trajectory')
        plt.grid(True, alpha=0.3)
        plt.xticks(np.arange(0, max(x_values)+1, 250))
        if zoom:
            print('Zooming in on x-axis to: ', zoom_xlim)
            plt.xlim(-1, zoom_xlim)
        else:
            print('No zoom on x-axis')
            plt.xlim(-1, max(x_values))
        plt.legend()
        plt.tight_layout()
        plt.show()
        if zoom:
            fig.savefig(self.figures_path + f'Target_Trajectory_zoomed_to_{zoom_xlim}.png', dpi=300)
        else:
            fig.savefig(self.figures_path + 'Target_Trajectory.png', dpi=300)
        plt.close()
        print('Plotting complete')



    # Defining function to plot comparison target trajectory and nature run:
    def Plot_Target_Trajectory_Comparison(self,x_nature,zoom,zoom_window):
        # Target trajectory
        target = np.array(self.target)
        # Generate overlapped time axis (same as Plot_Perturbed_Forecasts)
        # target_array has shape (num_cycles, Ta, 3)
        L = self.Ta
        K = target.shape[0]
        print(f'K (number of cycles) = {K}')
        print(f'L (steps per cycle) = {L}')
        # Create overlapped time values
        time_axis_overlapped = self.overlapped_values(np.array(target).shape[0], L)
        print(f'✓ Time axis (overlapped): length = {len(time_axis_overlapped)}')
        print(f'✓ Time axis range: {time_axis_overlapped[0]} to {time_axis_overlapped[-1]}')
        # Flatten target trajectory for X dimension
        # target_array[:, :, 0] extracts X component from all cycles and all steps
        target_x_flat = target[:, :, 0].ravel()
        print(f'✓ Target X flattened: length = {len(target_x_flat)}\n')

        # Define zoom window
        zoom_mask = time_axis_overlapped <= zoom_window
        # ==================== X DIMENSION ANALYSIS (ONLY TARGET VS NATURE RUN - FULL) ====================
        fig = plt.figure(figsize=(12, 6))

        # ========== PLOT: X dimension - Full trajectory (Target vs Nature Run) ==========
        ax = plt.subplot(1, 1, 1)
        # Plot nature run with matching time axis
        ax.plot(time_axis_overlapped, x_nature[0, 0:len(time_axis_overlapped)], 
                label='Nature Run (X)', color='#95A5A6', linewidth=1.0, alpha=0.6)
        ax.plot(time_axis_overlapped, target_x_flat, label='Target (X) with Control', 
                color='green', linewidth=1.5)
        ax.set_title('X Dimension: Target vs Nature Run (Full)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('X Value', fontsize=12)
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=11)
        ax.set_xlim(0, time_axis_overlapped[-1])

        plt.tight_layout()
        plt.savefig(self.figures_path + 'Target_vs_Nature_Run_Full.png', dpi=300, bbox_inches='tight')
        print(f'✓ Figura guardada: {self.figures_path}Target_vs_Nature_Run_Full.png')
        plt.show()
        plt.close()


    def Plot_Target_Trajectory_Comparison_With_Interventions(self, x_nature, intervention_history=None, output_filename='Target_vs_Nature_Run_Full_Interventions.png'):
        # Target trajectory
        target = np.array(self.target)
        L = self.Ta
        K = target.shape[0]

        time_axis_overlapped = self.overlapped_values(K, L)
        target_x_flat = target[:, :, 0].ravel()

        if intervention_history is None:
            intervention_history = self.intervention_history

        intervention_indices = []
        for fallback_cycle, item in enumerate(intervention_history):
            norms = np.asarray(item.get('norms', []), dtype=float)
            if not item.get('controlled', bool(np.any(norms > 0))):
                continue

            cycle_index = item.get('cycle_index', fallback_cycle)
            if cycle_index is None or cycle_index < 0 or cycle_index >= K:
                continue

            for substep in np.where(norms > 0)[0]:
                if substep < L:
                    flat_index = int(cycle_index) * L + int(substep)
                    if flat_index < len(target_x_flat):
                        intervention_indices.append(flat_index)

        intervention_indices = np.unique(intervention_indices)

        fig = plt.figure(figsize=(12, 6))
        ax = plt.subplot(1, 1, 1)
        ax.plot(time_axis_overlapped, x_nature[0, 0:len(time_axis_overlapped)],
                label='Nature Run (X)', color='#95A5A6', linewidth=1.0, alpha=0.6)
        ax.plot(time_axis_overlapped, target_x_flat, label='Target (X) with Control',
                color='green', linewidth=1.5)

        if intervention_indices.size:
            ax.scatter(time_axis_overlapped[intervention_indices], target_x_flat[intervention_indices],
                       label='Intervention applied', color='tab:orange', edgecolors='black',
                       linewidths=0.3, s=22, alpha=0.9, zorder=5)

        ax.set_title('X Dimension: Target vs Nature Run (Intervention Points)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('X Value', fontsize=12)
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=11)
        ax.set_xlim(0, time_axis_overlapped[-1])

        plt.tight_layout()
        plt.savefig(self.figures_path + output_filename, dpi=300, bbox_inches='tight')
        print(f'Intervention points plotted: {len(intervention_indices)}')
        print(f'Figura guardada: {self.figures_path}{output_filename}')
        plt.show()
        plt.close()


    def Animate_3D_Multi_Trajectory(self, x_nature, trajectory_noisy, dt_output, 
                                      ensemble_members=[0, 1, 2, 3, 4],
                                      frame_skip=2, trail_length=100, 
                                      max_frames=680, fps=15,
                                      gif_filename=None):
        """
        Create a 3D animated GIF showing the evolution of multiple trajectories.
        
        Parameters:
        -----------
        x_nature : np.ndarray
            Nature run trajectory with shape (3, time_steps)
        trajectory_noisy : np.ndarray
            Ensemble trajectories with shape (n_ensemble, time_steps, 3)
        dt_output : float
            Time step for output
        ensemble_members : list, optional
            List of ensemble member indices to plot (default: [0, 1, 2, 3, 4])
        frame_skip : int, optional
            Use every Nth point to speed up animation (default: 2)
        trail_length : int, optional
            Number of previous points to show as trail (default: 100)
        max_frames : int, optional
            Limit frames for animation (default: 680)
        fps : int, optional
            Frames per second (default: 15)
        gif_filename : str, optional
            Output filename for GIF. If None, uses default name.
        
        Returns:
        --------
        None
            Saves the animation as a GIF file
        """
        from matplotlib.animation import FuncAnimation, PillowWriter
        
        # Set default filename if not provided
        if gif_filename is None:
            gif_filename = self.figures_path + 'lorenz63_3d_trajectory_multi.gif'
        
        # Colors for ensemble members
        colors = ['red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'pink', 'brown', 'gray']
        
        n_ensemble = len(ensemble_members)
        print(f"\n{'='*60}")
        print(f"  Creating animation with {n_ensemble} ensemble members")
        print(f"  Ensemble indices: {ensemble_members}")
        print(f"{'='*60}\n")
        
        # Prepare nature run data
        x_nature_transposed = x_nature.T  # Shape: (time_steps, 3)
        data_points_nature = x_nature_transposed[::frame_skip][:max_frames]
        n_frames = len(data_points_nature)
        
        # Prepare ensemble data - store in a list
        ensemble_data = []
        for idx in ensemble_members:
            data = trajectory_noisy[idx, ::frame_skip, :][:max_frames]
            ensemble_data.append(data)
            print(f"  Ensemble #{idx:2d} shape: {data.shape}")
        
        print(f"  Nature run shape: {data_points_nature.shape}")
        print(f"  Total frames: {n_frames}\n")
        
        # Create figure and 3D axis
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Combine all data to set plot limits
        all_data = [data_points_nature] + ensemble_data
        all_x = np.concatenate([d[:, 0] for d in all_data])
        all_y = np.concatenate([d[:, 1] for d in all_data])
        all_z = np.concatenate([d[:, 2] for d in all_data])
        
        ax.set_xlim([all_x.min() - 2, all_x.max() + 2])
        ax.set_ylim([all_y.min() - 2, all_y.max() + 2])
        ax.set_zlim([all_z.min() - 2, all_z.max() + 2])
        
        # Labels
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title(f'Lorenz63 Multi-Trajectory Evolution\n(Nature Run + {n_ensemble} Ensemble Members)', 
                     fontsize=14, pad=20)
        
        # Initialize plot objects for nature run (blue)
        trail_nature, = ax.plot([], [], [], 'b-', alpha=0.8, linewidth=2.5, label='Nature Run')
        point_nature, = ax.plot([], [], [], 'bo', markersize=9)
        full_traj_nature, = ax.plot([], [], [], 'b-', alpha=0.15, linewidth=0.5)
        
        # Show full nature trajectory in background
        full_traj_nature.set_data(data_points_nature[:, 0], data_points_nature[:, 1])
        full_traj_nature.set_3d_properties(data_points_nature[:, 2])
        
        # Initialize plot objects for ensemble members dynamically
        ensemble_trails = []
        ensemble_points = []
        ensemble_full_trajs = []
        
        for i, idx in enumerate(ensemble_members):
            color = colors[i % len(colors)]  # Cycle through colors if needed
            
            # Create trail, point, and full trajectory for this ensemble member
            trail, = ax.plot([], [], [], color=color, alpha=0.6, linewidth=1.5, 
                             label=f'Ensemble #{idx}')
            point, = ax.plot([], [], [], 'o', color=color, markersize=6)
            full_traj, = ax.plot([], [], [], '-', color=color, alpha=0.15, linewidth=0.5)
            
            # Show full trajectory in background
            full_traj.set_data(ensemble_data[i][:, 0], ensemble_data[i][:, 1])
            full_traj.set_3d_properties(ensemble_data[i][:, 2])
            
            ensemble_trails.append(trail)
            ensemble_points.append(point)
            ensemble_full_trajs.append(full_traj)
        
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        
        # Text annotation for frame number
        time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=10)
        
        def init():
            """Initialize animation"""
            # Initialize nature run
            trail_nature.set_data([], [])
            trail_nature.set_3d_properties([])
            point_nature.set_data([], [])
            point_nature.set_3d_properties([])
            
            # Initialize all ensemble members
            artists = [trail_nature, point_nature]
            for trail, point in zip(ensemble_trails, ensemble_points):
                trail.set_data([], [])
                trail.set_3d_properties([])
                point.set_data([], [])
                point.set_3d_properties([])
                artists.extend([trail, point])
            
            time_text.set_text('')
            artists.append(time_text)
            return tuple(artists)
        
        def update(frame):
            """Update function for animation"""
            # Calculate trail indices
            start_idx = max(0, frame - trail_length)
            
            # Update nature run
            trail_x_nat = data_points_nature[start_idx:frame+1, 0]
            trail_y_nat = data_points_nature[start_idx:frame+1, 1]
            trail_z_nat = data_points_nature[start_idx:frame+1, 2]
            trail_nature.set_data(trail_x_nat, trail_y_nat)
            trail_nature.set_3d_properties(trail_z_nat)
            point_nature.set_data([data_points_nature[frame, 0]], [data_points_nature[frame, 1]])
            point_nature.set_3d_properties([data_points_nature[frame, 2]])
            
            # Update all ensemble members
            artists = [trail_nature, point_nature]
            for i, (trail, point, data) in enumerate(zip(ensemble_trails, ensemble_points, ensemble_data)):
                trail_x = data[start_idx:frame+1, 0]
                trail_y = data[start_idx:frame+1, 1]
                trail_z = data[start_idx:frame+1, 2]
                trail.set_data(trail_x, trail_y)
                trail.set_3d_properties(trail_z)
                point.set_data([data[frame, 0]], [data[frame, 1]])
                point.set_3d_properties([data[frame, 2]])
                artists.extend([trail, point])
            
            # Update text
            actual_time = frame * frame_skip * dt_output
            time_text.set_text(f'Frame: {frame}/{n_frames-1}\nTime: {actual_time:.2f}')
            
            # Rotate the view slightly for dynamic effect
            ax.view_init(elev=20, azim=frame * 0.5)
            
            artists.append(time_text)
            return tuple(artists)
        
        # Create animation
        print("Generating animation frames...")
        anim = FuncAnimation(fig, update, init_func=init, frames=n_frames, 
                             interval=1000/fps, blit=False, repeat=True)
        
        # Save as GIF
        print(f"Saving animation to {gif_filename}...")
        writer = PillowWriter(fps=fps)
        anim.save(gif_filename, writer=writer, dpi=100)
        
        print(f"✓ Animation saved successfully!")
        print(f"  File: {gif_filename}")
        print(f"  Total frames: {n_frames}")
        print(f"  Duration: {n_frames/fps:.2f} seconds")
        
        plt.show()
        plt.close(fig)
    
    def plot_CLE_comparison_all_modes(self, modes_data, lambda_theoretical, PathFigures):
        """
        Create a bar chart comparison of Conditional Lyapunov Exponents across all modes.
        
        Parameters:
        -----------
        modes_data : dict
            Dictionary with mode data: {'z_only': λ_value, 'xy_only': λ_value, 'all': λ_value}
            Values can be None if mode data is not available.
        lambda_theoretical : float
            Theoretical Lyapunov exponent value for comparison.
        PathFigures : str
            Path where the figure will be saved.
        
        Returns:
        --------
        dict
            Summary statistics and best mode information.
        """
        import os
        
        # Define mode information
        modes_info = {
            'z_only': {'label': 'Z-only', 'color': '#3498db'},
            'xy_only': {'label': 'XY-only', 'color': '#e74c3c'},
            'all': {'label': 'ALL', 'color': '#2ecc71'}
        }
        
        # Create bar chart comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for plotting
        labels = []
        values = []
        colors = []
        bar_positions = []
        
        # Add each mode if data exists
        position = 0
        for mode_key in ['z_only', 'xy_only', 'all']:
            if modes_data[mode_key] is not None:
                labels.append(modes_info[mode_key]['label'])
                values.append(modes_data[mode_key])
                colors.append(modes_info[mode_key]['color'])
                bar_positions.append(position)
                position += 1
        
        # Add theoretical value
        labels.append('Theoretical\n(Nature)')
        values.append(lambda_theoretical)
        colors.append('#95a5a6')
        bar_positions.append(position)
        
        # Create bar chart
        bars = ax.bar(bar_positions, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on top of bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.4f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Calculate improvements
        if all(modes_data[k] is not None for k in modes_data.keys()):
            for i, mode_key in enumerate(['z_only', 'xy_only', 'all']):
                improvement = (1 - modes_data[mode_key]/lambda_theoretical) * 100
                ax.text(bar_positions[i], 0.05, f'{improvement:+.1f}%',
                        ha='center', va='bottom', fontsize=10, 
                        color='white', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.9))
        
        # Customize plot
        ax.set_ylabel('Conditional Lyapunov Exponent (λ)', fontsize=14, fontweight='bold')
        ax.set_title('CLE Comparison: Effect of Selective Control on Lorenz-63 Stability', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, max(values) * 1.15)
        
        # Add horizontal line for theoretical value
        ax.axhline(y=lambda_theoretical, color='gray', linestyle='--', linewidth=2, 
                   alpha=0.5, label='Theoretical LLE')
        
        # Add legend with interpretation
        legend_elements = [
            plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=2, label='Theoretical LLE'),
            plt.Rectangle((0, 0), 1, 1, fc='#3498db', alpha=0.8, label='Z-only'),
            plt.Rectangle((0, 0), 1, 1, fc='#e74c3c', alpha=0.8, label='XY-only'),
            plt.Rectangle((0, 0), 1, 1, fc='#2ecc71', alpha=0.8, label='ALL')
        ]
        ax.legend(handles=legend_elements, loc='upper center', fontsize=10, framealpha=0.9)
        
        # Add annotation box with summary
        summary_stats = {}
        if all(modes_data[k] is not None for k in modes_data.keys()):
            best_mode = min(modes_data.keys(), key=lambda k: modes_data[k])
            best_mode_label = modes_info[best_mode]['label']
            best_improvement = (1 - modes_data[best_mode]/lambda_theoretical) * 100
            
            # Store summary statistics
            summary_stats = {
                'best_mode': best_mode,
                'best_mode_label': best_mode_label,
                'best_improvement': best_improvement,
                'best_lambda': modes_data[best_mode]
            }
            
            textstr = f'Best Control Strategy: {best_mode_label}\n'
            textstr += f'Stability Improvement: {best_improvement:.1f}%\n'
            textstr += f'λ reduction: {lambda_theoretical - modes_data[best_mode]:.4f}'
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                    verticalalignment='top', bbox=props, fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save figure
        mode_comparison_filename = os.path.join(PathFigures, 'CLE_Comparison_All_Modes.png')
        plt.savefig(mode_comparison_filename, dpi=300, bbox_inches='tight')
        print(f"\n{'='*70}")
        print(f"✓ Figure saved: {os.path.basename(mode_comparison_filename)}")
        print(f"{'='*70}\n")
        
        plt.show()
        
        # Print detailed comparison
        if all(modes_data[k] is not None for k in modes_data.keys()):
            print(f"\n{'='*70}")
            print(f"  DETAILED COMPARISON")
            print(f"{'='*70}\n")
            print(f"  {'Mode':<15} {'λ_CLE':<12} {'Δλ':<12} {'Improvement':<12}")
            print(f"  {'-'*51}")
            print(f"  {'Theoretical':<15} {lambda_theoretical:<12.6f} {'-':<12} {'baseline':<12}")
            
            for mode_key in ['z_only', 'xy_only', 'all']:
                mode_label = modes_info[mode_key]['label']
                lambda_val = modes_data[mode_key]
                delta = lambda_val - lambda_theoretical
                improvement = (1 - lambda_val/lambda_theoretical) * 100
                print(f"  {mode_label:<15} {lambda_val:<12.6f} {delta:<+12.6f} {improvement:>10.2f}%")
            
            print(f"\n{'='*70}\n")
        
        return summary_stats


def plot_controlled_vs_free_nature_run(
    evo_history,
    x_nature,
    t0,
    n_completed,
    Ta,
    y_obs,
    yo_history,
    branch_history,
    perturbed_time,
    xdim,
    path_figures,
    output_filename='nr_controlled_vs_nr_qiwenevents.png',
):
    """Plot the controlled and free nature runs with CSE cycle diagnostics."""
    plot_steps = np.arange(n_completed * Ta + 1)
    cycle_steps = np.arange(n_completed) * Ta
    obs_steps = np.arange(1, n_completed + 1) * Ta

    controlled_traj = evo_history[:, :n_completed * Ta + 1]
    free_traj = x_nature[:, t0:t0 + n_completed * Ta + 1]
    cycle_controlled = evo_history[:, cycle_steps]

    y_obs_array = np.asarray(y_obs)
    if y_obs_array.shape[0] == xdim:
        osse_obs = y_obs_array[:, t0 + obs_steps]
    else:
        osse_obs = y_obs_array[t0 + obs_steps, :].T
    cse_obs = np.asarray(yo_history).T

    branch_history_array = np.asarray(branch_history)
    free_mask = branch_history_array == 'free'
    current_safe_mask = branch_history_array == 'current_safe_control'
    past_safe_mask = branch_history_array == 'past_safe_control'

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(18, 14),
        sharex=True,
        constrained_layout=True,
    )
    var_labels = ['x', 'y', 'z']

    for j, ax in enumerate(axes):
        ax.plot(
            plot_steps,
            controlled_traj[j],
            color='red',
            linewidth=2.0,
            label='NR controlled' if j == 0 else None,
        )
        ax.plot(
            plot_steps,
            free_traj[j],
            color='black',
            linewidth=1.8,
            label='NR' if j == 0 else None,
        )
        ax.axhline(0, color='gray', linestyle='--', linewidth=1.0)

        if np.any(free_mask):
            ax.scatter(
                cycle_steps[free_mask],
                cycle_controlled[j, free_mask],
                color='tab:blue',
                s=45,
                zorder=5,
                label='no control required' if j == 0 else None,
            )
        if np.any(current_safe_mask):
            ax.scatter(
                cycle_steps[current_safe_mask],
                cycle_controlled[j, current_safe_mask],
                color='green',
                marker='o',
                s=50,
                zorder=6,
                label='control applied' if j == 0 else None,
            )
        if np.any(past_safe_mask):
            ax.scatter(
                cycle_steps[past_safe_mask],
                cycle_controlled[j, past_safe_mask],
                color='green',
                marker='D',
                edgecolor='black',
                linewidth=0.6,
                s=55,
                zorder=7,
                label='control applied: all members shift' if j == 0 else None,
            )

        ax.scatter(
            obs_steps,
            osse_obs[j],
            facecolors='none',
            edgecolors='black',
            marker='o',
            s=30,
            linewidth=0.9,
            zorder=4,
            label='OSSE observations around NR' if j == 0 else None,
        )
        ax.scatter(
            obs_steps,
            cse_obs[j],
            facecolors='none',
            edgecolors='tab:purple',
            marker='s',
            s=30,
            linewidth=0.9,
            zorder=8,
            label='CSE observations around NR controlled' if j == 0 else None,
        )

        if perturbed_time.size:
            marker_times = perturbed_time[perturbed_time < n_completed * Ta]
            ax.scatter(
                marker_times,
                np.full(marker_times.shape, 0.025),
                color='gray',
                s=12,
                alpha=0.75,
                transform=ax.get_xaxis_transform(),
                clip_on=False,
                label='perturbed steps' if j == 0 else None,
            )

        ax.set_ylabel(var_labels[j])
        ax.grid(True, alpha=0.18)

    axes[-1].set_xlabel('Relative time step from t0')
    axes[0].set_title('NR controlled vs NR')
    axes[0].legend(loc='best', ncols=2)
    output_path = path_figures / output_filename
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.show()

    return fig, axes, output_path
