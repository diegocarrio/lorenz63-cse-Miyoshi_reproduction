# Loading Python Libraries:
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint


# Global parameters for the Lorenz63 system
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

def lorenz63(t,x):
    d = np.zeros(3)
    d[0] = sigma * (x[1] - x[0])
    d[1] = x[0] * (rho - x[2]) - x[1]
    d[2] = x[0] * x[1] - beta * x[2]
    # Return the state derivatives
    return d


class run_RK4:
    def __init__(self, func, sigma, beta, rho, initial_state, h, t_final, str_figures_path='unknown'):
        """
        Initializes the RK4 solver object for numerical integration.

        Parameters:
          func: function defining the system of ODEs.
          initial_state: vector with the initial state [x0, y0, z0].
          h: integration step size.
          t_final: final time for the integration.
        """
        self.func = func
        self.state = np.array(initial_state, dtype=float)         # Convert initial state to a float array
        self.h = h                                                # Step size
        self.t_final = t_final                                    # Final integration time
        self.num_steps = int(t_final / h)                         # Total number of steps
        self.times = np.linspace(0, t_final, self.num_steps + 1)  # Time vector
        #
        # Array to store the trajectory: each row corresponds to the state at a given time
        #
        self.trajectory = np.empty((self.num_steps + 1, len(initial_state)))
        self.trajectory[0] = self.state  # Store the initial state
        #
        # Lorenz63 parameters
        #
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.figures_path = str_figures_path


    def rk4_step(self, t):
        """
        Performs a single step of the RK4 integration.

        Parameters:
          t: current time.

        Returns:
          The new state after a single RK4 step.
        """

        h = self.h
        f = self.func

        #print("h: ", h)
        k1 = h * f(self.state, t)
        k2 = h * f(self.state + 0.5 * k1 , t + 0.5 * h)
        k3 = h * f(self.state + 0.5 * k2 , t + 0.5 * h)
        k4 = h * f(self.state + k3 , t + h)
        return self.state + (k1 + 2 * k2 + 2 * k3 + k4) / 6


    def integrate(self):
        """
        Integrates the system of ODEs using the RK4 method.
        """
        for i, t in enumerate(self.times[:-1]):
            self.state = self.rk4_step(t)
            self.trajectory[i + 1] = self.state
        return self.trajectory


    def run(self, x0, t, t_output):
        x = solve_ivp(lorenz63, (t[0],t[-1]), x0, t_eval = t_output)
        self.trajectory = x.y.T
        return self.trajectory


    def plot_trajectory(self):
        """
        Plots the trajectory of the system.
        """
        # Plot x vs. time
        fig = plt.figure(figsize=(12, 6))
        ax  = fig.add_subplot(111)
        print("times size: ", self.times.size)
        plt.plot(self.times[0:5001], self.trajectory[0:5001, 0], linestyle = 'dashed', marker = '.', markersize = 1 ,label='NR before control', color='blue')

        # Add horizontal and vertical lines
        plt.axhline(y=0, color='red', linestyle='--', linewidth=1)  # Horizontal line at x = 0
        plt.axvline(x=0, color='green', linestyle='--', linewidth=1)  # Vertical line at Time = 0

        # Set x-ticks every 50 time steps
        ticks = np.arange(self.times[0], self.times[5000], 500 * (self.times[1]-self.times[0]))
        ax.set_xticks(ticks)
        print("ticks size: ", ticks.size)
        ax.set_xticklabels(range(0, 500*ticks.size, 500))

        # Add labels, title, and legend
        plt.xlabel('Time Steps (1 step = 0.01)', fontsize=12)
        plt.ylabel('State x', fontsize=12)
        plt.title('Evolution Variable X [No Control]', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()
        fig.savefig(self.figures_path + 'lorenz63_nr.png')


    def plot_3d_trajectory(self):
        """
        Plots the trajectory of the system in 3D.
        """
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1,1,1, projection='3d')
        ax.scatter(self.trajectory[0:5000, 0], self.trajectory[0:5000, 1], self.trajectory[0:5000, 2], label='NR before control', c=np.linspace(0, 1, len(self.trajectory[0:5000,:])) ,cmap='viridis', s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Lorenz63 System Attractor [No Control]')
        ax.legend()
        plt.show()
        fig.savefig(self.figures_path + 'lorenz63_3d_nr.png')


    # Plot nature run and observations:
    def plot_nature_obs(self, nature, obs, t0, tf, obs_freq, str_figures_path):
        """
        Plots the nature run and its associated observations.
        """
        fig = plt.figure(figsize=(12, 14))
        ax = fig.add_subplot(311)
        plt.plot(nature[0, t0:tf], linestyle='dashed', marker='.', markersize=3, color='blue', label="Nature Run")
        plt.scatter(np.arange(0,tf - t0,obs_freq), obs[0, t0:tf:obs_freq], marker='o', s=25, facecolors='none', edgecolors='red', label="Observations")
        plt.axhline(y=0, linestyle='dashed', color='grey')
        plt.axvline(x=0, linestyle='dashed', color='grey')
        plt.axvline(x=300, linestyle='dashed', color='grey')
        plt.ylabel('State x', fontsize=14)
        plt.title('Lorenz-63 Nature Run and Observations', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim(-18, 18)
        plt.xlim(-5, (tf - t0) + 5)
        ax.legend(fontsize=12)
        #
        ax = fig.add_subplot(312)
        plt.plot(nature[1, t0:tf], linestyle='dashed', marker='.', markersize=3, color='blue', label="Nature Run")
        plt.scatter(np.arange(0,tf - t0,obs_freq), obs[1, t0:tf:obs_freq], marker='o', s=25, facecolors='none', edgecolors='red', label="Observations")
        plt.axhline(y=0, linestyle='dashed', color='grey')
        plt.axvline(x=0, linestyle='dashed', color='grey')
        plt.axvline(x=300, linestyle='dashed', color='grey')
        plt.ylabel('State y', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(-5, (tf - t0) + 5)
        ax.legend(fontsize=12)
        #
        ax = fig.add_subplot(313)
        plt.plot(nature[2, t0:tf], linestyle='dashed', marker='.', markersize=3, color='blue', label="Nature Run")
        plt.scatter(np.arange(0,tf - t0,obs_freq), obs[2, t0:tf:obs_freq], marker='o', s=25, facecolors='none', edgecolors='red', label="Observations")
        plt.axhline(y=0, linestyle='dashed', color='grey')
        plt.axvline(x=0, linestyle='dashed', color='grey')
        plt.axvline(x=300, linestyle='dashed', color='grey')
        plt.xlabel('Time step (1 step = 0.01)', fontsize=14)
        plt.ylabel('State y', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(-5, (tf - t0) + 5)
        ax.legend(fontsize=12)
        fig.savefig(str_figures_path + 'Lorenz63_nature_run_observations_xyz', bbox_inches='tight')
        plt.show()
        plt.close()

