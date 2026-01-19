import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from tqdm import tqdm

import DGM

class Problem_Solver:

    def __init__(self, 
                 n = 1000, 
                 dt = 0.01, 
                 tol = 1e-4, 
                 alpha = 0.5, 
                 beta = 0.5, 
                 gamma = 0.15, 
                 tilde_sigma = 0.3, 
                 delta = 0.05, 
                 a_0 = 0, 
                 a_1 = 5, # a_I
                 a_2 = 2, # a^I_m - a^S_m
                 a_3 = 0.5, # a_r
                 a_4 = 0.5, # a^S_m
                 x_min = 0.005, 
                 x_max = 0.995
                 ): 
        
        # Parameters
        self.n = n
        self.h = (x_max - x_min) / n  # Step size for the state space
        self.dt = dt
        self.tol_main = tol

        # Problem-specific parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tilde_sigma = tilde_sigma
        self.delta = delta

        # Coefficients for the reward function
        self.a_0 = a_0
        self.a_1 = a_1
        self.a_2 = a_2
        self.a_3 = a_3
        self.a_4 = a_4

        # Truncated boundaries of controlled SDE
        self.x_min = x_min
        self.x_max = x_max
        self.x_grid = np.linspace(self.x_min, self.x_max, self.n)

    def b_func(self, x, eta, rho):
        """
        Drift function b(x, eta, rho) = η * α * (1 - x) + x * (η^2 * β * (1 - x) - (γ + ρ)).
        """
        return eta * self.alpha * (1 - x) + x * (eta ** 2 * self.beta * (1 - x) - (self.gamma + rho))

    def sigma_func(self, x):
        """
        Diffusion function σ(x) = 𝜏 * x * (1 - x).
        """
        return self.tilde_sigma * x * (1 - x)

    def f_func(self, x, eta, rho):
        """
        Reward function f(x, η, ρ) = a_0 + a_1 * x + a_2 * x * (1 - η)^2 + a_3 * x * ρ^2.
        """
        return self.a_0 + self.a_1 * x + (self.a_2 * x + self.a_4) * (1 - eta)**2 + self.a_3 * x * rho**2

    def objective_func(
        self,
        x_initial,
        x_grid,
        eta,
        rho,
        num_simulations = 2000, 
        T_stopped = 75, 
        if_plot = False
    ):
        """
        Vectorized version:
        - X_t can go beyond [x_min, x_max].
        - Controls η and ρ are evaluated using boundary values when X_t is out of bounds.
        """

        T = int(T_stopped / self.dt) + 1
        t = np.linspace(0, T_stopped, T)

        X = np.zeros((num_simulations, T))
        X[:, 0] = x_initial

        # Controls will clip X internally when evaluated:
        def eta_func(x_eval):
            x_clipped = np.clip(x_eval, self.x_min, self.x_max)
            return np.interp(x_clipped, x_grid, eta)

        def rho_func(x_eval):
            x_clipped = np.clip(x_eval, self.x_min, self.x_max)
            return np.interp(x_clipped, x_grid, rho)

        dW = np.sqrt(self.dt) * np.random.randn(num_simulations, T - 1)

        for k in range(1, T):
            X_prev = X[:, k - 1]

            eta_t = eta_func(X_prev)  # control evaluated at clipped X
            rho_t = rho_func(X_prev)

            b = self.b_func(X_prev, eta_t, rho_t)
            sigma = self.sigma_func(X_prev)

            X[:, k] = X_prev + b * self.dt + sigma * dW[:, k - 1]

        f_X = self.f_func(X, eta_func(X), rho_func(X))
        discount = np.exp(-self.delta * t)
        discounted_f_X = f_X * discount

        integral = simpson(discounted_f_X, t, axis=1)

        if if_plot:
            plt.figure(figsize=(10, 6))
            for i in range(min(20, num_simulations)):
                plt.plot(t, X[i], alpha=0.6)
            plt.xlabel("Time (t)")
            plt.ylabel("State (X_t)")
            plt.title(f"Sample Paths (Up to 20 of {num_simulations})")
            plt.grid(True)
            plt.show()

        return t, X, integral

    
    def monte_carlo_value(
        self,
        x_initial,
        x_grid,
        eta,
        rho,
        num_simulations = 2000,
        if_plot = False
    ):
        """
        Compute mean value function from vectorized simulations.

        Returns:
        - estimated_value: scalar mean of all path integrals.
        """
        _, _, cumulative_rewards = self.objective_func(
            x_initial=x_initial,
            x_grid=x_grid,
            eta=eta,
            rho=rho,
            num_simulations=num_simulations,
            if_plot=if_plot
        )

        estimated_value = np.mean(cumulative_rewards)
        print(f"Estimated value function: {estimated_value:.4f}")

        return estimated_value

    def L2_norm(self, y1, y2): 
        """
        Compute the L2 norm of the difference between two arrays of different length by interpolating.

        Parameters:
        - x1: First array of x values.
        - x2: Second array of x values.
        - y1: First array of y values.
        - y2: Second array of y values.
        Returns:
        - L2 norm of the difference.
        """
        # Compute the normalized L2 norm
        return np.linalg.norm(y1 - y2) / np.sqrt(len(self.x_grid))

    def policy_improve(self, eta_guess, rho_guess, if_plot = False): 
        """
        Policy improvement step for the Bellman equation.

        Parameters:
        - eta_guess: Initial guess for control η.
        - rho_guess: Initial guess for control ρ.
        - if_plot: Whether to plot the results.
        """
        # Initial guess for the value function
        v0 = self.monte_carlo_value(self.x_min, self.x_grid, eta_guess, rho_guess)
        v1 = self.monte_carlo_value(self.x_max, self.x_grid, eta_guess, rho_guess)

        # Initialization of Deep Galerkin Method (DGM) solver
        params = {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "tilde_sigma": self.tilde_sigma,
            "a_0": self.a_0,
            "a_1": self.a_1,
            "a_2": self.a_2,
            "a_3": self.a_3,
            "a_4": self.a_4,
            "delta": self.delta, 
            "v0_true": v0,  # Known boundary condition at x=0
            "v1_true": v1,   # Known boundary condition at x=1
            "x0": self.x_min,  # Default x0
            "x1": self.x_max,  # Default x1
            "n": self.n  # Default number of collocation points
        }

        DGM_solver = DGM.BellmanSolverDGM(
            x_grid_np = self.x_grid, 
            eta_np = eta_guess, 
            rho_np = rho_guess, 
            params = params
        )
        DGM_solver.train(n_epochs = 10000, n_collocation = 1000, print_every = 200, tol = 5e-3)
        v_int = DGM_solver.predict()[1].flatten()  # Initial value function from DGM
        p_int = DGM_solver.predict()[2].flatten()  # Initial derivative from DGM
        
        iter = 0
        error = []

        while True:
            eta_new = np.clip(
                1 - 
                ((self.alpha * (1 - self.x_grid) + self.beta * self.x_grid * (1 - self.x_grid)) * p_int) 
                / (2 * (self.a_2 * self.x_grid + self.a_4)), 
                0, 1)
            # eta_new = eta_guess.copy()
            rho_new = np.clip(p_int / (2 * self.a_3), 0, 10)

            # Compute the new BC using the updated controls
            v0_new = self.monte_carlo_value(self.x_min, self.x_grid, eta_new, rho_new)
            v1_new = self.monte_carlo_value(self.x_max, self.x_grid, eta_new, rho_new)

            # Update the DGM solver with new controls and boundary conditions
            DGM_solver.eta_np = eta_new
            DGM_solver.rho_np = rho_new
            DGM_solver.v0_true = v0_new
            DGM_solver.v1_true = v1_new

            DGM_solver.train(n_epochs = 50000, 
                             n_collocation = 1000, 
                             print_every = 200, 
                             learning_rate = 1e-3, 
                             tol = 5e-3, 
                             penalty = 0.1
                             )
            v_update = DGM_solver.predict()[1].flatten()  # Update the value function from DGM
            p_update = DGM_solver.predict()[2].flatten()  # Update the derivative from DGM

            if self.L2_norm(v_int, v_update) <= self.tol_main: 
                print(f"Updated value function: {v_update}")
                print(f"Updated controls: η = {eta_new}, ρ = {rho_new}")
                break

            if iter >= 5: 
                print(f"Maximum iterations {iter} reached.")
                print(f"Updated value function: {v_update}")
                print(f"Updated controls: η = {eta_new}, ρ = {rho_new}")
                break

            error.append(self.L2_norm(v_int, v_update))
            iter += 1
            v_int = v_update
            p_int = p_update

            print(f"------------------The {iter}th iteration done: The normalized L2 norm = {error[-1]}-------------------")

        if if_plot:
            # Plot the error in a separate figure
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(error)), error, label="Error", color='blue', marker='o')
            plt.xlabel("Iteration")
            plt.ylabel("Error")
            plt.title("Error Convergence", fontsize=14)
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()

            # Plot the value function in another separate figure
            plt.figure(figsize=(10, 6))
            plt.plot(self.x_grid, v_update, label="Value Function", color='green')
            plt.xlabel("State (x)")
            plt.ylabel("Value Function V(x)")
            plt.title("Value Function V(x) after Policy Improvement", fontsize=14)
            plt.legend()
            plt.tight_layout()
            

            # Plot the controls η and ρ in another separate figure
            plt.figure(figsize=(10, 6))
            x_small = np.linspace(self.x_min, self.x_max, 500)
            eta_plotting = interp1d(self.x_grid, eta_new, kind='cubic', fill_value='extrapolate')(x_small)
            rho_plotting = interp1d(self.x_grid, rho_new, kind='cubic', fill_value='extrapolate')(x_small)
            plt.plot(x_small, eta_plotting, label="Control η", color='blue')
            plt.plot(x_small, rho_plotting, label="Control ρ", color='red')
            plt.xlabel("State (x)")
            plt.ylabel("Control")
            plt.title("Controls η and ρ over State", fontsize=14)  
            plt.legend()
            plt.tight_layout()
            plt.show()

        return eta_new, rho_new, v_update
    
    def sensitivity_analysis(self, x, x_upd, eta, rho, v, baseline_eta = 0.5, baseline_rho = 1, if_plot=True):
        """
        Perform sensitivity analysis on the value function with respect to the controls η and ρ.

        Parameters:
        - eta: Control η.
        - rho: Control ρ.
        - baseline: Baseline value for the controls (percentage perturbation).
        - if_plot: Whether to plot the results.

        Returns:
        - sensitivity_results: A dictionary containing the value functions for perturbed controls.
        """
        # Define perturbations for η and ρ
        perturbations = {
            'eta_up': np.clip(eta + baseline_eta, 0, 1),
            'eta_down': np.clip(eta - baseline_eta, 0, 1),
            'rho_up': np.maximum(0, rho + baseline_rho),
            'rho_down': np.maximum(0, rho - baseline_rho)
        }

        # Compute the value function for each perturbation
        sensitivity_results = {}
        for key, perturbed_control in perturbations.items():
            print(f'...............Starting sensitivity analysis for {key}..................')
            if 'eta' in key:
                # Perturb η while keeping ρ constant
                v0 = self.monte_carlo_value(self.x_min, x, perturbed_control, rho)
                v1 = self.monte_carlo_value(self.x_max, x, perturbed_control, rho)
                x_sensitive, value, _ = self.Bellman_solver(x, v0, v1, perturbed_control, rho)
            elif 'rho' in key:
                # Perturb ρ while keeping η constant
                v0 = self.monte_carlo_value(self.x_min, x, eta, perturbed_control)
                v1 = self.monte_carlo_value(self.x_max, x, eta, perturbed_control)
                x_sensitive, value, _ = self.Bellman_solver(x, v0, v1, eta, perturbed_control)
            sensitivity_results[key] = (x_sensitive, value)

        # Plot the sensitivity analysis results if requested
        if if_plot:
            plt.figure(figsize=(10, 6))

            # Fetch the x and y values for each perturbation from dictionary
            x_eta_up, y_eta_up = sensitivity_results['eta_up']
            x_eta_down, y_eta_down = sensitivity_results['eta_down']
            x_rho_up, y_rho_up = sensitivity_results['rho_up']
            x_rho_down, y_rho_down = sensitivity_results['rho_down']

            plt.plot(x_eta_up, y_eta_up, label="η Up", color='blue', linestyle='--')
            plt.plot(x_eta_down, y_eta_down, label="η Down", color='blue', linestyle=':')
            plt.plot(x_rho_up, y_rho_up, label="ρ Up", color='red', linestyle='--')
            plt.plot(x_rho_down, y_rho_down, label="ρ Down", color='red', linestyle=':')
            plt.plot(x_upd, v, label="Baseline", color='black')
            plt.xlabel("State (x)")
            plt.ylabel("Value Function")
            plt.title("Sensitivity Analysis of Value Function")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()

        return sensitivity_results

    def sensitivity_steps(self, x, x_upd, eta, rho, v, if_plot=True): 
        """
        Perform stepwise sensitivity analysis on the value function with respect to the controls η and ρ.

        Parameters:
        - x: Original x grid.
        - x_upd: Updated x grid.
        - eta: Control η.
        - rho: Control ρ.
        - v: Baseline value function.
        - if_plot: Whether to plot the results.

        Returns:
        - sensitivity_results: A dictionary containing the value functions for each stepwise perturbation.
        """
        # Define stepwise perturbations for η and ρ
        eta_steps = [0.25, 0.5, 0.75]
        rho_steps = [0.5, 1, 2]

        sensitivity_results = {}

        # Stepwise perturbation for η (plus and down)
        for step in eta_steps:
            perturbed_eta_plus = np.clip(eta + step, 0, 1)
            v0 = self.monte_carlo_value(self.x_min, x, perturbed_eta_plus, rho)
            v1 = self.monte_carlo_value(self.x_max, x, perturbed_eta_plus, rho)
            x_sensitive, value, _ = self.Bellman_solver(x, v0, v1, perturbed_eta_plus, rho)
            key = f"eta_plus_{step}"
            sensitivity_results[key] = (x_sensitive, value)

            perturbed_eta_down = np.clip(eta - step, 0, 1)
            v0 = self.monte_carlo_value(self.x_min, x, perturbed_eta_down, rho)
            v1 = self.monte_carlo_value(self.x_max, x, perturbed_eta_down, rho)
            x_sensitive, value, _ = self.Bellman_solver(x, v0, v1, perturbed_eta_down, rho)
            key = f"eta_down_{step}"
            sensitivity_results[key] = (x_sensitive, value)

        # Stepwise perturbation for ρ (plus and down)
        for step in rho_steps:
            perturbed_rho_plus = np.maximum(0, rho + step)
            v0 = self.monte_carlo_value(self.x_min, x, eta, perturbed_rho_plus)
            v1 = self.monte_carlo_value(self.x_max, x, eta, perturbed_rho_plus)
            x_sensitive, value, _ = self.Bellman_solver(x, v0, v1, eta, perturbed_rho_plus)
            key = f"rho_plus_{step}"
            sensitivity_results[key] = (x_sensitive, value)

            perturbed_rho_down = np.maximum(0, rho - step)
            v0 = self.monte_carlo_value(self.x_min, x, eta, perturbed_rho_down)
            v1 = self.monte_carlo_value(self.x_max, x, eta, perturbed_rho_down)
            x_sensitive, value, _ = self.Bellman_solver(x, v0, v1, eta, perturbed_rho_down)
            key = f"rho_down_{step}"
            sensitivity_results[key] = (x_sensitive, value)

        # Plot eta_plus results in one figure
        if if_plot:
            # η plus
            plt.figure(figsize=(10, 6))
            plt.plot(x_upd, v, label="Baseline", color='black')
            for step in eta_steps:
                x_eta_plus, v_eta_plus = sensitivity_results[f"eta_plus_{step}"]
                plt.plot(x_eta_plus, v_eta_plus, label=f"η + {step}", linestyle='--')
            plt.xlabel("State (x)")
            plt.ylabel("Value Function")
            plt.title("Stepwise Sensitivity: η Increase")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()

            # η down
            plt.figure(figsize=(10, 6))
            plt.plot(x_upd, v, label="Baseline", color='black')
            for step in eta_steps:
                x_eta_down, v_eta_down = sensitivity_results[f"eta_down_{step}"]
                plt.plot(x_eta_down, v_eta_down, label=f"η - {step}", linestyle=':')
            plt.xlabel("State (x)")
            plt.ylabel("Value Function")
            plt.title("Stepwise Sensitivity: η Decrease")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()

            # ρ plus
            plt.figure(figsize=(10, 6))
            plt.plot(x_upd, v, label="Baseline", color='black')
            for step in rho_steps:
                x_rho_plus, v_rho_plus = sensitivity_results[f"rho_plus_{step}"]
                plt.plot(x_rho_plus, v_rho_plus, label=f"ρ + {step}", linestyle='--')
            plt.xlabel("State (x)")
            plt.ylabel("Value Function")
            plt.title("Stepwise Sensitivity: ρ Increase")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()

            # ρ down
            plt.figure(figsize=(10, 6))
            plt.plot(x_upd, v, label="Baseline", color='black')
            for step in rho_steps:
                x_rho_down, v_rho_down = sensitivity_results[f"rho_down_{step}"]
                plt.plot(x_rho_down, v_rho_down, label=f"ρ - {step}", linestyle=':')
            plt.xlabel("State (x)")
            plt.ylabel("Value Function")
            plt.title("Stepwise Sensitivity: ρ Decrease")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()

        return sensitivity_results

    def compare_a_groups(self, a_groups, eta_init=None, rho_init=None):
        """
        Compare value functions and controls for different (a_1, a_2, a_3) groups and plot them.

        Parameters:
        - a_groups: List of tuples [(a_1, a_2, a_3), ...]
        - eta_init: Optional initial eta array (default: zeros)
        - rho_init: Optional initial rho array (default: zeros)
        """
        if eta_init is None:
            eta_init = np.zeros(self.n)
        if rho_init is None:
            rho_init = np.zeros(self.n)

        legends = []
        v_list = []
        eta_list = []
        rho_list = []

        for a in a_groups:
            # Set coefficients
            self.beta = a

            # Solve for optimal policy and value function
            eta_opt, rho_opt, v_opt = self.policy_improve(eta_init, rho_init, if_plot=False)
            v_list.append(v_opt)
            eta_list.append(eta_opt)
            rho_list.append(rho_opt)
            legends.append(f"beta={a}")

        x = self.x_grid

        fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

        # Plot value functions
        for v, label in zip(v_list, legends):
            axs[0].plot(x, v, lw=2, label=label)
        axs[0].set_ylabel("Value Function")
        axs[0].set_title("Value Function for Different Groups")
        axs[0].legend()

        # Plot eta controls
        for eta, label in zip(eta_list, legends):
            axs[1].plot(x, eta, lw=2, label=label)
        axs[1].set_ylabel("Control η")
        axs[1].set_title("Optimal η for Different Groups")
        axs[1].legend()

        # Plot rho controls
        for rho, label in zip(rho_list, legends):
            axs[2].plot(x, rho, lw=2, label=label)
        axs[2].set_xlabel("State (x)")
        axs[2].set_ylabel("Control ρ")
        axs[2].set_title("Optimal ρ for Different Groups")
        axs[2].legend()

        plt.tight_layout()
        plt.show()

def main():
    solver = Problem_Solver()
    eta = 1 * np.ones(solver.n)
    rho = 1 * np.ones(solver.n)

    eta_opt, rho_opt, v = solver.policy_improve(eta, rho, if_plot=True)
    # solver.compare_a_groups(
    #     a_groups = [0.1, 0.35, 0.65, 0.9],
    #     eta_init = eta,
    #     rho_init = rho
    # )

    # Simulate for multiple initial points in [0.01, 0.99]

    # Run Bellman_solver for a single initial point
    # vx_01 = solver.monte_carlo_value(
    #     x_initial = solver.x_min,
    #     x_grid = solver.x_grid,
    #     eta = eta,
    #     rho = rho,
    #     num_simulations = 2000,
    #     if_plot = False
    # )

    # vx_02 = solver.monte_carlo_value(
    #     x_initial = solver.x_max,
    #     x_grid = solver.x_grid,
    #     eta = eta,
    #     rho = rho,
    #     num_simulations = 2000,
    #     if_plot = True
    # )

    # params = {
    #     "alpha": solver.alpha,
    #     "beta": solver.beta,
    #     "gamma": solver.gamma,
    #     "tilde_sigma": solver.tilde_sigma,
    #     "a_0": solver.a_0,
    #     "a_1": solver.a_1,
    #     "a_2": solver.a_2,
    #     "a_3": solver.a_3,
    #     "a_4": solver.a_4,
    #     "delta": solver.delta, 
    #     "v0_true": vx_01,  # Known boundary condition at x=0
    #     "v1_true": vx_02   # Known boundary condition at x=1
    # }

    # odesolver = DGM.BellmanSolverDGM(solver.x_grid, eta, rho, params)
    # odesolver.train(n_epochs = 10000, n_collocation = 250, learning_rate = 0.005)

    # V_vals, p_vals = odesolver.predict()

    # plt.plot(solver.x_grid, V_vals, label="DGM Value Function")
    # plt.xlabel("x")
    # plt.ylabel("V(x)")
    # plt.title("Bellman Equation Solution via Deep Galerkin Method")
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()