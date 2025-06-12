import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from tqdm import tqdm

class Problem_Solver:

    def __init__(self, 
                 n = 1000, 
                 dt = 0.01, 
                 tol = 1e-6, 
                 alpha = 0.35, 
                 beta = 0.5, 
                 gamma = 0.15, 
                 tilde_sigma = 0.3, 
                 delta = 0.05, 
                 a_0 = 0.5, 
                 a_1 = 1, 
                 a_2 = 5, 
                 a_3 = 2.5, 
                 x_min = 0.1, 
                 x_max = 0.9
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
        return self.a_0 + self.a_1 * x + self.a_2 * x * (1 - eta)**2 + self.a_3 * x * rho**2

    def objective_func(self, x_initial, x_grid, eta, rho, if_plot=False): 
        """
        Simulate the stochastic process and compute the integral of the discounted reward function.

        Parameters:
        - x_initial: Initial state.
        - eta: Control η (array of size self.n).
        - rho: Control ρ (scalar or array of size self.n).
        - if_plot: Whether to plot the path of X_t.

        Returns:
        - t: Time points.
        - X: Simulated state path.
        - integral: Integral of the discounted reward function.
        """
        # Interpolate eta as a function of x
        eta_func = lambda x_eval: np.interp(x_eval, x_grid, eta)
        rho_func = lambda x_eval: np.interp(x_eval, x_grid, rho)

        t = [0]
        X = [x_initial]
        integral = 0.0

        T_stopped = 75

        # Simulate the whole sample path of process X
        while t[-1] < T_stopped: 
            # Evaluate Markov control eta and rho at the current state X_t
            eta_t = eta_func(X[-1])
            rho_t = rho_func(X[-1])  

            # Compute drift and diffusion
            b = self.b_func(X[-1], eta_t, rho_t)
            sigma = self.sigma_func(X[-1])

            # Simulate the next state using Euler-Maruyama
            dW = np.sqrt(self.dt) * np.random.randn()
            X_next = X[-1] + b * self.dt + sigma * dW
            t_next = t[-1] + self.dt

            # Append the new state and time
            t.append(t_next)
            X.append(X_next)

        # Compute the reward and its discounted value
        f_X = self.f_func(np.array(X), eta_func(X), rho_func(X))
        discounted_f_X = np.exp(-self.delta * np.array(t)) * f_X
        # Compute the integral using the Simpson rule
        integral = simpson(discounted_f_X, t)

        if if_plot:
            plt.figure(figsize=(10, 6))
            plt.plot(t, X, label="Path of X_t")
            plt.xlabel("Time (t)")
            plt.ylabel("State (X_t)")
            plt.title("Path of X_t over Time")
            plt.legend()
            plt.grid()
            plt.show()

        return t, X, integral
    
    def monte_carlo_value(self, 
                          x_initial, 
                          x_grid, 
                          eta, 
                          rho, 
                          max_simulations = 2000, 
                          if_plot = False, 
                          ):
        """
        Estimate the value function using Monte Carlo simulation with confidence interval and stability-based stopping rules.

        Parameters:
        - x_initial: Initial state.
        - eta: Control η.
        - rho: Control ρ.
        - epsilon: Desired tolerance for the confidence interval.
        - delta: Stability threshold for the value function.
        - M: Window size for stability-based stopping rule.
        - max_simulations: Maximum number of simulations to prevent infinite loops.
        - if_plot: Whether to plot the convergence of the estimated value and confidence interval width.

        Returns:
        - Estimated value function.
        """
        estimated_value = 0.0  # Initialize the value estimate
        cumulative_rewards = []  # Store cumulative rewards for standard deviation calculation
        sim = 0  # Simulation counter
        convergence = []  # Store the estimated value at each step for plotting

        with tqdm(total=max_simulations, desc="Monte Carlo Simulation", unit="sim") as pbar:
            while True:
                sim += 1
                # Use the existing objective_func to simulate the path and compute the reward
                _, _, cumulative_reward = self.objective_func(x_initial, x_grid, eta, rho)

                # Store the cumulative reward for standard deviation calculation
                cumulative_rewards.append(cumulative_reward)

                if sim >= max_simulations:
                    print(f"Simulated with maximum ({max_simulations}).")
                    break
                
                pbar.update(1)

        estimated_value = np.mean(cumulative_rewards)  # Estimate the value function as the mean of cumulative rewards
        print(f"Estimated value function: {estimated_value}")

        if if_plot:
            # Plot the convergence of the estimated value
            plt.subplots(2, 1, figsize=(10, 12))
            plt.plot(range(1, len(convergence) + 1), convergence, label="Estimated Value")
            plt.axhline(estimated_value, color='r', linestyle='--', label=f"Converged Value: {estimated_value:.4f}")
            plt.set_xlabel("Number of Simulations")
            plt.set_ylabel("Estimated Value")
            plt.set_title("Convergence of Monte Carlo Simulation")
            plt.legend()
            plt.grid()

            # Adjust layout and show the figure
            plt.tight_layout()
            plt.show()

        return estimated_value

    def Bellman_solver(self, x_grid, v0, v1, eta, rho, if_plot = False):
        """
        Solve the Bellman equation using the shooting method with boundary value problem solver.

        Parameters:
        - v0: Boundary condition at x=0.1.
        - v1: Boundary condition at x=0.9.
        - eta: Control η.
        - rho: Control ρ.
        - if_plot: Whether to plot the solution.

        Returns:
        - x_sol: The grid points where the solution is computed.
        - v_sol: The value function solution.
        - p_sol: The derivative of the value function (dv/dx).
        """
        # Interpolate eta and rho to handle any input shape
        eta_interp = interp1d(x_grid, eta, kind='cubic', fill_value='extrapolate')
        rho_interp = interp1d(x_grid, rho, kind='cubic', fill_value='extrapolate')

        def bellman_rhs(x, V):
            v, p = V
            eta_val = eta_interp(x)  # Interpolated η
            rho_val = rho_interp(x)  # Interpolated ρ
            b_val = self.b_func(x, eta_val, rho_val)
            sigma_val = self.sigma_func(x)
            f_val = self.f_func(x, eta_val, rho_val)
            dv_dx = p
            dp_dx = (self.delta * v - b_val * p - f_val) / (0.5 * sigma_val**2)
            return np.array([dv_dx, dp_dx])

        def boundary_conditions(Va, Vb):
            return [Va[0] - v0, Vb[0] - v1]

        x = x_grid

        # Initial guess for the value function and its derivative
        v_guess = v0 + (v1 - v0) * (x - x.min()) **2 / (x.max() - x.min()) **2 # Linear
        p_guess = np.gradient(v_guess, x)

        V_guess = np.vstack([v_guess, p_guess])

        solution = solve_bvp(bellman_rhs, boundary_conditions, x, V_guess, max_nodes=int(1e6), tol=1e-5)

        if solution.success:
            print("Solver converged successfully.")
        else:
            print("Solver failed to converge.")

        x_sol = solution.x
        v_sol = solution.y[0]  # Value function
        p_sol = solution.y[1]  # Derivative of the value function

        if if_plot:
            plt.figure(figsize=(10, 8))
            plt.plot(x_sol, v_sol, label="Value Function")
            # plt.plot(x_sol, p_sol, label="Derivative of Value Function (dv/dx)", linestyle='--')
            plt.xlabel("State (x)")
            plt.ylabel("Value / Derivative")
            plt.title("Solution to Bellman ODE and its Derivative")
            plt.legend()
            plt.grid()
            plt.show()

        return x_sol, v_sol, p_sol
    
    def L2_norm(self, x1, x2, y1, y2): 
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
        if len(x1) == len(x2) and len(x1) == len(self.x_grid):
            # If lengths are the same, directly compute the L2 norm
            y1_interp = y1
            y2_interp = y2
        else:
            # Interpolate y1 to the length of y2
            y1_interp  = interp1d(x1, y1, kind = 'cubic', fill_value = 'extrapolate')(self.x_grid)
            y2_interp = interp1d(x2, y2, kind = 'cubic', fill_value = 'extrapolate')(self.x_grid)
        # Compute the normalized L2 norm
        return np.linalg.norm(y1_interp - y2_interp) / np.sqrt(len(self.x_grid))

    def policy_improve(self, eta_guess, rho_guess, if_plot = False, if_compare = True): 
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

        x, v_int, p_int = self.Bellman_solver(self.x_grid, v0, v1, eta_guess, rho_guess, if_plot=False)
        if if_compare:
            x_begin = x.copy()
            v_begin = v_int.copy()
        iter = 0
        error = []

        while True:
            x_previous = x.copy()

            eta_new = np.clip(
                (2 * x * self.a_2 - self.alpha * (1 - x) * p_int) / (2 * x * self.a_2 + 2 * self.beta * x * (1 - x) * p_int), 
                0, 1)
            rho_new = np.maximum(0, p_int / (2 * self.a_3))

            v0_new = self.monte_carlo_value(self.x_min, x, eta_new, rho_new)
            v1_new = self.monte_carlo_value(self.x_max, x, eta_new, rho_new)

            x, v_update, p_update = self.Bellman_solver(x, v0_new, v1_new, eta_new, rho_new, if_plot=False)

            if self.L2_norm(x_previous, x, v_int, v_update) <= self.tol_main:
                print(f"Updated value function: {v_update}")
                print(f"Updated controls: η = {eta_new}, ρ = {rho_new}")
                break

            if iter >= 5:
                print(f"Maximum iterations {iter} reached.")
                print(f"Updated value function: {v_update}")
                print(f"Updated controls: η = {eta_new}, ρ = {rho_new}")
                break

            error.append(self.L2_norm(x_previous, x, v_int, v_update))
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

            # Plot the value function in a separate figure
            if if_compare:
                plt.figure(figsize=(10, 6))
                plt.plot(x_begin, v_begin, label="Initial Value Function", color='orange', linestyle='--')
                plt.plot(x, v_update, label="Final Value Function", color='purple')
                plt.xlabel("State (x)")
                plt.ylabel("Value (v)")
                plt.title("Comparison of Initial and Final Value Functions", fontsize=14)
                plt.legend()
                plt.tight_layout()
                plt.show()

            plt.figure(figsize=(10, 6))
            plt.plot(x, v_update, label="Value Function", color='purple')
            plt.xlabel("State (x)")
            plt.ylabel("Value (v)")
            plt.title("Solution to Bellman ODE", fontsize=14) 
            plt.legend()
            plt.tight_layout()
            plt.show()

            # Plot the controls η and ρ in another separate figure
            plt.figure(figsize=(10, 6))
            x_small = np.linspace(self.x_min, self.x_max, 500)
            eta_plotting = interp1d(x_previous, eta_new, kind='cubic', fill_value='extrapolate')(x_small)
            rho_plotting = interp1d(x_previous, rho_new, kind='cubic', fill_value='extrapolate')(x_small)
            plt.plot(x_small, eta_plotting, label="Control η", color='blue')
            plt.plot(x_small, rho_plotting, label="Control ρ", color='red')
            plt.xlabel("State (x)")
            plt.ylabel("Control")
            plt.title("Controls η and ρ over State", fontsize=14)  
            plt.legend()
            plt.tight_layout()
            plt.show()

        x_update = x.copy()

        return eta_new, rho_new, v_update, x_previous, x_update
    
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


def main():
    """
    Main function to demonstrate the usage of the Problem_Solver class.
    """
    # Create an instance of the Problem_Solver
    solver = Problem_Solver()

    # Define initial state and controls
    eta = 0 * np.ones(solver.n)  # Initial guess of constant control η
    rho = np.zeros(solver.n)  # Initial guess of constant control ρ

    # Run the policy improvement algorithm
    eta_opt, rho_opt, v_opt, x_pre, x_upd = solver.policy_improve(eta, rho, if_plot = True, if_compare = True)
    # solver.sensitivity_steps(x_pre, x_upd, eta_opt, rho_opt, v_opt)
    # print(solver.objective_func(
    #     x_initial=0.1, 
    #     x_grid=solver.x_grid, 
    #     eta=eta, 
    #     rho=rho, 
    #     if_plot=True
    # )[2])

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()