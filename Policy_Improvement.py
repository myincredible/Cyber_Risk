import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class Problem_Solver:

    def __init__(self, 
                 n = 1000, 
                 dt = 0.01, 
                 tol = 1e-6, 
                 alpha = 0.1, 
                 beta = 0.1, 
                 gamma = 0.2, 
                 tilde_sigma = 0.6, 
                 delta = 1, 
                 a_0 = 1, 
                 a_1 = 1, 
                 a_2 = 1, 
                 a_3 = 1, 
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
        Drift function b(x, eta, rho) = η * α * (1 - x) + x * (η * β * (1 - x) - (γ + ρ)).
        """
        return eta * self.alpha * (1 - x) + x * (eta * self.beta * (1 - x) - (self.gamma + rho))

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
        discounted_f_X = 1.0
        tol_valuefunction = 1e-2  # Tolerance for the value function

        while discounted_f_X > tol_valuefunction: 
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

            # Compute the reward and its discounted value
            f_X = self.f_func(X_next, eta_t, rho_t)
            discounted_f_X = np.exp(-self.delta * t_next) * f_X

            # Update the integral using the trapezoidal rule
            if len(t) > 1:
                integral += 0.5 * (discounted_f_X + np.exp(-self.delta * t[-1]) * self.f_func(X[-1], eta_t, rho_t)) * self.dt

            # Append the new state and time
            t.append(t_next)
            X.append(X_next)

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
                          epsilon = 1e-3, 
                          delta = 1e-6, 
                          M = 500, 
                          max_simulations = 5000, 
                          if_plot = False, 
                          if_stopping = False
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
        confidence_widths = []  # Store confidence interval widths for plotting

        while True:
            sim += 1
            # Use the existing objective_func to simulate the path and compute the reward
            _, _, cumulative_reward = self.objective_func(x_initial, x_grid, eta, rho)

            # Update the estimated value using the updating rule
            estimated_value += (cumulative_reward - estimated_value) / sim

            # Store the cumulative reward for standard deviation calculation
            cumulative_rewards.append(cumulative_reward)

            if if_stopping:
                # Criteria 1: Compute the standard deviation and confidence interval width
                if sim > 1:
                    std = np.std(cumulative_rewards, ddof=1)  # Sample standard deviation
                    confidence_interval_width = 1.96 * std / np.sqrt(sim)
                    confidence_widths.append(confidence_interval_width)

                    # Check the confidence interval stopping rule
                    if confidence_interval_width < epsilon:
                        print(f"Converged after {sim} simulations (confidence interval).")
                        break

                # Store the current estimate for plotting
                convergence.append(estimated_value)

                # Criteria 2: Check the stability-based stopping rule
                if sim > M:
                    recent_average = np.mean(convergence[-M:])  # Average of the last M estimates
                    if abs(estimated_value - recent_average) < delta:
                        print(f"Converged after {sim} simulations (stability-based stopping).")
                        break

                # Prevent infinite loops by limiting the number of simulations
                if sim >= max_simulations:
                    print(f"Reached maximum simulations ({max_simulations}) without full convergence.")
                    break
            else:
                if sim >= max_simulations:
                    print(f"Simulated with maximum ({max_simulations}).")
                    break

        if if_plot:
            # Create a single figure with two subplots
            axes = plt.subplots(2, 1, figsize=(10, 12))

            # Plot the convergence of the estimated value
            axes[0].plot(range(1, len(convergence) + 1), convergence, label="Estimated Value")
            axes[0].axhline(estimated_value, color='r', linestyle='--', label=f"Converged Value: {estimated_value:.4f}")
            axes[0].set_xlabel("Number of Simulations")
            axes[0].set_ylabel("Estimated Value")
            axes[0].set_title("Convergence of Monte Carlo Simulation")
            axes[0].legend()
            axes[0].grid()

            # Plot the confidence interval width
            axes[1].plot(range(2, len(confidence_widths) + 2), confidence_widths, label="Confidence Interval Width")
            axes[1].axhline(epsilon, color='r', linestyle='--', label=f"Tolerance: {epsilon}")
            axes[1].set_xlabel("Number of Simulations")
            axes[1].set_ylabel("Confidence Interval Width")
            axes[1].set_title("Confidence Interval Width During Monte Carlo Simulation")
            axes[1].legend()
            axes[1].grid()

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
        """

        # Obtaining Markov control by interpolating eta and rho
        eta = interp1d(x_grid, eta, kind = 'cubic', fill_value = 'extrapolate')
        rho = interp1d(x_grid, rho, kind = 'cubic', fill_value = 'extrapolate')

        def bellman_rhs(x, V):
            v, p = V
            eta_val = eta(x)
            rho_val = rho(x)
            b_val = self.b_func(x, eta_val, rho_val)
            sigma_val = self.sigma_func(x)
            f_val = self.f_func(x, eta_val, rho_val)
            dv_dx = p
            dp_dx = (self.delta * v - b_val * p - f_val) / (0.5 * sigma_val**2)
            return np.array([dv_dx, dp_dx])

        def boundary_conditions(Va, Vb):
            return [Va[0] - v0, Vb[0] - v1]

        x = x_grid
        v_guess = x
        p_guess = np.ones_like(x)
        V_guess = np.vstack([v_guess, p_guess])

        solution = solve_bvp(bellman_rhs, boundary_conditions, x, V_guess, max_nodes = 1e+4, tol = 1e-4)

        if solution.success:
            print("Solver converged successfully.")
        else:
            print("Solver failed to converge.")

        x_sol = solution.x
        v_sol = solution.y[0]

        if if_plot:
            plt.figure(figsize=(10, 8))
            plt.plot(x_sol, v_sol, label="Value Function")
            plt.xlabel("State (x)")
            plt.ylabel("Value (v)")
            plt.title("Solution to Bellman ODE using solve_bvp")
            plt.legend()
            plt.grid()
            plt.show()

        return x_sol, v_sol
    
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
        v0 = self.monte_carlo_value(self.x_min, self.x_grid, eta_guess, rho_guess, if_plot=False, if_stopping=True)
        v1 = self.monte_carlo_value(self.x_max, self.x_grid, eta_guess, rho_guess, if_plot=False, if_stopping=True)

        x, v_int = self.Bellman_solver(self.x_grid, v0, v1, eta_guess, rho_guess, if_plot=False)
        if if_compare:
            x_begin = x.copy()
            v_begin = v_int.copy()
        iter = 0
        error = []

        while True:
            x_previous = x.copy()
            D_x_v = np.zeros_like(v_int)
            D_x_v[1:] = (v_int[1:] - v_int[:-1]) / self.h
            D_x_v[0] = D_x_v[1]

            eta_new = np.clip(1 - (self.alpha + self.beta * x) * (1 - x) * D_x_v / (2 * x * self.a_2), 0, 1)
            rho_new = np.maximum(0, D_x_v / (2 * self.a_3))

            v0_new = self.monte_carlo_value(self.x_min, x, eta_new, rho_new, if_plot=False, if_stopping=True)
            v1_new = self.monte_carlo_value(self.x_max, x, eta_new, rho_new, if_plot=False, if_stopping=True)

            x, v_update = self.Bellman_solver(x, v0_new, v1_new, eta_new, rho_new, if_plot=False)

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

        return eta_new, rho_new, v_update


def main():
    """
    Main function to demonstrate the usage of the Problem_Solver class.
    """
    # Create an instance of the Problem_Solver
    solver = Problem_Solver()

    # Define initial state and controls
    eta = 0.5 * np.ones(solver.n)  # Initial guess of constant control η
    rho = np.zeros(solver.n)  # Initial guess of constant control ρ

    # Run the policy improvement algorithm
    solver.policy_improve(eta, rho, if_plot = True)


# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()