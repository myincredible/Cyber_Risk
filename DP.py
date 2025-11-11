import numpy as np
import matplotlib.pyplot as plt
import random

# ---- Parameters ----
h = 0.05
x_states = np.arange(h, 1.0, h)
n_x = len(x_states)
regimes = [0, 1]

# Model parameters
alpha = [0.2, 0.8]
beta = [0.2, 0.8]
gamma = [0.1, 0.2]
sigma_tilde = [0.15, 0.35]
q = np.array([[-1, 1], [1, -1]])
delta = 0.1

a0, aI, aS_m, aI_m, a_r = 0.0, 1.5, 0.5, 1.5, 1.0

# Control discretization
eta_vals = np.linspace(0.0, 1.0, 5)  # smaller grid for speed
rho_vals = np.linspace(0.0, 1.0, 5)
actions = [(e, r) for e in eta_vals for r in rho_vals]

# ---- Model functions ----
def b(x, l, eta, rho):
    return eta*alpha[l]*(1-x) + x*(eta**2*beta[l]*(1-x) - (gamma[l]+rho))

def sigma(x, l):
    return sigma_tilde[l]*x*(1-x)

def f(x, eta, rho):
    return a0 + aI*x + aS_m*(1-eta)**2 + (aI_m-aS_m)*x*(1-eta)**2 + a_r*x*(rho**2)

def compute_Qh(x, l, eta, rho):
    drift = b(x, l, eta, rho)
    vol = sigma(x, l)
    q_ii = q[l, l]
    Qh = vol**2 + abs(drift)*h - h**2*q_ii + h
    return max(Qh, 1e-12), drift, vol

def trans_prob(x, l, eta, rho):
    Qh, drift, vol = compute_Qh(x, l, eta, rho)
    p_up = (vol**2/2 + max(drift,0)*h) / Qh
    p_down = (vol**2/2 - min(drift,0)*h) / Qh
    p_stay = h / Qh
    p_switch = (h**2 * q[l, 1-l]) / Qh
    return {"p_up": p_up, "p_down": p_down, "p_stay": p_stay, "p_switch": p_switch, "Qh": Qh}

def sample_next_state(x, l, eta, rho):
    p = trans_prob(x, l, eta, rho)
    r = random.random()
    if r < p["p_down"]:
        x_next, l_next = max(h, x-h), l
    elif r < p["p_down"] + p["p_stay"]:
        x_next, l_next = x, l
    elif r < p["p_down"] + p["p_stay"] + p["p_up"]:
        x_next, l_next = min(1-h, x+h), l
    elif r < p["p_down"] + p["p_stay"] + p["p_up"] + p["p_switch"]:
        x_next, l_next = x, 1-l
    else:
        x_next, l_next = x, l
    return x_next, l_next, p["Qh"]

# ---- DP parameters ----
tol = 1e-6
max_iter = 10000

# Initialize value function
V = np.zeros((n_x, len(regimes)))
V_new = np.ones_like(V)

# ---- DP iteration with explicit accumulated time ----
for it in range(max_iter):
    for xi, x in enumerate(x_states):
        for l in regimes:
            vals = []
            for a_idx, (eta, rho) in enumerate(actions):
                p = trans_prob(x, l, eta, rho)
                
                # map next states to indices
                x_up = min(1 - h, x + h)
                x_down = max(h, x - h)
                l_other = 1 - l

                xi_up = max(0, min(n_x - 1, int(round((x_up - h)/h))))
                xi_down = max(0, min(n_x - 1, int(round((x_down - h)/h))))
                xi_stay = max(0, min(n_x - 1, int(round((x - h)/h))))

                # compute expected value
                expected_V = (
                    p["p_up"] * V[xi_up, l] +
                    p["p_down"] * V[xi_down, l] +
                    p["p_stay"] * V[xi_stay, l] +
                    p["p_switch"] * V[xi_stay, l_other]
                )

                # --- accumulated time discount ---
                Delta_t = h**2 / p["Qh"]
                total_cost = f(x, eta, rho) * Delta_t
                total = np.exp(-delta * Delta_t) * expected_V + total_cost
                vals.append(total)
                
            V_new[xi, l] = min(vals)

    diff = np.max(np.abs(V_new - V))
    if it % 50 == 0 or it == max_iter - 1:
        print(f"DP iteration {it+1}, max diff={diff:.2e}")
    V[:] = V_new
    if diff < tol:
        print(f"DP converged at iteration {it+1}, max diff={diff:.2e}")
        break
else:
    print("DP did not fully converge")

# ---- Extract optimal policy ----
policy = np.zeros((n_x, len(regimes)), dtype=int)
for xi, x in enumerate(x_states):
    for l in regimes:
        vals = []
        for a_idx, (eta, rho) in enumerate(actions):
            p = trans_prob(x, l, eta, rho)
            x_up = min(1 - h, x + h)
            x_down = max(h, x - h)
            l_other = 1 - l
            xi_up = max(0, min(n_x - 1, int(round((x_up - h)/h))))
            xi_down = max(0, min(n_x - 1, int(round((x_down - h)/h))))
            xi_stay = max(0, min(n_x - 1, int(round((x - h)/h))))

            expected_V = (
                p["p_up"] * V[xi_up, l] +
                p["p_down"] * V[xi_down, l] +
                p["p_stay"] * V[xi_stay, l] +
                p["p_switch"] * V[xi_stay, l_other]
            )
            Delta_t = h**2 / p["Qh"]
            total = f(x, eta, rho) * Delta_t + np.exp(-delta * Delta_t) * expected_V
            vals.append(total)

        policy[xi, l] = np.argmin(vals)

# ---- Print optimal policy ----
print("\nDP optimal policy (η, ρ):")
for xi, x in enumerate(x_states):
    for l in regimes:
        eta, rho = actions[policy[xi, l]]
        print(f"x={x:.2f}, regime={l} -> (η,ρ)=({eta:.2f}, {rho:.2f})")

# ---- Plot the value function ----
plt.figure()
plt.plot(x_states, V[:, 0], label="Regime 0")
plt.plot(x_states, V[:, 1], label="Regime 1")
plt.xlabel("x")
plt.ylabel("Value function (cost)")
plt.title("Dynamic Programming Value Function with t_n^h discount")
plt.grid(True)
plt.show()