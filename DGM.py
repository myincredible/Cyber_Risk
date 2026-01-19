import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class BellmanSolverDGM:

    def __init__(self, x_grid_np, eta_np, rho_np, params):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Problem parameters
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        self.gamma = params["gamma"]
        self.tilde_sigma = params["tilde_sigma"]
        self.a_0 = params["a_0"]
        self.a_1 = params["a_1"]
        self.a_2 = params["a_2"]
        self.a_3 = params["a_3"]
        self.a_4 = params["a_4"]
        self.delta = params["delta"]
        self.v0_true = params.get("v0_true", 1.0)  # Default value if not provided
        self.v1_true = params.get("v1_true", 0.5)
        self.x0 = params.get("x0", 0.01) if "x0" in params else 0.01  # Default x0
        self.x1 = params.get("x1", 0.99) if "x1" in params else 0.99  # Default x1
        self.n = params.get("n", 100)  # Default number of collocation points

        # Interpolation functions for eta and rho
        self.eta_interp_np = interp1d(x_grid_np, eta_np, kind='cubic', fill_value='extrapolate')
        self.rho_interp_np = interp1d(x_grid_np, rho_np, kind='cubic', fill_value='extrapolate')

        # Neural network model
        self.net = self.build_net().to(self.device)

    def build_net(self, hidden_dim = 32, num_layers = 1):

        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]

        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]

        layers.append(nn.Linear(hidden_dim, 1))  # output linear

        return nn.Sequential(*layers)

    def b_func(self, x, eta, rho):
        return eta * self.alpha * (1 - x) + x * (eta ** 2 * self.beta * (1 - x) - (self.gamma + rho))

    def sigma_func(self, x):
        return self.tilde_sigma * x * (1 - x)

    def f_func(self, x, eta, rho):
        return self.a_0 + self.a_1 * x + (self.a_2 * x + self.a_4) * (1 - eta) ** 2 + self.a_3 * x * rho ** 2

    def eta_torch(self, x_torch):
        x_np = x_torch.detach().cpu().numpy().flatten()
        val_np = self.eta_interp_np(x_np)
        return torch.tensor(val_np, dtype=torch.float32, device=self.device).reshape(-1, 1)

    def rho_torch(self, x_torch):
        x_np = x_torch.detach().cpu().numpy().flatten()
        val_np = self.rho_interp_np(x_np)
        return torch.tensor(val_np, dtype=torch.float32, device=self.device).reshape(-1, 1)

    def train(
        self,
        n_epochs=5000,
        n_collocation=1000,
        print_every=200,
        tol=1e-3,
        learning_rate=1e-4,
        penalty=0.1
    ):
        epoch = 0

        self.optimizer = optim.AdamW(
            self.net.parameters(),
            lr=learning_rate,
            # weight_decay=1e-2  # typical weight decay to regularize
        )

        while True:
            self.optimizer.zero_grad()

            # Sample collocation points
            x = torch.empty(n_collocation, 1, device=self.device).uniform_(self.x0, self.x1).requires_grad_(True)
            eta_vals = self.eta_torch(x)
            rho_vals = self.rho_torch(x)

            V = self.net(x)
            dV_dx = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
            d2V_dx2 = torch.autograd.grad(dV_dx.sum(), x, create_graph=True)[0]

            b_val = self.b_func(x, eta_vals, rho_vals)
            sigma_val = self.sigma_func(x)
            weight = torch.clamp(1.0 / (sigma_val**2), max=1e3)
            f_val = self.f_func(x, eta_vals, rho_vals)

            # Regularization term
            # mean_sigma = 0.5 * sigma_val.pow(2).mean()
            # mean_rhs = (self.delta * V - b_val * dV_dx - f_val).abs().mean()
            # scale_factor = mean_rhs / mean_sigma

            # Compute the residual of the Bellman equation
            residual =  d2V_dx2 - 2 / sigma_val ** 2 * (self.delta * V - b_val * dV_dx - f_val)
            physics_loss = residual.pow(2).mean()

            # Boundary conditions loss
            V_x0 = self.net(torch.tensor([[self.x0]], device=self.device))
            V_x1 = self.net(torch.tensor([[self.x1]], device=self.device))
            bc_loss = penalty * ((V_x0 - self.v0_true).pow(2) + (V_x1 - self.v1_true).pow(2))

            loss = physics_loss + bc_loss

            loss.backward()
            self.optimizer.step()

            if epoch % print_every == 0 or loss.item() < tol or epoch == n_epochs - 1:
                print(f"Epoch {epoch:5d} | Loss: {loss.item():.4e} | Physics: {physics_loss.item():.4e} | BC: {bc_loss.item():.4e}")

            if loss.item() < tol or epoch >= n_epochs - 1:
                print(f"Training stopped at epoch {epoch} with loss {loss.item():.4e}")
                break

            epoch += 1

    def predict(self):
        self.net.eval()

        x = torch.linspace(self.x0, self.x1, self.n, device = self.device).reshape(-1, 1)
        x.requires_grad_(True)

        # Compute the value function V and its derivative
        V = self.net(x)
        dV_dx = torch.autograd.grad(
        V.sum(), x, create_graph=False, retain_graph=True
        )[0]

        return x.detach().cpu().numpy(), V.detach().cpu().numpy(), dV_dx.detach().cpu().numpy()


# ---------------------------
# USAGE EXAMPLE
# ---------------------------

# Suppose you have your x_grid, eta, rho as numpy arrays from your PIA step:
# (Replace these example arrays with your actual data)
# x_grid_np = np.linspace(0.01, 0.99, 100)
# eta_np = np.ones_like(x_grid_np)  # Initial control function eta(x)
# rho_np = np.ones_like(x_grid_np)  # Initial control function rho(x)

# params = {
#     "alpha": 0.5,
#     "beta": 0.5,
#     "gamma": 0.15,
#     "tilde_sigma": 0.3,
#     "a_0": 0.5,
#     "a_1": 2.5,
#     "a_2": 0.5,
#     "a_3": 1.5,
#     "a_4": 2,
#     "delta": 0.05, 
#     "v0_true": 37.9707,  # Known boundary condition at x=0
#     "v1_true": 40.3641   # Known boundary condition at x=1
# }


# solver = BellmanSolverDGM(x_grid_np, eta_np, rho_np, params)
# solver.train(n_epochs = 10000, n_collocation=200)

# x_vals, V_vals, v_grad = solver.predict()

# plt.plot(x_vals, V_vals, label="DGM Value Function")
# plt.xlabel("x")
# plt.ylabel("V(x)")
# plt.title("Bellman Equation Solution via Deep Galerkin Method")
# plt.legend()
# plt.show()