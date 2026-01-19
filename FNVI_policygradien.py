import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import Envir

plt.style.use('grayscale')


# =======================
# Small Neural Networks
# =======================
class ValueNet(nn.Module):
    def __init__(self, state_dim=2, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class PolicyNet(nn.Module):
    def __init__(self, state_dim = 2, hidden_dim = 32, M = 10.0):
        super().__init__()
        self.M = M
        
        # Simple 2-layer network
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.activation = nn.LeakyReLU(0.01)
        self.fc2 = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        outputs = self.fc2(x)
        
        output1 = torch.sigmoid(outputs[:, 0])
        
        # For ρ: direct approach
        output2 = torch.exp(outputs[:, 1])  # (0, ∞)
        output2 = self.M * output2 / (1 + output2)  # Soft clamp to (0, M)
        
        return output1, output2
    
# =======================
# Deterministic AC
# =======================
class FastNFVI:
    def __init__(self, env, hidden_dim=16, lr_value=1e-3, lr_policy=1e-4):  # lr_policy << lr_value
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.value_net = ValueNet(hidden_dim=hidden_dim).to(self.device)
        self.policy_net = PolicyNet(hidden_dim=hidden_dim).to(self.device)
        
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=lr_value)
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=lr_policy)

        self.loss_history = []

        self.all_states = [[x, float(r)] for x in self.env.x_states for r in self.env.regimes]
        self.states_tensor = torch.FloatTensor(self.all_states).to(self.device)
        print(f"NFVI initialized. Total states: {len(self.all_states)}")

    def _find_nearest_state_index(self, x, regime):
        x_array = np.array([s[0] for s in self.all_states])
        regime_array = np.array([int(s[1]) for s in self.all_states])
        idx = np.argmin(np.abs(x_array - x) + 1e6 * (regime_array != regime))
        return int(idx)

    def _compute_Q_value(self, state, eta, rho):
        """Compute the Bellman RHS"""
        x, regime = state[0], int(state[1])
        
        trans_probs = self.env.trans_prob(x, regime, eta, rho)
        Qh = trans_probs["Qh"]
        Delta_t = self.env.h ** 2 / Qh
        immediate_cost = self.env.f(x, eta, rho) * Delta_t

        expected_next_value = 0.0
        for next_x, next_regime, prob in [
            (max(self.env.h, x - self.env.h), regime, trans_probs["p_down"]),
            (x, regime, trans_probs["p_stay"]),
            (min(1.0 - self.env.h, x + self.env.h), regime, trans_probs["p_up"]),
            (x, 1 - regime, trans_probs["p_switch"])
        ]:
            idx = self._find_nearest_state_index(next_x, next_regime)
            next_val = self.value_net(self.states_tensor[idx:idx+1])
            expected_next_value += prob * next_val

        return immediate_cost + torch.exp(-self.env.delta * Delta_t) * expected_next_value

    def train_iteration(self):
        # Step1: Critic step
        with torch.no_grad():
            eta_all, rho_all = self.policy_net(self.states_tensor)
            targets = []
            
            for i, state in enumerate(self.all_states):
                Q_value = self._compute_Q_value(state, eta_all[i], rho_all[i])
                targets.append(Q_value.item())
            
            targets_tensor = torch.FloatTensor(targets).to(self.device)

        # Optimize Value Network
        self.optimizer_value.zero_grad()
        predicted_values = self.value_net(self.states_tensor)
        value_loss = nn.MSELoss()(predicted_values, targets_tensor)
        value_loss.backward()
        self.optimizer_value.step()

        # Step 2: Actor step
        self.optimizer_policy.zero_grad()
        
        # Re-compute policy outputs with gradients
        eta_all, rho_all = self.policy_net(self.states_tensor)
        policy_loss = 0
        
        for i, state in enumerate(self.all_states):

            # Bellman RHS with gradient tracking
            Q_current = self._compute_Q_value(state, eta_all[i], rho_all[i])
            policy_loss += Q_current

        policy_loss /= len(self.all_states)
                
        policy_loss.backward()
        self.optimizer_policy.step()

        # record losses
        self.loss_history.append(value_loss.item())
        
        return value_loss.item()

    def get_policy(self):
        """Extract the learned policy"""
        with torch.no_grad():
            eta_all, rho_all = self.policy_net(self.states_tensor)
            policies = []
            for i in range(len(self.all_states)):
                policies.append((eta_all[i].item(), rho_all[i].item()))
            return policies

    def plot_results(self):
        """Plot function"""
        policies = self.get_policy()
        
        x_values = self.env.x_states
        values_0, values_1 = [], []
        eta_0, eta_1, rho_0, rho_1 = [], [], [], []

        for i, (x, regime) in enumerate(self.all_states):
            with torch.no_grad():
                val = self.value_net(torch.FloatTensor([x, regime]).to(self.device)).item()
            if int(regime) == 0:
                values_0.append(val)
                eta_0.append(policies[i][0])
                rho_0.append(policies[i][1])
            else:
                values_1.append(val)
                eta_1.append(policies[i][0])
                rho_1.append(policies[i][1])

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Bellman Residual
        axes[0,0].plot(self.loss_history)
        axes[0,0].set_title('Value Residual')
        axes[0,0].set_yscale('log')

        # Value Function
        axes[0,1].plot(x_values, values_0, label='Regime 0')
        axes[0,1].plot(x_values, values_1, label='Regime 1')
        axes[0,1].set_title('Value Function')
        axes[0,1].legend()

        # Policy: eta
        axes[1,0].plot(x_values, eta_0, label='eta 0')
        axes[1,0].plot(x_values, eta_1, label='eta 1')
        axes[1,0].set_title('Policy: Eta')
        axes[1,0].legend()

        # Policy: rho
        axes[1,1].plot(x_values, rho_0, label='rho 0')
        axes[1,1].plot(x_values, rho_1, label='rho 1')
        axes[1,1].set_title('Policy: Rho')
        axes[1,1].legend()

        plt.tight_layout()
        plt.show()


# =======================
# Run Deterministic Actor-Critic Algorithm
# =======================
if __name__ == "__main__":
    env = Envir.PandemicControlEnvironment()
    ## lr_policy must be much smaller than lr_value!
    nfvi = FastNFVI(env, hidden_dim=16, lr_value = 5e-3, lr_policy=5e-4)  
    
    n_iter = 10000
    for it in range(n_iter):
        value_loss = nfvi.train_iteration()
        if it % 50 == 0:
            print(f"Iter {it}: ValueLoss={value_loss:.10f}")

    nfvi.plot_results()