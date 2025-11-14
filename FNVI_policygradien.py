import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import Envir


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
    def __init__(self, state_dim=2, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid()
        )


    def forward(self, x):
        actions = self.net(x)
        return actions[:, 0], actions[:, 1]  # eta, rho


# =======================
# Fast NFVI with Policy Network
# =======================
class FastNFVI:
    def __init__(self, env, hidden_dim=16, lr_value=1e-3, lr_policy=1e-4):  # 降低策略学习率
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.value_net = ValueNet(hidden_dim=hidden_dim).to(self.device)
        self.policy_net = PolicyNet(hidden_dim=hidden_dim).to(self.device)
        
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=lr_value)
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=lr_policy)

        self.loss_history = []
        self.policy_loss_history = []
        self.residuals_history = []

        self.all_states = [[x, float(r)] for x in self.env.x_states for r in self.env.regimes]
        self.states_tensor = torch.FloatTensor(self.all_states).to(self.device)
        print(f"NFVI initialized. Total states: {len(self.all_states)}")

    def _find_nearest_state_index(self, x, regime):
        x_array = np.array([s[0] for s in self.all_states])
        regime_array = np.array([int(s[1]) for s in self.all_states])
        idx = np.argmin(np.abs(x_array - x) + 1e6 * (regime_array != regime))
        return int(idx)

    def _compute_Q_value(self, state, eta, rho):
        """计算 Q 值"""
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

    def train_iteration(self, lambda_output = 1e-3):
        """一次训练迭代 - 修正版本"""
        # 步骤1: 策略评估 - 用当前策略计算目标值
        with torch.no_grad():
            eta_all, rho_all = self.policy_net(self.states_tensor)
            targets = []
            
            for i, state in enumerate(self.all_states):
                Q_value = self._compute_Q_value(state, eta_all[i], rho_all[i])
                targets.append(Q_value.item())
            
            targets_tensor = torch.FloatTensor(targets).to(self.device)

        # 更新价值函数
        self.optimizer_value.zero_grad()
        predicted_values = self.value_net(self.states_tensor)
        value_loss = nn.MSELoss()(predicted_values, targets_tensor)
        value_loss.backward()
        self.optimizer_value.step()

        # 步骤2: 策略改进 - 使用价值函数梯度来改进策略
        self.optimizer_policy.zero_grad()
        
        # 重新计算当前策略的Q值（需要梯度）
        eta_all, rho_all = self.policy_net(self.states_tensor)
        policy_loss = 0
        
        for i, state in enumerate(self.all_states):

            # 计算 Q(s, a) 也要有梯度！！
            Q_current = self._compute_Q_value(state, eta_all[i], rho_all[i])

            policy_loss += Q_current

        policy_loss /= len(self.all_states)

        # L2-Regularization on policy outputs
        entropy = -(eta_all * torch.log(eta_all + 1e-8) + rho_all * torch.log(rho_all + 1e-8))
        policy_loss += lambda_output * torch.mean(entropy)
        
        # final_policy_loss.backward()
        policy_loss.backward()
        self.optimizer_policy.step()

        # 记录结果
        self.loss_history.append(value_loss.item())
        self.policy_loss_history.append(policy_loss.item())
        
        with torch.no_grad():
            residual = torch.mean(torch.abs(predicted_values - targets_tensor)).item()

        return value_loss.item(), policy_loss.item(), residual

    def get_policy(self):
        """获取最终策略"""
        with torch.no_grad():
            eta_all, rho_all = self.policy_net(self.states_tensor)
            policies = []
            for i in range(len(self.all_states)):
                policies.append((eta_all[i].item(), rho_all[i].item()))
            return policies

    def plot_results(self):
        """绘制结果"""
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

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 训练损失
        axes[0,0].plot(self.loss_history, label='Value Loss')
        axes[0,0].plot(self.policy_loss_history, label='Policy Loss')
        axes[0,0].set_title('Training Losses')
        axes[0,0].set_yscale('log')
        axes[0,0].legend()
        axes[0,0].grid(True)

        # Bellman残差
        axes[0,1].plot(self.residuals_history)
        axes[0,1].set_title('Bellman Residual')
        axes[0,1].set_yscale('log')
        axes[0,1].grid(True)

        # 价值函数
        axes[0,2].plot(x_values, values_0, label='Regime 0')
        axes[0,2].plot(x_values, values_1, label='Regime 1')
        axes[0,2].set_title('Value Function')
        axes[0,2].legend()
        axes[0,2].grid(True)

        # 策略 eta
        axes[1,0].plot(x_values, eta_0, label='eta 0')
        axes[1,0].plot(x_values, eta_1, label='eta 1')
        axes[1,0].set_title('Policy: Eta')
        axes[1,0].legend()
        axes[1,0].grid(True)

        # 策略 rho
        axes[1,1].plot(x_values, rho_0, label='rho 0')
        axes[1,1].plot(x_values, rho_1, label='rho 1')
        axes[1,1].set_title('Policy: Rho')
        axes[1,1].legend()
        axes[1,1].grid(True)

        # 组合图
        axes[1,2].plot(x_values, eta_0, '--', label='eta 0')
        axes[1,2].plot(x_values, eta_1, '--', label='eta 1')
        axes[1,2].plot(x_values, rho_0, ':', label='rho 0')
        axes[1,2].plot(x_values, rho_1, ':', label='rho 1')
        axes[1,2].set_title('Combined Policies')
        axes[1,2].legend()
        axes[1,2].grid(True)

        plt.tight_layout()
        plt.show()


# =======================
# Run NFVI
# =======================
if __name__ == "__main__":
    env = Envir.PandemicControlEnvironment()
    nfvi = FastNFVI(env, hidden_dim=16, lr_policy=5e-3)  # 使用更小的策略学习率

    n_iter = 10000
    for it in range(n_iter):
        value_loss, policy_loss, residual = nfvi.train_iteration()
        if it % 50 == 0:
            print(f"Iter {it}: ValueLoss={value_loss:.10f}, PolicyLoss={policy_loss:.8f}, "
                  f"Residual={residual:.4f}")

    nfvi.plot_results()