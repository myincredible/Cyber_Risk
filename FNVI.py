import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import Envir 

class FastValueNetwork(nn.Module):
    """快速价值网络"""
    def __init__(self, state_dim=2, hidden_dim=16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.network(x).squeeze(-1)

class FastNFVI:
    """快速神经拟合值迭代"""
    def __init__(self, env, hidden_dim=16, lr=5e-3):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.value_net = FastValueNetwork(hidden_dim=hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        self.loss_history = []
        self.best_actions_history = []  # 保存每轮的最优动作
        
        # 预计算所有状态
        self.all_states = self._precompute_states()
        self.states_tensor = torch.FloatTensor(self.all_states).to(self.device)
        
        print(f"快速NFVI初始化完成")
        print(f"状态数量: {len(self.all_states)}")
    
    def _precompute_states(self):
        """预计算所有状态"""
        states = []
        for x in self.env.x_states:
            for regime in self.env.regimes:
                states.append([x, float(regime)])
        return states
    
    def compute_bellman_targets_batch(self):
        """批量计算贝尔曼目标 - 返回目标和最优动作"""
        targets = []
        best_actions = []
        
        for i, state in enumerate(self.all_states):
            x, regime = state[0], int(state[1])
            min_target = float('inf')
            best_action = 0
            
            for action_idx in range(self.env.n_actions): 
                eta, rho = self.env.actions[action_idx]
                
                # 计算转移概率
                trans_probs = self.env.trans_prob(x, regime, eta, rho)
                Qh = trans_probs["Qh"]
                Delta_t = self.env.h**2 / Qh
                immediate_cost = self.env.f(x, eta, rho) * Delta_t
                
                # 计算期望下一状态价值
                expected_next_value = 0.0
                
                # 主要转移方向
                transitions = [
                    (max(self.env.h, x - self.env.h), regime, trans_probs["p_down"]),
                    (x, regime, trans_probs["p_stay"]),
                    (min(1.0 - self.env.h, x + self.env.h), regime, trans_probs["p_up"]),
                    (x, 1 - regime, trans_probs["p_switch"])
                ]
                
                for next_x, next_regime, prob in transitions:
                    # if prob > 1e-8:
                    state_idx = self._find_nearest_state_index(next_x, next_regime)
                    with torch.no_grad():
                        next_value = self.value_net(self.states_tensor[state_idx:state_idx + 1]).item()
                    expected_next_value += prob * next_value
                
                target = immediate_cost + np.exp(-self.env.delta * Delta_t) * expected_next_value
                
                if target < min_target:
                    min_target = target
                    best_action = action_idx
            
            targets.append(min_target)
            best_actions.append(best_action)
        
        return torch.FloatTensor(targets).to(self.device), best_actions
    
    def _find_nearest_state_index(self, x, regime):
        """找到最近状态的索引"""
        for i, state in enumerate(self.all_states):
            if abs(state[0] - x) < 1e-6 and int(state[1]) == regime:
                return i
        return 0
    
    def train_iteration(self):
        """执行一次值迭代 - 返回最优动作"""
        targets, best_actions = self.compute_bellman_targets_batch()
        
        self.optimizer.zero_grad()
        predicted_values = self.value_net(self.states_tensor)
        loss = nn.MSELoss()(predicted_values, targets)
        loss.backward()
        self.optimizer.step()
        
        self.loss_history.append(loss.item())
        self.best_actions_history.append(best_actions)  # 保存这轮的最优动作
        
        return loss.item(), np.mean(predicted_values.detach().cpu().numpy()), best_actions
    
    def get_final_policy(self):
        """直接返回最后一轮训练的最优策略"""
        if not self.best_actions_history:
            raise ValueError("还没有进行训练，请先调用 train_iteration()")
        
        # 返回最后一轮的最优动作
        final_best_actions = self.best_actions_history[-1]
        
        # 创建策略字典：状态 -> 最优动作
        policy_dict = {}
        for i, state in enumerate(self.all_states):
            x, regime = state[0], int(state[1])
            policy_dict[(x, regime)] = final_best_actions[i]
        
        return policy_dict
    
    def get_policy(self, state):
        """根据最终策略获取动作 - 直接从保存的策略中获取"""
        if not self.best_actions_history:
            # 如果没有训练过，回退到计算方式
            return self._compute_policy_fallback(state)
        
        # 直接从最后一轮训练的策略中获取
        x, regime = state
        final_best_actions = self.best_actions_history[-1]
        
        # 找到对应状态的索引
        state_idx = self._find_nearest_state_index(x, regime)
        return final_best_actions[state_idx]
    
    def get_policy_parameters(self, state):
        """根据状态获取策略参数 (eta, rho)"""
        action_idx = self.get_policy(state)
        return self.env.actions[action_idx]
    
    # ==================== 画图函数 ====================
    def plot_final_results(self):
        """绘制最终的价值函数和策略参数"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. 损失曲线
        axes[0, 0].plot(self.loss_history, 'b-', linewidth=2)
        axes[0, 0].set_title('Training Loss History')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 价值函数 - Regime 0
        x_values = self.env.x_states
        values_regime0 = []
        for x in x_values:
            state_tensor = torch.FloatTensor([x, 0.0]).to(self.device)
            with torch.no_grad():
                value = self.value_net(state_tensor).item()
            values_regime0.append(value)
        
        axes[0, 1].plot(x_values, values_regime0, 'g-', linewidth=2)
        axes[0, 1].set_title('Value Function - Regime 0')
        axes[0, 1].set_xlabel('Infection Rate (x)')
        axes[0, 1].set_ylabel('State Value V(s)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 价值函数 - Regime 1
        values_regime1 = []
        for x in x_values:
            state_tensor = torch.FloatTensor([x, 1.0]).to(self.device)
            with torch.no_grad():
                value = self.value_net(state_tensor).item()
            values_regime1.append(value)
        
        axes[0, 2].plot(x_values, values_regime1, 'r-', linewidth=2)
        axes[0, 2].set_title('Value Function - Regime 1')
        axes[0, 2].set_xlabel('Infection Rate (x)')
        axes[0, 2].set_ylabel('State Value V(s)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Eta 控制参数 - 两个Regime
        eta_values_0 = []
        eta_values_1 = []
        
        if self.best_actions_history:
            final_best_actions = self.best_actions_history[-1]
            for i, state in enumerate(self.all_states):
                x, regime = state[0], int(state[1])
                action_idx = final_best_actions[i]
                eta, rho = self.env.actions[action_idx]
                if regime == 0:
                    eta_values_0.append(eta)
                else:
                    eta_values_1.append(eta)
        
        # 确保列表长度一致
        eta_values_0 = eta_values_0[:len(x_values)]
        eta_values_1 = eta_values_1[:len(x_values)]
        
        axes[1, 0].plot(x_values, eta_values_0, 'b-', linewidth=2, label='Regime 0', alpha=0.7)
        axes[1, 0].plot(x_values, eta_values_1, 'r-', linewidth=2, label='Regime 1', alpha=0.7)
        axes[1, 0].set_title('Control Parameter η (Transmission Control)')
        axes[1, 0].set_xlabel('Infection Rate (x)')
        axes[1, 0].set_ylabel('η Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Rho 控制参数 - 两个Regime
        rho_values_0 = []
        rho_values_1 = []
        
        if self.best_actions_history:
            final_best_actions = self.best_actions_history[-1]
            for i, state in enumerate(self.all_states):
                x, regime = state[0], int(state[1])
                action_idx = final_best_actions[i]
                eta, rho = self.env.actions[action_idx]
                if regime == 0:
                    rho_values_0.append(rho)
                else:
                    rho_values_1.append(rho)
        
        # 确保列表长度一致
        rho_values_0 = rho_values_0[:len(x_values)]
        rho_values_1 = rho_values_1[:len(x_values)]
        
        axes[1, 1].plot(x_values, rho_values_0, 'b-', linewidth=2, label='Regime 0', alpha=0.7)
        axes[1, 1].plot(x_values, rho_values_1, 'r-', linewidth=2, label='Regime 1', alpha=0.7)
        axes[1, 1].set_title('Control Parameter ρ (Treatment Intensity)')
        axes[1, 1].set_xlabel('Infection Rate (x)')
        axes[1, 1].set_ylabel('ρ Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 策略参数散点图
        all_eta_0 = []
        all_rho_0 = []
        all_eta_1 = []
        all_rho_1 = []
        
        if self.best_actions_history:
            final_best_actions = self.best_actions_history[-1]
            for i, state in enumerate(self.all_states):
                x, regime = state[0], int(state[1])
                action_idx = final_best_actions[i]
                eta, rho = self.env.actions[action_idx]
                if regime == 0:
                    all_eta_0.append(eta)
                    all_rho_0.append(rho)
                else:
                    all_eta_1.append(eta)
                    all_rho_1.append(rho)
        
        scatter0 = axes[1, 2].scatter(all_eta_0, all_rho_0, c=x_values[:len(all_eta_0)], 
                                     cmap='viridis', alpha=0.7, s=50, label='Regime 0')
        scatter1 = axes[1, 2].scatter(all_eta_1, all_rho_1, c=x_values[:len(all_eta_1)], 
                                     cmap='plasma', alpha=0.7, s=50, label='Regime 1', marker='s')
        axes[1, 2].set_xlabel('η (Transmission Control)')
        axes[1, 2].set_ylabel('ρ (Treatment Intensity)')
        axes[1, 2].set_title('Policy Parameters Scatter Plot')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # 添加颜色条
        plt.colorbar(scatter0, ax=axes[1, 2], label='Infection Rate (x)')
        
        # 打印策略统计信息
        print(f"\n策略参数统计:")
        if all_eta_0:
            print(f"Regime 0 - η范围: {min(all_eta_0):.2f} 到 {max(all_eta_0):.2f}")
            print(f"Regime 0 - ρ范围: {min(all_rho_0):.2f} 到 {max(all_rho_0):.2f}")
        if all_eta_1:
            print(f"Regime 1 - η范围: {min(all_eta_1):.2f} 到 {max(all_eta_1):.2f}")
            print(f"Regime 1 - ρ范围: {min(all_rho_1):.2f} 到 {max(all_rho_1):.2f}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_policy_parameters_comparison(self):
        """专门绘制策略参数对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        x_values = self.env.x_states
        
        if not self.best_actions_history:
            print("没有训练历史，无法绘制策略参数")
            return
        
        final_best_actions = self.best_actions_history[-1]
        
        # 提取eta和rho参数
        eta_0, rho_0 = [], []
        eta_1, rho_1 = [], []
        
        for i, state in enumerate(self.all_states):
            x, regime = state[0], int(state[1])
            action_idx = final_best_actions[i]
            eta, rho = self.env.actions[action_idx]
            
            if regime == 0:
                eta_0.append(eta)
                rho_0.append(rho)
            else:
                eta_1.append(eta)
                rho_1.append(rho)
        
        # 确保长度一致
        x_0 = x_values[:len(eta_0)]
        x_1 = x_values[:len(eta_1)]
        
        # Eta 参数对比
        axes[0].plot(x_0, eta_0, 'b-', linewidth=3, label='Regime 0', alpha=0.8)
        axes[0].plot(x_1, eta_1, 'r-', linewidth=3, label='Regime 1', alpha=0.8)
        axes[0].set_xlabel('Infection Rate (x)')
        axes[0].set_ylabel('η Value')
        axes[0].set_title('Transmission Control Parameter η')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Rho 参数对比
        axes[1].plot(x_0, rho_0, 'b-', linewidth=3, label='Regime 0', alpha=0.8)
        axes[1].plot(x_1, rho_1, 'r-', linewidth=3, label='Regime 1', alpha=0.8)
        axes[1].set_xlabel('Infection Rate (x)')
        axes[1].set_ylabel('ρ Value')
        axes[1].set_title('Treatment Intensity Parameter ρ')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# 立即测试
if __name__ == "__main__":
    env = Envir.PandemicControlEnvironment()
    
    print("=== 快速NFVI训练 ===")
    print(f"状态空间: {len(env.x_states)}感染率 × {len(env.regimes)}regime")
    print(f"动作空间: {env.n_actions}动作")
    print(f"动作参数范围: η∈[0,1], ρ∈[0,1]")
    
    nfvi = FastNFVI(env, hidden_dim=32)
    
    # 训练几个迭代看效果
    for iteration in range(10000):
        loss, avg_value, best_actions = nfvi.train_iteration()
        
        if iteration % 10 == 0:
            print(f"{iteration}, Loss: {loss:.10f} | Averaged NN Value: {avg_value:.4f}")
    
    # 训练完成后画图 - 显示策略参数
    print("\n训练完成，开始画图...")
    nfvi.plot_final_results()
    nfvi.plot_policy_parameters_comparison()
    
    # 获取最终策略并显示示例
    final_policy = nfvi.get_final_policy()
    print(f"\n最终策略包含 {len(final_policy)} 个状态-动作对")
    
    # 显示几个状态的具体策略参数
    print("\n策略参数示例:")
    test_states = [(0.1, 0), (0.5, 0), (0.9, 0), (0.1, 1), (0.5, 1), (0.9, 1)]
    for state in test_states:
        action_idx = nfvi.get_policy(state)
        eta, rho = nfvi.get_policy_parameters(state)
        print(f"  状态 {state} -> 动作{action_idx}: η={eta:.2f}, ρ={rho:.2f}")