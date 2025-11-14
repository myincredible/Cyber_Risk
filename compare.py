import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import DP  # 导入DP文件
import Envir  # 导入环境文件

# ============================================================
# 优化后的有监督学习网络 - 使用Sigmoid
# ============================================================
class OptimizedSupervisedNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=32):  # 输出维度改为1，预测最优值
        super(OptimizedSupervisedNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# ============================================================
# 有监督学习训练器 - 直接学习DP的最优值函数
# ============================================================
class ValueFunctionLearner:
    def __init__(self, state_dim, lr=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.state_dim = state_dim
        
        # 网络 - 直接学习最优值函数 V*(s)
        self.value_net = OptimizedSupervisedNetwork(state_dim, output_dim=1).to(self.device)
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # 训练数据存储
        self.training_data = []
        
    def add_training_data(self, states, targets):
        """添加训练数据：状态 -> DP最优值"""
        for state, target in zip(states, targets):
            self.training_data.append((state, target))
            
    def train_epoch(self, batch_size=128, epochs=100):
        """训练一个epoch"""
        if len(self.training_data) < batch_size:
            return 0.0
            
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            # 随机打乱数据
            random.shuffle(self.training_data)
            
            for i in range(0, len(self.training_data), batch_size):
                batch = self.training_data[i:i+batch_size]
                if len(batch) < batch_size:
                    continue
                    
                states, targets = zip(*batch)
                
                states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
                targets_tensor = torch.FloatTensor(np.array(targets)).unsqueeze(1).to(self.device)  # 保持维度一致
                
                # 前向传播
                predictions = self.value_net(states_tensor)
                
                # 计算损失
                loss = self.criterion(predictions, targets_tensor)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def predict_value(self, state):
        """预测状态的最优值"""
        self.value_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            value = self.value_net(state_tensor).cpu().numpy()[0, 0]
        return value

# ============================================================
# 从DP解生成训练数据 - 直接使用DP的最优值
# ============================================================
def generate_value_training_data(env, num_samples=10000):
    """从DP最优值函数生成训练数据"""
    
    states = []
    values = []
    
    print("Generating training data from DP optimal value function...")
    
    for sample in range(num_samples):
        # 随机采样状态
        x = np.random.choice(env.x_states)
        l = np.random.choice(env.regimes)
        state = [x, l]
        
        # 获取DP最优值
        xi = env.get_discrete_state_index(x)
        dp_value = DP.V[xi, l]
        
        states.append(state)
        values.append(dp_value)
        
        if (sample + 1) % 1000 == 0:
            print(f"Generated {sample + 1}/{num_samples} samples")
    
    return np.array(states), np.array(values)

# ============================================================
# 基于学习值函数的策略
# ============================================================
class ValueBasedPolicy:
    def __init__(self, value_learner, env):
        self.value_learner = value_learner
        self.env = env
        self.h = env.h
        self.delta = env.delta
        
    def get_optimal_action(self, state):
        """基于学习值函数选择最优动作"""
        x, l = state
        best_action = 0
        best_value = float('inf')
        
        # 对每个动作，计算期望值
        for a_idx, (eta, rho) in enumerate(self.env.actions):
            p = self.env.trans_prob(x, l, eta, rho)
            
            # 计算下一个状态的期望值
            expected_value = 0.0
            
            # 上移状态
            x_up = min(1 - self.h, x + self.h)
            state_up = [x_up, l]
            value_up = self.value_learner.predict_value(state_up)
            expected_value += p["p_up"] * value_up
            
            # 下移状态
            x_down = max(self.h, x - self.h)
            state_down = [x_down, l]
            value_down = self.value_learner.predict_value(state_down)
            expected_value += p["p_down"] * value_down
            
            # 停留状态
            state_stay = [x, l]
            value_stay = self.value_learner.predict_value(state_stay)
            expected_value += p["p_stay"] * value_stay
            
            # 切换制度状态
            state_switch = [x, 1 - l]
            value_switch = self.value_learner.predict_value(state_switch)
            expected_value += p["p_switch"] * value_switch
            
            # 计算总成本
            Delta_t = self.h**2 / p["Qh"]
            immediate_cost = self.env.f(x, eta, rho) * Delta_t
            total_value = immediate_cost + np.exp(-self.delta * Delta_t) * expected_value
            
            if total_value < best_value:
                best_value = total_value
                best_action = a_idx
                
        return best_action

# ============================================================
# 评估函数
# ============================================================
def evaluate_value_based_policy(policy, env, test_episodes=100, max_steps=500):
    """评估基于值函数的策略性能"""
    costs = []
    
    for episode in range(test_episodes):
        state = env.reset()
        episode_cost = 0.0
        current_discount = 1.0
        
        for step in range(max_steps):
            # 使用基于值函数的策略选择动作
            action = policy.get_optimal_action(state)
            next_state, reward, discount, info = env.step(action)
            
            episode_cost += current_discount * reward
            current_discount *= discount
            
            state = next_state
            
            # 终止条件
            if info['x'] >= 0.98 or info['x'] <= 0.02 or step == max_steps - 1:
                break
                
        costs.append(episode_cost)
        
    avg_cost = np.mean(costs)
    std_cost = np.std(costs)
    
    print(f"Value-Based Policy Performance:")
    print(f"Average Cost: {avg_cost:.4f} ± {std_cost:.4f}")
    print(f"Min Cost: {np.min(costs):.4f}")
    print(f"Max Cost: {np.max(costs):.4f}")
    
    return avg_cost, costs

# ============================================================
# 可视化比较
# ============================================================
def plot_value_comparison(value_learner, env):
    """绘制学习值函数与DP值函数的对比"""
    
    x_states = env.x_states
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Regime 0 对比
    learned_values_0 = []
    for x in x_states:
        state = [x, 0]
        learned_value = value_learner.predict_value(state)
        learned_values_0.append(learned_value)
    
    axes[0].plot(x_states, learned_values_0, 'b-', label='Learned Value', linewidth=2)
    axes[0].plot(x_states, DP.V[:, 0], 'r--', label='DP Optimal Value', linewidth=2)
    axes[0].set_xlabel('Infection Rate (x)')
    axes[0].set_ylabel('Optimal Value')
    axes[0].set_title('Value Function: Learned vs DP (Regime 0)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Regime 1 对比
    learned_values_1 = []
    for x in x_states:
        state = [x, 1]
        learned_value = value_learner.predict_value(state)
        learned_values_1.append(learned_value)
    
    axes[1].plot(x_states, learned_values_1, 'b-', label='Learned Value', linewidth=2)
    axes[1].plot(x_states, DP.V[:, 1], 'r--', label='DP Optimal Value', linewidth=2)
    axes[1].set_xlabel('Infection Rate (x)')
    axes[1].set_ylabel('Optimal Value')
    axes[1].set_title('Value Function: Learned vs DP (Regime 1)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 计算拟合误差
    mse_0 = np.mean((np.array(learned_values_0) - DP.V[:, 0])**2)
    mse_1 = np.mean((np.array(learned_values_1) - DP.V[:, 1])**2)
    
    print(f"Value Function Approximation Error:")
    print(f"Regime 0 MSE: {mse_0:.6f}")
    print(f"Regime 1 MSE: {mse_1:.6f}")
    print(f"Average MSE: {(mse_0 + mse_1)/2:.6f}")

# ============================================================
# 主训练流程
# ============================================================
def main():
    # 创建环境
    env = Envir.PandemicControlEnvironment()
    
    # 创建值函数学习器
    state_dim = 2
    learner = ValueFunctionLearner(state_dim, lr=1e-3)
    
    # 生成训练数据
    print("Step 1: Generating training data from DP optimal values...")
    states, values = generate_value_training_data(env, num_samples=20000)
    
    # 添加训练数据
    learner.add_training_data(states, values)
    print(f"Training data size: {len(learner.training_data)}")
    
    # 训练网络
    print("\nStep 2: Training value function network...")
    losses = []
    for epoch in range(1000):
        loss = learner.train_epoch(batch_size=128, epochs=1)
        losses.append(loss)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/1000, Loss: {loss:.6f}")
    
    # 绘制值函数对比
    print("\nStep 3: Plotting value function comparison...")
    plot_value_comparison(learner, env)
    
    # 创建基于值函数的策略
    print("\nStep 4: Creating value-based policy...")
    policy = ValueBasedPolicy(learner, env)
    
    # 评估策略性能    
    print(f"\n🎯 Final Results:")
    print(f"Value-Based Policy Average Cost: {avg_cost:.4f}")
    
    return learner, policy, avg_cost

if __name__ == "__main__":
    learner, policy, avg_cost = main()