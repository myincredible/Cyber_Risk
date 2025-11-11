import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
import Envir  # environment file


# ============================================================
# DQN Network
# ============================================================
class DQN(nn.Module):
    """Deep Q-Network for cost minimization"""
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  
        )

    def forward(self, x):
        return self.model(x)

# ============================================================
# Replay Buffer
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# DQN Agent
# ============================================================
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.SGD(self.policy_net.parameters(), lr=lr, momentum=0.9)
        self.memory = ReplayBuffer(20000)

        self.batch_size = 64
        self.epsilon_start = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 2.5e-4
        self.epsilon = self.epsilon_start

        self.target_update = 500
        self.steps_done = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.policy_net.model[-1].out_features - 1)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.policy_net(s)
            return q.argmin().item()

    def update_epsilon(self, eps):
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * np.exp(-self.epsilon_decay * eps)

    def soft_update_target_net(self, tau=0.01):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)


    def train_step(self, gamma):
        if len(self.memory) < self.batch_size:
            return None

        # 采样经验
        states, actions, rewards, next_states = self.memory.sample(self.batch_size)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        
        # 当前Q值
        current_q_values = self.policy_net(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: 用policy_net选择动作，用target_net评估Q值
        with torch.no_grad():
            # 1. 用policy_net选择下一个状态的最佳动作
            next_actions = self.policy_net(next_states).argmin(1, keepdim=True)
            
            # 2. 用target_net评估这些动作的Q值
            next_q_values = self.target_net(next_states)
            next_q = next_q_values.gather(1, next_actions).squeeze(1)
            
            # 计算目标Q值（无终止状态）
            target_q = rewards + gamma * next_q
        
        # 计算损失
        loss = nn.MSELoss()(current_q, target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        if self.steps_done % self.target_update == 0:
            self.soft_update_target_net(tau = 0.75)
        
        self.steps_done += 1
        
        return loss.item()

    def evaluate_policy_performance(self, env, test_episodes=50, max_steps=750):
        """评估策略性能"""
        test_points = 0.05
        test_regimes = 0
                
        cost_list = []
                
        for episode in range(test_episodes):
            env.step_count = 0
            state = [test_points, test_regimes]
            
            episode_cost = 0.0
            current_discount = 1.0
            
            for step in range(max_steps):
                s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q = self.policy_net(s)
                action = q.argmin().item()  
                
                next_state, reward, discount_factor, info = env.step(action)
                
                episode_cost += current_discount * reward
                current_discount *= discount_factor
                
                state = next_state
                
                if info['x'] >= 0.98 or info['x'] <= 0.02 or step == max_steps - 1:
                    break        
                
            cost_list.append(episode_cost)
        
        return np.mean(cost_list)

    def get_policy_net_weights(self):
        """获取policy net的权重数据用于对比"""
        weights = {}
        for name, param in self.policy_net.named_parameters():
            weights[name] = {
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'min': param.data.min().item(),
                'max': param.data.max().item()
            }
        return weights

# ============================================================
# Training with Evaluation
# ============================================================
def train_dqn(env, episodes=5000, max_steps=100):
    state_dim = 2
    action_dim = env.n_actions
    agent = DQNAgent(state_dim, action_dim)
    
    # 只记录每100次的评估结果
    evaluation_results = []
    
    for ep in range(episodes):
        state = env.reset()

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, gamma_step, info = env.step(action)

            agent.memory.push(state, action, reward, next_state)
            agent.train_step(gamma=gamma_step)
            state = next_state

        agent.update_epsilon(eps=ep)
        
        # 每100轮进行评估
        if ep % 100 == 0:
            print('Evaluating policy performance...')
            test_cost = agent.evaluate_policy_performance(env)
            print(f'The {ep} iteration gives {test_cost}')
            
            # 记录评估结果和网络权重
            weights_info = agent.get_policy_net_weights()
            evaluation_results.append({
                'episode': ep,
                'test_cost': test_cost,
                'epsilon': agent.epsilon,
                'weights': weights_info
            })

    print("Training completed.")
    return agent, evaluation_results

# ============================================================
# Plotting Functions
# ============================================================
def plot_evaluation_results(evaluation_results):
    """绘制评估结果"""
    episodes = [result['episode'] for result in evaluation_results]
    test_costs = [result['test_cost'] for result in evaluation_results]
    epsilons = [result['epsilon'] for result in evaluation_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 测试成本
    ax1.plot(episodes, test_costs, 'bo-', linewidth=2, markersize=6, label='Test Cost')
    ax1.set_xlabel('Training Episode')
    ax1.set_ylabel('Test Cost')
    ax1.set_title('Policy Performance Evaluation')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 探索率
    ax2.plot(episodes, epsilons, 'r-', linewidth=2, label='Exploration Rate')
    ax2.set_xlabel('Training Episode')
    ax2.set_ylabel('Epsilon')
    ax2.set_title('Exploration Rate Decay')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 打印性能统计
    print(f"\n📊 Performance Summary:")
    print(f"Final Cost: {test_costs[-1]:.4f}")
    print(f"Best Cost: {min(test_costs):.4f}")
    print(f"Average Cost: {np.mean(test_costs):.4f}")

def plot_policy_net_comparison(evaluation_results):
    """对比policy net权重变化"""
    if len(evaluation_results) < 2:
        print("Not enough data for policy net comparison")
        return
    
    episodes = [result['episode'] for result in evaluation_results]
    
    # 选择第一个层的权重进行对比
    weight_key = 'model.0.weight'
    if weight_key in evaluation_results[0]['weights']:
        weight_means = [result['weights'][weight_key]['mean'] for result in evaluation_results]
        weight_stds = [result['weights'][weight_key]['std'] for result in evaluation_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(episodes, weight_means, 'g-', linewidth=2, label='Weight Mean')
        ax1.set_xlabel('Training Episode')
        ax1.set_ylabel('Weight Mean')
        ax1.set_title('Policy Net Weight Means')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(episodes, weight_stds, 'purple', linewidth=2, label='Weight Std')
        ax2.set_xlabel('Training Episode')
        ax2.set_ylabel('Weight Std')
        ax2.set_title('Policy Net Weight Standard Deviations')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

def extract_policy(agent, env):
    """提取最终策略"""
    policy = {}
    agent.policy_net.eval()
    
    for l in env.regimes:
        policy[l] = []
        for x in env.x_states:
            state = torch.tensor([x, l], dtype=torch.float32).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q_values = agent.policy_net(state)
                best_action = int(q_values.argmin().item())
            policy[l].append(best_action)
    return policy

def plot_final_policy(policy, env):
    """绘制最终策略"""
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    
    for l in env.regimes:
        eta_policy = []
        rho_policy = []
        for a_idx in policy[l]:
            action = env.actions[int(a_idx)]
            if isinstance(action, (list, tuple, np.ndarray)) and len(action) == 2:
                eta_policy.append(action[0])
                rho_policy.append(action[1])
            else:
                eta_policy.append(action)
                rho_policy.append(action)
        
        ax[0].plot(env.x_states, eta_policy, label=f"Regime {l}", linewidth=2)
        ax[1].plot(env.x_states, rho_policy, label=f"Regime {l}", linewidth=2)
    
    ax[0].set_ylabel("η Policy")
    ax[0].set_title("Final Policy: η Component")
    ax[1].set_ylabel("ρ Policy")
    ax[1].set_xlabel("Infection Rate x")
    ax[1].set_title("Final Policy: ρ Component")
    
    for a in ax:
        a.grid(True)
        a.legend()
    
    plt.tight_layout()
    plt.show()

def plot_final_q_function(agent, env):
    """绘制最终Q函数"""
    x_grid = np.arange(env.h, 1.0, env.h)
    Q_star = np.zeros((len(x_grid), len(env.regimes)))

    agent.policy_net.eval()
    with torch.no_grad():
        for i, x in enumerate(x_grid):
            for j, l in enumerate(env.regimes):
                state = torch.tensor([x, l], dtype=torch.float32).to(agent.device)
                q = agent.policy_net(state)
                Q_star[i, j] = torch.min(q).item()

    plt.figure(figsize=(8, 6))
    plt.plot(x_grid, Q_star[:, 0], label='Regime 0', lw=2)
    plt.plot(x_grid, Q_star[:, 1], label='Regime 1', lw=2)
    plt.xlabel("Infection Rate x")
    plt.ylabel("Optimal Q*(x, l)")
    plt.title("Final Optimal Q-Value Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    env = Envir.PandemicControlEnvironment()
    
    # 训练并获取评估结果
    agent, evaluation_results = train_dqn(env, episodes=25000, max_steps=100)
    
    # 绘制评估结果
    plot_evaluation_results(evaluation_results)
    
    # 绘制policy net对比
    plot_policy_net_comparison(evaluation_results)
    
    # 提取并绘制最终策略
    policy = extract_policy(agent, env)
    plot_final_policy(policy, env)
    
    # 绘制最终Q函数
    plot_final_q_function(agent, env)
    
    # 最终性能报告
    final_test_cost = agent.evaluate_policy_performance(env)
    print(f"\n🎯 Final Performance Report:")
    print(f"Final Test Cost: {final_test_cost:.4f}")
    
    # 打印最终网络权重信息
    final_weights = agent.get_policy_net_weights()
    print(f"\n🔧 Final Policy Net Weights Summary:")
    for name, info in final_weights.items():
        print(f"  {name}: mean={info['mean']:.4f}, std={info['std']:.4f}")