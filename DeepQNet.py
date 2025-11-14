import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
import Envir  # environment file
import Plot

# ============================================================
# 使用Sigmoid的稳定DQN网络
# ============================================================
class StableDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super(StableDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),  
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


# ============================================================
# 改进的经验回放
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity=50000):  # 增大缓冲区
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
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
# 改进的稳定DQN Agent
# ============================================================
class StableDQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-5):  # 大幅降低学习率
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.action_dim = action_dim
        
        # 使用Sigmoid的稳定网络
        self.policy_net = StableDQN(state_dim, action_dim).to(self.device)
        self.target_net = StableDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器与梯度裁剪
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(50000)

        # 更稳定的超参数
        self.batch_size = 128  # 增大批大小
        self.epsilon_start = 1.0
        self.epsilon_min = 0.05  # 保留少量探索
        self.epsilon_decay = 5e-5  # 更慢的衰减
        self.epsilon = self.epsilon_start

        # 改为软更新
        self.tau = 0.001  # 软更新参数
        self.steps_done = 0
        
        # 训练监控
        self.losses = []
        self.grad_norms = []

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.policy_net(s)
            return q.argmin().item()  # 注意：这里是最小化成本

    def update_epsilon(self):
        """逐步衰减探索率"""
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * \
                      np.exp(-self.epsilon_decay * self.steps_done)

    def soft_update(self):
        """软更新目标网络"""
        for target_param, policy_param in zip(self.target_net.parameters(), 
                                            self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + 
                                  (1.0 - self.tau) * target_param.data)

    def train_step(self, gamma):
        if len(self.memory) < self.batch_size:
            return None

        sample = self.memory.sample(self.batch_size)
        if sample is None:
            return None
            
        states, actions, rewards, next_states = sample
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        
        # 当前Q值
        current_q_values = self.policy_net(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN
        with torch.no_grad():
            # 注意：这里是最小化成本，所以用argmin
            next_actions = self.policy_net(next_states).argmin(1, keepdim=True)
            next_q_values = self.target_net(next_states)
            next_q = next_q_values.gather(1, next_actions).squeeze(1)
            target_q = rewards + gamma * next_q
        
        # Huber损失更稳定
        loss = F.smooth_l1_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度监控和裁剪
        total_norm = 0
        for p in self.policy_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.grad_norms.append(total_norm)
        
        # 更严格的梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
        
        self.optimizer.step()
        
        # 软更新目标网络
        self.soft_update()
        
        self.steps_done += 1
        self.losses.append(loss.item())
        
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
    
    def get_training_metrics(self):
        """获取训练指标"""
        if len(self.losses) == 0:
            return 0, 0
        recent_losses = self.losses[-100:] if len(self.losses) >= 100 else self.losses
        recent_grads = self.grad_norms[-100:] if len(self.grad_norms) >= 100 else self.grad_norms
        return np.mean(recent_losses), np.mean(recent_grads)


# ============================================================
# 改进的训练函数
# ============================================================
def train_dqn(env, episodes=10000, max_steps=100):
    state_dim = 2
    action_dim = env.n_actions
    agent = StableDQNAgent(state_dim, action_dim, lr=1e-5)  # 使用很低的学习率
    
    evaluation_results = []
    best_cost = float('inf')
    no_improvement_count = 0
    
    print("Pre-filling replay buffer...")
    # 预填充经验回放缓冲区
    while len(agent.memory) < 10000:
        state = env.reset()
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, gamma_step, info = env.step(action)
            agent.memory.push(state, action, reward, next_state)
            state = next_state
            if info.get('done', False) or step == max_steps - 1:
                break
    
    print(f"Replay buffer filled with {len(agent.memory)} samples")
    
    for ep in range(episodes):
        state = env.reset()
        episode_losses = []

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, gamma_step, info = env.step(action)

            agent.memory.push(state, action, reward, next_state)
            loss = agent.train_step(gamma=gamma_step)
            if loss is not None:
                episode_losses.append(loss)
                
            state = next_state
            if info.get('done', False) or step == max_steps - 1:
                break

        agent.update_epsilon()

        # 每100轮进行评估和记录
        if ep % 100 == 0:
            avg_loss, avg_grad_norm = agent.get_training_metrics()
            test_cost = agent.evaluate_policy_performance(env)
            
            print(f'Episode {ep:5d} | Test Cost: {test_cost:8.4f} | '
                  f'Loss: {avg_loss:8.4f} | Grad Norm: {avg_grad_norm:8.4f} | '
                  f'Epsilon: {agent.epsilon:.4f}')
            
            evaluation_results.append({
                'episode': ep,
                'test_cost': test_cost,
                'loss': avg_loss,
                'grad_norm': avg_grad_norm,
                'epsilon': agent.epsilon,
            })
            
            # 保存最佳模型（基于测试成本）
            if test_cost < best_cost:
                best_cost = test_cost
                torch.save(agent.policy_net.state_dict(), 'best_model.pth')
                no_improvement_count = 0
                print(f"New best model saved with cost: {best_cost:.4f}")
            else:
                no_improvement_count += 1
            
            # 早停检查
            if no_improvement_count > 50:  # 连续50次评估无改进
                print(f"Early stopping at episode {ep}")
                break

    print("Training completed.")
    # 加载最佳模型
    agent.policy_net.load_state_dict(torch.load('best_model.pth'))
    return agent, evaluation_results


# ============================================================
# 改进的绘图函数
# ============================================================
def plot_evaluation_results(evaluation_results):
    if not evaluation_results:
        print("No evaluation results to plot")
        return
        
    episodes = [result['episode'] for result in evaluation_results]
    test_costs = [result['test_cost'] for result in evaluation_results]
    losses = [result['loss'] for result in evaluation_results]
    grad_norms = [result['grad_norm'] for result in evaluation_results]
    epsilons = [result['epsilon'] for result in evaluation_results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 测试成本
    ax1.plot(episodes, test_costs, 'bo-', linewidth=2, markersize=4, label='Test Cost')
    ax1.set_xlabel('Training Episode')
    ax1.set_ylabel('Test Cost')
    ax1.set_title('Policy Performance')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 损失函数
    ax2.plot(episodes, losses, 'r-', linewidth=2, label='Training Loss')
    ax2.set_xlabel('Training Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 梯度范数
    ax3.plot(episodes, grad_norms, 'g-', linewidth=2, label='Gradient Norm')
    ax3.set_xlabel('Training Episode')
    ax3.set_ylabel('Gradient Norm')
    ax3.set_title('Gradient Norm (Clipped at 0.5)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 探索率
    ax4.plot(episodes, epsilons, 'm-', linewidth=2, label='Exploration Rate')
    ax4.set_xlabel('Training Episode')
    ax4.set_ylabel('Epsilon')
    ax4.set_title('Exploration Rate Decay')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nPerformance Summary:")
    print(f"Final Cost: {test_costs[-1]:.4f}")
    print(f"Best Cost: {min(test_costs):.4f}")
    print(f"Average Cost: {np.mean(test_costs):.4f}")


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    env = Envir.PandemicControlEnvironment()
    
    print("Starting Stable DQN Training...")
    print("Key improvements:")
    print("- Very low learning rate (1e-5)")
    print("- Soft target network updates (tau=0.001)")
    print("- Strict gradient clipping (max_norm=0.5)")
    print("- Larger batch size (128)")
    print("- Pre-filled replay buffer")
    print("- Comprehensive monitoring")
    
    agent, evaluation_results = train_dqn(env, episodes=10000, max_steps=100)
    
    plot_evaluation_results(evaluation_results)
    
    final_test_cost = agent.evaluate_policy_performance(env)
    print(f"\nFinal Performance: {final_test_cost:.4f}")
    
    # 绘制策略和Q值分布
    Plot.plot_optimal_policy_and_q(agent, env)
    Plot.plot_q_value_distribution(agent, env)