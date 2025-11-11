import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import random
from Envir import PandemicControlEnvironment

class StablePolicyNetwork(nn.Module):
    """数值稳定的策略网络"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(StablePolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),  # 使用Tanh避免梯度爆炸
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        
    def forward(self, x):
        logits = self.network(x)
        return torch.softmax(logits, dim=-1)
    
    def get_action(self, state):
        """根据状态选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.forward(state_tensor)
        
        # 添加数值稳定性检查
        if torch.isnan(action_probs).any():
            print("Warning: NaN detected in action probabilities, using uniform distribution")
            action_probs = torch.ones_like(action_probs) / action_probs.shape[-1]
        
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

class StableREINFORCE:
    
    def __init__(self, state_dim, action_dim, learning_rate=1e-4):  # 降低学习率
        self.policy_net = StablePolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate, weight_decay=1e-5)  # 添加权重衰减
        
        # 存储经验
        self.episode_log_probs = []
        self.episode_rewards = []
        self.episode_discounts = []
    
    def select_action(self, state):
        """选择动作"""
        return self.policy_net.get_action(state)
    
    def store_transition(self, log_prob, reward, discount):
        """存储转换经验"""
        self.episode_log_probs.append(log_prob)
        self.episode_rewards.append(reward)
        self.episode_discounts.append(discount)
    
    def update_policy(self):
        """使用环境折旧因子更新策略"""
        if len(self.episode_rewards) == 0:
            return 0
        
        # 使用环境提供的折旧因子计算折扣回报
        returns = []
        R = 0
        for i in range(len(self.episode_rewards)-1, -1, -1):
            r = self.episode_rewards[i]
            discount = self.episode_discounts[i]
            R = r + discount * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns)
        
        # 标准化回报以降低方差
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 计算策略损失
        policy_loss = []
        for log_prob, R in zip(self.episode_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # 添加梯度裁剪
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)  # 梯度裁剪
        self.optimizer.step()
        
        # 清空经验
        self.episode_log_probs = []
        self.episode_rewards = []
        self.episode_discounts = []
        
        return policy_loss.item()

def train_reinforce_stable(env, num_episodes=500, max_steps=200, learning_rate=1e-4):
    """训练稳定的REINFORCE算法"""
    
    state_dim = 2  # [x, regime]
    action_dim = env.n_actions
    
    agent = StableREINFORCE(state_dim, action_dim, learning_rate)
    
    # 记录训练过程
    episode_rewards = []
    episode_costs = []
    episode_losses = []
    
    print("开始训练稳定的REINFORCE算法...")
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0
        current_discount = 1.0
        
        while step_count < max_steps:
            try:
                # 选择动作
                action_idx, log_prob = agent.select_action(state)
                
                # 执行动作
                next_state, cost, discount, info = env.step(action_idx)
                current_discount *= discount
                
                # 注意：环境返回的是成本，我们取负作为奖励
                reward = -cost * current_discount
                
                # 存储经验
                agent.store_transition(log_prob, reward, discount)
                
                state = next_state
                total_reward += reward
                step_count += 1
                
                # 如果感染率达到边界，提前终止
                if env.x >= 0.95 or env.x <= 0.05:
                    break
                    
            except Exception as e:
                print(f"Error in step {step_count}: {e}")
                break
        
        # 更新策略
        if len(agent.episode_rewards) > 0:
            loss = agent.update_policy()
            episode_losses.append(loss)
        else:
            episode_losses.append(0)
        
        episode_rewards.append(total_reward)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_cost = np.mean(episode_costs[-50:])
            avg_loss = np.mean(episode_losses[-50:]) if episode_losses else 0
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Avg Loss: {avg_loss:.2f}, "
                  f"Steps: {step_count}")
    
    return agent, episode_rewards, episode_costs, episode_losses

def evaluate_policy_stable(env, agent, num_episodes=1000, max_steps=500):
    """评估训练好的策略"""
    print("\n评估策略...")
    
    episode_costs = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_cost = 0
        step_count = 0
        
        while step_count < max_steps:
            action_idx, _ = agent.select_action(state)
            next_state, cost, discount, info = env.step(action_idx)
            
            state = next_state
            total_cost += cost
            step_count += 1
            
            if env.x >= 0.95 or env.x <= 0.05:
                break
        
        episode_costs.append(total_cost)
    
    avg_cost = np.mean(episode_costs)
    std_cost = np.std(episode_costs)
    
    print(f"平均成本: {avg_cost:.4f} ± {std_cost:.4f}")
    return avg_cost, std_cost, episode_costs

def plot_results_stable(episode_rewards, episode_costs, episode_losses):
    """绘制结果图"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    # 计算移动平均
    window = 20
    
    # 绘制奖励曲线
    if len(episode_rewards) >= window:
        rewards_ma = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(episode_rewards, alpha=0.3, linewidth=0.8, label='每回合奖励')
        ax1.plot(range(window-1, len(episode_rewards)), rewards_ma, linewidth=1.5, label=f'{window}回合移动平均')
    else:
        ax1.plot(episode_rewards, linewidth=1.5, label='每回合奖励')
    ax1.set_xlabel('回合数')
    ax1.set_ylabel('累计奖励')
    ax1.set_title('训练奖励曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制成本曲线
    if len(episode_costs) >= window:
        costs_ma = np.convolve(episode_costs, np.ones(window)/window, mode='valid')
        ax2.plot(episode_costs, alpha=0.3, linewidth=0.8, color='red', label='每回合成本')
        ax2.plot(range(window-1, len(episode_costs)), costs_ma, linewidth=1.5, color='darkred', label=f'{window}回合移动平均')
    else:
        ax2.plot(episode_costs, linewidth=1.5, color='red', label='每回合成本')
    ax2.set_xlabel('回合数')
    ax2.set_ylabel('累计成本')
    ax2.set_title('训练成本曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 绘制损失曲线
    if episode_losses and len(episode_losses) >= window:
        losses_ma = np.convolve(episode_losses, np.ones(window)/window, mode='valid')
        ax3.plot(episode_losses, alpha=0.3, linewidth=0.8, color='green', label='每回合损失')
        ax3.plot(range(window-1, len(episode_losses)), losses_ma, linewidth=1.5, color='darkgreen', label=f'{window}回合移动平均')
        ax3.set_xlabel('回合数')
        ax3.set_ylabel('策略损失')
        ax3.set_title('训练损失曲线')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 设置随机种子以便复现
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 创建环境
    env = PandemicControlEnvironment()
    
    print("环境信息:")
    print(f"状态空间: {len(env.x_states)}个离散感染率状态 + 2个regime")
    print(f"动作空间: {env.n_actions}个动作")
    print(f"环境折旧率delta: {env.delta}")
    
    # 训练稳定的REINFORCE算法
    agent, episode_rewards, episode_costs, episode_losses = train_reinforce_stable(
        env, 
        num_episodes=5000,  # 先训练少一些回合
        max_steps=500,
        learning_rate=1e-4  # 使用更小的学习率
    )
    
    # 绘制结果
    plot_results_stable(episode_rewards, episode_costs, episode_losses)
    
    # 评估训练好的策略
    avg_cost, std_cost, test_costs = evaluate_policy_stable(env, agent)
    
    # 与随机策略对比
    print("\n与随机策略对比:")
    random_costs = []
    for _ in range(100):
        state = env.reset()
        total_cost = 0
        step_count = 0
        while step_count < 100:
            action_idx = random.randint(0, env.n_actions - 1)
            next_state, cost, discount, info = env.step(action_idx)
            total_cost += cost
            state = next_state
            step_count += 1
            if env.x >= 0.95 or env.x <= 0.05:
                break
        random_costs.append(total_cost)
    
    random_avg_cost = np.mean(random_costs)
    print(f"随机策略平均成本: {random_avg_cost:.4f}")
    print(f"REINFORCE平均成本: {avg_cost:.4f}")
    
    if random_avg_cost > 0:
        improvement = ((random_avg_cost - avg_cost) / random_avg_cost) * 100
        print(f"成本降低: {improvement:.1f}%")