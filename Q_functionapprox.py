import numpy as np
import random
import matplotlib.pyplot as plt
import Envir

# --- 简化版的Agent ---
class SimpleTDAgent:
    def __init__(self, env, alpha=0.005, lam=0, epsilon_decay=0.0001, min_epsilon=0.05):
        self.env = env
        self.alpha = alpha  # 非常小的学习率
        self.lam = lam
        self.epsilon_decay = epsilon_decay
        self.epsilon_start = 1.0
        self.epsilon_min = min_epsilon
        self.epsilon = self.epsilon_start
        
        # 简化特征：只使用线性特征
        self.n_features = 2  # [1, x, l]
        self.n_actions = env.n_actions
        
        # 更小的权重初始化
        self.weights = np.random.normal(0, 0.1, (self.n_actions, self.n_features))
        
        # Eligibility traces
        self.e_trace = np.zeros_like(self.weights)
    
    def featurize(self, state):
        x, l = state
        # 简单特征，避免非线性
        return np.array([1.0, x, l], dtype=np.float32)
    
    def predict(self, state):
        phi = self.featurize(state)
        return np.dot(self.weights, phi)
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        q_values = self.predict(state)
        return int(np.argmin(q_values))  # 最小化成本
    
    def update(self, state, action, reward, next_state, discount_factor):
        phi = self.featurize(state)
        q_current = np.dot(self.weights[action], phi)
        
        # 下一个状态的Q值
        q_next_values = self.predict(next_state)
        q_next = np.min(q_next_values)
        
        # TD error - 添加梯度裁剪
        delta = reward + discount_factor * q_next - q_current
        delta = np.clip(delta, -10, 10)  # 防止TD误差爆炸
        
        # 更新eligibility traces
        self.e_trace = self.e_trace * discount_factor * self.lam
        self.e_trace[action] += phi
        
        # 权重更新 - 添加学习率衰减和梯度裁剪
        update = self.alpha * delta * self.e_trace
        # update = np.clip(update, -0.1, 0.1)  # 限制更新幅度
        self.weights += update
    
    def decay_epsilon(self, eps):
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * np.exp(-self.epsilon_decay * eps)
    
    def reset_traces(self):
        self.e_trace.fill(0.0)

# --- 调试和监控工具 ---
class TrainingMonitor:
    def __init__(self):
        self.rewards = []
        self.q_values = []
        self.weights_norm = []
        self.deltas = []
    
    def record(self, reward, q_vals, weights, delta):
        self.rewards.append(reward)
        self.q_values.append(np.max(np.abs(q_vals)))
        self.weights_norm.append(np.linalg.norm(weights))
        self.deltas.append(delta)
    
    def plot_progress(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # 奖励
        ax1.plot(self.rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_ylabel('Total Reward')
        
        # Q值范围
        ax2.plot(self.q_values)
        ax2.set_title('Max |Q| Value')
        ax2.set_ylabel('Q Value')
        
        # 权重范数
        ax3.plot(self.weights_norm)
        ax3.set_title('Weights Norm')
        ax3.set_ylabel('Norm')
        
        # TD误差
        ax4.plot(self.deltas)
        ax4.set_title('TD Errors')
        ax4.set_ylabel('Delta')
        
        plt.tight_layout()
        plt.show()

# --- 安全的训练函数 ---
def safe_train(env, agent, n_episodes=2000, max_steps=50):
    monitor = TrainingMonitor()
    
    print("开始安全训练...")
    for ep in range(n_episodes):
        state = env.reset()
        agent.reset_traces()
        total_reward = 0
        total_delta = 0
        steps = 0
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, discount, info = env.step(action)
            
            # 转换并缩放奖励
            scaled_reward = -reward / 1.0 
            
            # 记录更新前的状态
            old_weights = agent.weights.copy()
            
            # 更新
            agent.update(state, action, scaled_reward, next_state, discount)
            
            # 计算变化
            weight_change = np.linalg.norm(agent.weights - old_weights)
            total_delta += weight_change
            total_reward += scaled_reward
            
            state = next_state
            steps += 1
            
            # 检查数值稳定性
            if np.any(np.isnan(agent.weights)) or np.any(np.isinf(agent.weights)):
                print(f"Episode {ep}: 数值爆炸！停止训练")
                return monitor
        
        # 缓慢衰减epsilon
        if ep % 50 == 0:
            agent.decay_epsilon(ep)
        
        # 记录监控数据
        current_q = agent.predict(state)
        monitor.record(total_reward, current_q, agent.weights, total_delta)
        
        # 定期检查
        if (ep + 1) % 1000 == 0:
            q_max = np.max(np.abs(current_q))
            weight_norm = np.linalg.norm(agent.weights)
            print(f"Episode {ep+1}: Eps={agent.epsilon:.3f}, "
                  f"Max|Q|={q_max:.4f}, W_norm={weight_norm:.4f}, "
                  f"Avg Reward={np.mean(monitor.rewards[-100:]):.4f}")
            
            # 安全检查
            if q_max > 1000 or weight_norm > 100:
                print("检测到数值问题，停止训练")
                break
    
    return monitor

# --- 主训练流程 ---
env = Envir.PandemicControlEnvironment()
agent = SimpleTDAgent(env, alpha=0.001, lam=0.0, epsilon_decay=0.005, min_epsilon=0.1)

final_monitor = safe_train(env, agent, n_episodes=300000, max_steps=50)

# --- 分析和可视化结果 ---
def analyze_agent(agent, env):
    x_states = env.x_states
    regimes = env.regimes
    
    print("\n=== 最终分析 ===")
    print(f"权重范围: [{np.min(agent.weights):.6f}, {np.max(agent.weights):.6f}]")
    print(f"权重范数: {np.linalg.norm(agent.weights):.6f}")
    
    # 计算Q值
    Q_values = {}
    for l in regimes:
        Q_values[l] = np.zeros((agent.n_actions, len(x_states)))
        for a_idx in range(agent.n_actions):
            for i, x in enumerate(x_states):
                state = np.array([x, l], dtype=np.float32)
                Q_values[l][a_idx, i] = np.dot(agent.weights[a_idx], agent.featurize(state))
    
    # 检查Q值范围
    for l in regimes:
        q_min = np.min(Q_values[l])
        q_max = np.max(Q_values[l])
        print(f"Regime {l}: Q值范围 [{q_min:.4f}, {q_max:.4f}]")
    
    return Q_values

def plot_final_results(Q_values, env):
    x_states = env.x_states
    regimes = env.regimes
    
    for l in regimes:
        plt.figure(figsize=(15, 4))
        
        # Q函数
        plt.subplot(1, 3, 1)
        for a_idx, action in enumerate(env.actions):
            plt.plot(x_states, Q_values[l][a_idx], 
                    label=f"η={action[0]:.1f}, ρ={action[1]:.1f}", alpha=0.7)
        plt.xlabel("Infection rate x")
        plt.ylabel("Q-value")
        plt.title(f"Q-function - Regime {l}")
        plt.legend()
        
        # 最优策略
        plt.subplot(1, 3, 2)
        optimal_indices = np.argmin(Q_values[l], axis=0)
        optimal_eta = [env.actions[i][0] for i in optimal_indices]
        optimal_rho = [env.actions[i][1] for i in optimal_indices]
        
        plt.plot(x_states, optimal_eta, 'b-', label="Optimal η")
        plt.plot(x_states, optimal_rho, 'r-', label="Optimal ρ")
        plt.xlabel("Infection rate x")
        plt.ylabel("Optimal action")
        plt.title(f"Optimal Policy - Regime {l}")
        plt.legend()
        plt.ylim(-0.1, 1.1)
        
        # Q值热图
        plt.subplot(1, 3, 3)
        plt.imshow(Q_values[l], aspect='auto', cmap='viridis',
                  extent=[x_states[0], x_states[-1], agent.n_actions-0.5, -0.5])
        plt.colorbar(label='Q Value')
        plt.xlabel("Infection rate x")
        plt.ylabel("Action Index")
        plt.title(f"Q Values Heatmap - Regime {l}")
        
        plt.tight_layout()
        plt.show()

# 执行分析
Q_values = analyze_agent(agent, env)
plot_final_results(Q_values, env)

# 测试策略
print("\n=== 策略测试 ===")
test_states = [
    np.array([0.1, 0]),  # 低感染率
    np.array([0.5, 0]),  # 中感染率  
    np.array([0.9, 0]),  # 高感染率
]

for state in test_states:
    q_vals = agent.predict(state)
    best_action = np.argmin(q_vals)
    print(f"状态 x={state[0]:.1f}, l={state[1]}: "
          f"最佳动作 {best_action} (η={env.actions[best_action][0]:.1f}, ρ={env.actions[best_action][1]:.1f})")
    print(f"  各动作Q值: {[f'{q:.3f}' for q in q_vals]}")