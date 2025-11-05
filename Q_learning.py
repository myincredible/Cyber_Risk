import numpy as np
import matplotlib.pyplot as plt
import random

# ============================================================
# 1. Environment / MCA parameters
# ============================================================
h = 0.05
x_states = np.arange(h, 1.0, h)
n_x = len(x_states)
regimes = [0, 1]

alpha = [0.2, 0.8]
beta = [0.2, 0.8]
gamma = [0.1, 0.2]
sigma_tilde = [0.15, 0.35]
q = np.array([[-1, 1], [1, -1]])
delta = 0.05

a0, aI, aS_m, aI_m, a_r = 0.0, 1.5, 0.5, 1.5, 1.0

actions = [(eta, rho) for eta in np.linspace(0.0, 1.0, 5)
                      for rho in np.linspace(0.0, 1.0, 5)]
n_actions = len(actions)

# ============================================================
# 2. 修复的Drift, volatility, and cost函数
# ============================================================
def b(x, l, eta, rho):
    return eta * alpha[l] * (1 - x) + x * (eta**2 * beta[l] * (1 - x) - (gamma[l] + rho))

def sigma(x, l):
    return sigma_tilde[l] * x * (1 - x)

def f(x, eta, rho):
    return a0 + aI*x + aS_m*(1 - eta)**2 + (aI_m - aS_m)*x*(1 - eta)**2 + a_r*x*(rho**2)

def compute_Qh(x, l, eta, rho):
    drift = b(x, l, eta, rho)
    vol = sigma(x, l)
    q_ii = q[l, l]
    Qh = vol**2 + abs(drift)*h - h**2 *q_ii + h
    return max(Qh, 1e-12), drift, vol

def trans_prob(x, l, eta, rho):
    Qh, drift, vol = compute_Qh(x, l, eta, rho)
    p_up = (vol**2 / 2 + max(drift, 0)*h) / Qh
    p_down = (vol**2 / 2 - min(drift, 0)*h) / Qh
    p_stay = h / Qh
    p_switch = (h**2 * q[l, 1-l]) / Qh
    return {"p_up": p_up, "p_down": p_down, "p_stay": p_stay, "p_switch": p_switch, "Qh": Qh}

def sample_next_state(x, l, eta, rho):
    p = trans_prob(x, l, eta, rho)
    r = random.random()
    
    # 确保概率分布归一化（处理浮点误差）
    p_down = p["p_down"] 
    p_stay = p["p_stay"]
    p_up = p["p_up"]
    p_switch = p["p_switch"]
        
    if r < p_down:
        x_next = max(h, x - h)  # 确保不低于最小边界
        l_next = l
    elif r < p_down + p_stay:
        x_next = x
        l_next = l
    elif r < p_down + p_stay + p_up:
        x_next = min(1.0 - h, x + h)  # 确保不高于最大边界
        l_next = l
    elif r < p_down + p_stay + p_up + p_switch:
        x_next = x
        l_next = 1 - l
        
    return x_next, l_next, p["Qh"]

# ============================================================
# 3. 修复的Q-learning setup
# ============================================================
np.random.seed(0)
random.seed(0)

# 使用更合理的Q值初始化
Q = np.zeros((n_x, len(regimes), n_actions))
visit = np.zeros((n_x, len(regimes), n_actions), dtype=int)

alpha0 = 0.5
alpha_power = 0.6
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_episodes = 300000

episodes = 400000
max_steps = 50
meanQ_hist = []

def get_x_index(x):

    return np.argmin(np.abs(x_states - x))

def best_action_index(xi, l):

    return int(np.argmin(Q[xi, l, :]))


# ============================================================
# 4. 修复的Q-learning main loop
# ============================================================
for ep in range(episodes):
    eps = epsilon_start - (epsilon_start - epsilon_end) * min(ep / epsilon_episodes, 1.0)

    # 安全的状态初始化
    x = np.random.choice(x_states)
    l = np.random.choice(regimes)
    for step in range(max_steps):
        xi = get_x_index(x)

        # ε-greedy action selection
        if np.random.rand() < eps:
            a_idx = np.random.randint(n_actions)
        else:
            a_idx = best_action_index(xi, l)

        eta, rho = actions[a_idx]
        x_next, l_next, Qh_val = sample_next_state(x, l, eta, rho)
        
        # 确保下一个状态有效
        x_next = np.clip(x_next, h, 1.0 - h)

        Delta_t = h**2 / Qh_val
        gamma_step = np.exp(-delta * Delta_t)
        reward = f(x, eta, rho) * Delta_t

        xi_next = get_x_index(x_next)

        visit[xi, l, a_idx] += 1
        
        # 改进的学习率调度
        if ep < episodes * 0.2:
            alpha_lr = 0.25
        else:
            alpha_lr = max(0.01, alpha0 / (1 + 0.01 * visit[xi, l, a_idx]**alpha_power))

        target = reward + gamma_step * np.min(Q[xi_next, l_next, :])         
        Q[xi, l, a_idx] = Q[xi, l, a_idx] + alpha_lr * (target - Q[xi, l, a_idx])

        x, l = x_next, l_next
        
    meanQ_hist.append(np.mean(Q))
    if ep % 1000 == 0: 
        print(f"Ep {ep:6d} | eps={eps:.4f} | meanQ={meanQ_hist[-1]:.6f}")

# ============================================================
# 5. 修复的绘图函数
# ============================================================
def robust_policy_extraction(Q, x_states, regimes, smoothing_iterations=3):
    """更稳健的策略提取方法"""
    
    n_x = len(x_states)
    final_policies = {}
    
    for l in regimes:
        # 第一步：基于最终Q表的初始策略
        raw_policy = [np.argmin(Q[xi, l, :]) for xi in range(n_x)]
        
        # 第二步：多轮平滑
        smoothed_policy = raw_policy.copy()
        
        for iteration in range(smoothing_iterations):
            new_policy = []
            for xi in range(n_x):
                # 考虑邻域的策略一致性
                neighbors = []
                if xi > 0: 
                    neighbors.append(smoothed_policy[xi-1])
                if xi < n_x-1: 
                    # 如果是第一轮，用原始策略；后续用当前平滑策略
                    if iteration == 0:
                        neighbors.append(raw_policy[xi+1])
                    else:
                        neighbors.append(smoothed_policy[xi+1])
                
                current_action = smoothed_policy[xi]
                current_q = Q[xi, l, current_action]
                min_q = np.min(Q[xi, l, :])
                
                # 如果当前动作与邻域不一致，且Q值优势不大，则调整
                if neighbors and current_action not in neighbors:
                    # 如果当前动作优势很小，改用邻域动作
                    if current_q - min_q < 0.02:  # 优势阈值
                        # 选择邻域中出现最多的动作
                        from collections import Counter
                        if neighbors:
                            best_neighbor = Counter(neighbors).most_common(1)[0][0]
                            new_policy.append(best_neighbor)
                        else:
                            new_policy.append(current_action)
                    else:
                        new_policy.append(current_action)
                else:
                    new_policy.append(current_action)
            
            smoothed_policy = new_policy
        
        final_policies[l] = smoothed_policy
    
    return final_policies

optimal_policies = robust_policy_extraction(Q, x_states, regimes, smoothing_iterations=3)

def plot_robust_policy(optimal_policies, x_states, regimes, actions):
    """使用稳健策略绘制图像"""
    
    n_x = len(x_states)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    for regime_idx, l in enumerate(regimes):
        # 获取该regime的策略
        policy_indices = optimal_policies[l]
        
        # 提取策略参数
        eta_policy = [actions[idx][0] for idx in policy_indices]
        rho_policy = [actions[idx][1] for idx in policy_indices]
        
        # 绘制η策略
        axes[0, regime_idx].plot(x_states, eta_policy, 'b-o', markersize=3, linewidth=2)
        axes[0, regime_idx].set_title(f'Robust η Policy - Regime {l}')
        axes[0, regime_idx].set_xlabel('x')
        axes[0, regime_idx].set_ylabel('η')
        axes[0, regime_idx].grid(True)
        
        # 绘制ρ策略
        axes[1, regime_idx].plot(x_states, rho_policy, 'r-o', markersize=3, linewidth=2)
        axes[1, regime_idx].set_title(f'Robust ρ Policy - Regime {l}')
        axes[1, regime_idx].set_xlabel('x')
        axes[1, regime_idx].set_ylabel('ρ')
        axes[1, regime_idx].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 打印策略统计
    print("\n=== 稳健策略统计 ===")
    for l in regimes:
        policy_indices = optimal_policies[l]
        eta_values = [actions[idx][0] for idx in policy_indices]
        rho_values = [actions[idx][1] for idx in policy_indices]
        
        print(f"Regime {l}:")
        print(f"  平均 η: {np.mean(eta_values):.3f}")
        print(f"  平均 ρ: {np.mean(rho_values):.3f}")
        print(f"  策略变化次数: {np.sum(np.diff(policy_indices) != 0)}")

# 绘制稳健策略
plot_robust_policy(optimal_policies, x_states, regimes, actions)


def plot_Q_and_policy(Q, x_states, regimes, actions):
    n_x = len(x_states)
    optimal_Q = np.zeros((n_x, len(regimes)))
    best_action_idx = np.zeros((n_x, len(regimes)), dtype=int)

    for xi in range(n_x):
        for j, l in enumerate(regimes):
            q_vals = Q[xi, l, :]
            # 确保q_vals是1D数组
            if q_vals.ndim > 1:
                q_vals = q_vals.flatten()
            best_idx = int(np.argmin(q_vals))
            optimal_Q[xi, j] = q_vals[best_idx]
            best_action_idx[xi, j] = best_idx

    # Optimal Q
    plt.figure(figsize=(10,4))
    for j, l in enumerate(regimes):
        plt.plot(x_states, optimal_Q[:, j], label=f"Regime {l}")
    plt.xlabel("x"); plt.ylabel("Optimal Q")
    plt.title("Optimal Q vs state (per regime)")
    plt.legend(); plt.grid(True)
    plt.show()

    # Policy
    for j, l in enumerate(regimes):
        eta_chosen = [actions[idx][0] for idx in best_action_idx[:, j]]
        rho_chosen = [actions[idx][1] for idx in best_action_idx[:, j]]

        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(x_states, eta_chosen, '-')
        plt.xlabel("x"); plt.ylabel("η"); plt.title(f"η policy — regime {l}")
        plt.grid(True)
        plt.subplot(1,2,2)
        plt.plot(x_states, rho_chosen, '-', color='tab:orange')
        plt.xlabel("x"); plt.ylabel("ρ"); plt.title(f"ρ policy — regime {l}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# 绘制收敛图
plt.figure(figsize=(8,4))
plt.plot(meanQ_hist, alpha=0.4)
window = 2000
if len(meanQ_hist) > window:
    ma = np.convolve(meanQ_hist, np.ones(window)/window, mode='valid')
    plt.plot(np.arange(window-1, len(meanQ_hist)), ma, 'r', lw=2)
plt.xlabel("Episode")
plt.ylabel("Average Q")
plt.title("Q-learning convergence (smoothed)")
plt.grid(True)
plt.show()

plot_Q_and_policy(Q, x_states, regimes, actions)

# def analyze_q_surface_smoothness(Q, x_states, regimes, actions):
#     """分析Q值曲面的平滑性"""
    
#     for l in regimes:
#         print(f"\n=== Regime {l} Q值曲面分析 ===")
        
#         # 计算每个状态的Q值范围（最大-最小）
#         q_ranges = []
#         for xi, x in enumerate(x_states):
#             q_vals = Q[xi, l, :]
#             q_range = np.max(q_vals) - np.min(q_vals)
#             q_ranges.append(q_range)
            
#             # 输出问题区域
#             if q_range < 0.1:  # 如果Q值差异太小
#                 best_action = np.argmin(q_vals)
#                 second_best = np.argsort(q_vals)[1]  # 次优动作
#                 gap = q_vals[second_best] - q_vals[best_action]
#                 print(f"x={x:.3f}: Q值范围={q_range:.4f}, 最优-次优差距={gap:.4f}")
        
#         # 可视化Q值范围
#         plt.figure(figsize=(10, 4))
#         plt.plot(x_states, q_ranges, 'bo-', markersize=2)
#         plt.axhline(y=0.1, color='red', linestyle='--', label='临界阈值 (0.1)')
#         plt.title(f'Regime {l} - Q值范围分析')
#         plt.xlabel('State x')
#         plt.ylabel('Q值范围 (max-min)')
#         plt.legend()
#         plt.grid(True)
#         plt.show()

# # 运行分析
# analyze_q_surface_smoothness(Q, x_states, regimes, actions)
