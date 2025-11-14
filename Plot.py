import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import Envir  # 环境文件

def plot_optimal_policy_and_q(agent, env, dp_solver=None):
    """绘制最优策略和Q函数"""
    
    # 创建网格
    x_states = np.arange(env.h, 1.0, env.h)
    regimes = env.regimes
    
    # 初始化存储数组
    Q_values = np.zeros((len(x_states), len(regimes), env.n_actions))
    policy = np.zeros((len(x_states), len(regimes)))
    policy_actions = np.zeros((len(x_states), len(regimes), 2))  # 存储具体动作
    
    # 计算Q值和策略
    agent.policy_net.eval()
    with torch.no_grad():
        for i, x in enumerate(x_states):
            for j, l in enumerate(regimes):
                state = torch.tensor([x, l], dtype=torch.float32).to(agent.device)
                q_vals = agent.policy_net(state).cpu().numpy()
                Q_values[i, j] = q_vals
                best_action_idx = np.argmin(q_vals)  # 最小化成本
                policy[i, j] = best_action_idx
                
                # 获取具体动作值
                action = env.actions[int(best_action_idx)]
                if isinstance(action, (list, tuple, np.ndarray)) and len(action) == 2:
                    policy_actions[i, j, 0] = action[0]  # eta
                    policy_actions[i, j, 1] = action[1]  # rho
                else:
                    policy_actions[i, j, 0] = action
                    policy_actions[i, j, 1] = action

    # 创建图形
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 最优Q值曲面 (3D)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    X, L = np.meshgrid(x_states, regimes, indexing='ij')
    optimal_Q = np.min(Q_values, axis=2)  # 每个状态的最小Q值
    
    surf = ax1.plot_surface(X, L, optimal_Q, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Infection Rate (x)')
    ax1.set_ylabel('Regime (l)')
    ax1.set_zlabel('Optimal Q-Value')
    ax1.set_title('Optimal Q-Function Surface')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=20)
    
    # 2. 最优Q值等高线
    ax2 = fig.add_subplot(2, 3, 2)
    contour = ax2.contourf(X, L, optimal_Q, levels=20, cmap='viridis')
    ax2.set_xlabel('Infection Rate (x)')
    ax2.set_ylabel('Regime (l)')
    ax2.set_title('Optimal Q-Function Contour')
    plt.colorbar(contour, ax=ax2)
    
    # 3. 策略热图
    ax3 = fig.add_subplot(2, 3, 3)
    im = ax3.imshow(policy.T, extent=[env.h, 1.0-env.h, regimes[-1], regimes[0]], 
                   aspect='auto', cmap='tab10')
    ax3.set_xlabel('Infection Rate (x)')
    ax3.set_ylabel('Regime (l)')
    ax3.set_title('Optimal Policy (Action Index)')
    plt.colorbar(im, ax=ax3)
    
    # 4. eta策略分量
    ax4 = fig.add_subplot(2, 3, 4)
    for j, l in enumerate(regimes):
        ax4.plot(x_states, policy_actions[:, j, 0], label=f'Regime {l}', linewidth=2)
    ax4.set_xlabel('Infection Rate (x)')
    ax4.set_ylabel('η Policy')
    ax4.set_title('η Component of Optimal Policy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. rho策略分量
    ax5 = fig.add_subplot(2, 3, 5)
    for j, l in enumerate(regimes):
        ax5.plot(x_states, policy_actions[:, j, 1], label=f'Regime {l}', linewidth=2)
    ax5.set_xlabel('Infection Rate (x)')
    ax5.set_ylabel('ρ Policy')
    ax5.set_title('ρ Component of Optimal Policy')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 与DP解对比 (如果有的话)
    ax6 = fig.add_subplot(2, 3, 6)
    if dp_solver is not None:
        # 计算DP解
        dp_q_values = np.zeros((len(x_states), len(regimes)))
        for i, x in enumerate(x_states):
            for j, l in enumerate(regimes):
                dp_q_values[i, j] = dp_solver.get_optimal_value(x, l)
        
        # 绘制对比
        line1, = ax6.plot(x_states, optimal_Q[:, 0], 'b-', label='DQN Regime 0', linewidth=2)
        line2, = ax6.plot(x_states, optimal_Q[:, 1], 'r-', label='DQN Regime 1', linewidth=2)
        line3, = ax6.plot(x_states, dp_q_values[:, 0], 'b--', label='DP Regime 0', linewidth=2, alpha=0.7)
        line4, = ax6.plot(x_states, dp_q_values[:, 1], 'r--', label='DP Regime 1', linewidth=2, alpha=0.7)
        
        ax6.set_xlabel('Infection Rate (x)')
        ax6.set_ylabel('Optimal Value')
        ax6.set_title('DQN vs DP Solution Comparison')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        # 如果没有DP解，显示Q值随状态的变化
        for j, l in enumerate(regimes):
            ax6.plot(x_states, optimal_Q[:, j], label=f'Regime {l}', linewidth=2)
        ax6.set_xlabel('Infection Rate (x)')
        ax6.set_ylabel('Optimal Q-Value')
        ax6.set_title('Optimal Q-Value by Regime')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印策略统计
    print("\n📊 策略分析:")
    print(f"动作空间大小: {env.n_actions}")
    print(f"状态空间大小: {len(x_states)} x {len(regimes)}")
    
    # 分析策略模式
    unique_actions = np.unique(policy)
    print(f"使用的独特动作数量: {len(unique_actions)}")
    
    for action_idx in unique_actions:
        count = np.sum(policy == action_idx)
        percentage = count / policy.size * 100
        action_desc = env.actions[int(action_idx)] if hasattr(env, 'actions') else f"Action {action_idx}"
        print(f"  动作 {action_idx} ({action_desc}): {count} 状态 ({percentage:.1f}%)")

def plot_q_value_distribution(agent, env):
    """绘制Q值分布统计"""
    x_states = np.arange(env.h, 1.0, env.h)
    regimes = env.regimes
    
    all_q_values = []
    agent.policy_net.eval()
    
    with torch.no_grad():
        for x in x_states:
            for l in regimes:
                state = torch.tensor([x, l], dtype=torch.float32).to(agent.device)
                q_vals = agent.policy_net(state).cpu().numpy()
                all_q_values.extend(q_vals)
    
    all_q_values = np.array(all_q_values)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Q值分布直方图
    ax1.hist(all_q_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(all_q_values), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_q_values):.3f}')
    ax1.axvline(np.median(all_q_values), color='green', linestyle='--', 
                label=f'Median: {np.median(all_q_values):.3f}')
    ax1.set_xlabel('Q-Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Q-Value Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q值范围统计
    q_stats = {
        'Min': np.min(all_q_values),
        'Max': np.max(all_q_values), 
        'Mean': np.mean(all_q_values),
        'Std': np.std(all_q_values),
        '25%': np.percentile(all_q_values, 25),
        '75%': np.percentile(all_q_values, 75)
    }
    
    ax2.bar(range(len(q_stats)), list(q_stats.values()), color='lightcoral')
    ax2.set_xticks(range(len(q_stats)))
    ax2.set_xticklabels(list(q_stats.keys()))
    ax2.set_ylabel('Q-Value')
    ax2.set_title('Q-Value Statistics')
    
    # 在柱子上添加数值
    for i, v in enumerate(q_stats.values()):
        ax2.text(i, v + 0.1 * max(q_stats.values()), f'{v:.2f}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📈 Q值统计:")
    for stat, value in q_stats.items():
        print(f"  {stat}: {value:.4f}")

# 使用示例
if __name__ == "__main__":
    env = Envir.PandemicControlEnvironment()
    
    # 假设你已经训练好了agent
    # agent, evaluation_results = train_dqn(env, episodes=10000, max_steps=100)
    
    # 绘制图像
    # plot_optimal_policy_and_q(agent, env)
    # plot_q_value_distribution(agent, env)
    
    print("图像绘制函数已定义，在训练完成后调用即可")