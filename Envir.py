import numpy as np
import matplotlib.pyplot as plt
import random

class PandemicControlEnvironment:
    """
    Pandemic Control Environment with Regime Switching
    A stochastic optimal control problem for infectious disease management
    """
    
    def __init__(self):
        # Environment parameters
        self.h = 0.05  # State discretization step
        self.delta = 0.1  # Discount rate
        
        # State and action spaces
        self.x_states = np.arange(self.h, 1.0, self.h)  # Infection rate states
        self.n_x = len(self.x_states)
        self.regimes = [0, 1]  # Two epidemiological regimes
        
        # System parameters for each regime
        self.alpha = [0.2, 0.8]   # Transmission rates
        self.beta = [0.2, 0.8]    # Secondary transmission effects
        self.gamma = [0.1, 0.2]   # Recovery rates  
        self.sigma_tilde = [0.15, 0.35]  # Volatility coefficients
        self.q = np.array([[-1, 1], [1, -1]])  # Regime switching rates
        
        # Cost function parameters
        self.a0, self.aI, self.aS_m, self.aI_m, self.a_r = 0.0, 1.5, 0.5, 1.5, 1.0
        
        # Action space: combinations of control measures
        self.actions = [(eta, rho) for eta in np.linspace(0.0, 1.0, 5)
                                  for rho in np.linspace(0.0, 1.0, 5)]
        self.n_actions = len(self.actions)
        
        # Current state
        self.reset()
    
    def reset(self):
        """
        Reset environment to random initial state
        Returns: state vector [x, regime]
        """
        self.x = np.random.choice(self.x_states)
        self.l = np.random.choice(self.regimes)
        self.step_count = 0
        return self._get_state()
    
    def _get_state(self):
        """Get current state as numpy array"""
        return np.array([self.x, float(self.l)], dtype=np.float32)
    
    def b(self, x, l, eta, rho):
        """Drift term of the stochastic differential equation"""
        return eta * self.alpha[l] * (1 - x) + x * (eta**2 * self.beta[l] * (1 - x) - (self.gamma[l] + rho))
    
    def sigma(self, x, l):
        """Volatility term of the stochastic differential equation"""
        return self.sigma_tilde[l] * x * (1 - x)
    
    def f(self, x, eta, rho):
        """Instantaneous cost function"""
        return (self.a0 + self.aI * x + 
                self.aS_m * (1 - eta)**2 + 
                (self.aI_m - self.aS_m) * x * (1 - eta)**2 + 
                self.a_r * x * rho**2)
    
    def compute_Qh(self, x, l, eta, rho):
        """Compute transition probability denominator"""
        drift = self.b(x, l, eta, rho)
        vol = self.sigma(x, l)
        q_ii = self.q[l, l]
        Qh = vol**2 + abs(drift) * self.h - self.h**2 * q_ii + self.h
        return max(Qh, 1e-12), drift, vol
    
    def trans_prob(self, x, l, eta, rho):
        """Compute transition probabilities for Markov chain approximation"""
        Qh, drift, vol = self.compute_Qh(x, l, eta, rho)
        p_up = (vol**2 / 2 + max(drift, 0) * self.h) / Qh
        p_down = (vol**2 / 2 - min(drift, 0) * self.h) / Qh
        p_stay = self.h / Qh
        p_switch = (self.h**2 * self.q[l, 1-l]) / Qh
        return {"p_up": p_up, "p_down": p_down, "p_stay": p_stay, "p_switch": p_switch, "Qh": Qh}
    
    def sample_next_state(self, x, l, eta, rho):
        """Sample next state according to transition probabilities"""
        p = self.trans_prob(x, l, eta, rho)
        r = random.random()

        p_down, p_stay, p_up, p_switch = p["p_down"], p["p_stay"], p["p_up"], p["p_switch"]
        
        # Sample next state based on probabilities
        if r < p_down:
            x_next = max(self.h, x - self.h)  # Ensure lower bound
            l_next = l
        elif r < p_down + p_stay:
            x_next = x
            l_next = l
        elif r < p_down + p_stay + p_up:
            x_next = min(1.0 - self.h, x + self.h)  # Ensure upper bound
            l_next = l
        else:  # regime switch
            x_next = x
            l_next = 1 - l
        
        return x_next, l_next, p["Qh"]
    
    def step(self, action_idx):
        """
        Execute one environment step
        Args:
            action_idx: Index of action to take
            max_steps: Maximum steps before episode termination
        Returns:
            next_state: New state vector [x, regime]
            reward: Immediate reward (negative cost)
            done: Whether episode is finished
            info: Additional information dictionary
        """
        # Get action parameters
        eta, rho = self.actions[action_idx]
        
        # Sample next state
        x_next, l_next, Qh_val = self.sample_next_state(self.x, self.l, eta, rho)
        
        # Calculate time step and cost
        Delta_t = self.h**2 / Qh_val
        discount_factor = np.exp(-self.delta * Delta_t)
        reward = self.f(self.x, eta, rho) * Delta_t
        
        # Update state
        self.x = x_next
        self.l = l_next
        self.step_count += 1
                
        # Additional info for analysis
        info = {
            'x': self.x,
            'regime': self.l,
            'eta': eta,
            'rho': rho,
            'cost': reward,
            'Delta_t': Delta_t,
            'step': self.step_count
        }
        
        return self._get_state(), reward, discount_factor, info
    
    def get_state_info(self):
        """Get current state information"""
        return {
            'x': self.x,
            'regime': self.l,
            'step': self.step_count
        }
    
    def get_discrete_state_index(self, x):
        """Get discrete index for continuous x value"""
        return np.argmin(np.abs(self.x_states - x))

def calculate_averaged_total_cost(env, num_episodes=1000, max_steps_per_episode=1000):
    episode_costs = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_cost = 0.0
        step_count = 0
        current_discount = 1.0
        
        while step_count < max_steps_per_episode:
            # 随机选择动作（在实际应用中，这里应该使用学习到的策略）
            action_idx = random.randint(0, env.n_actions - 1)
            # 执行一步
            next_state, reward, discount, info = env.step(action_idx)
            
            # 累加成本（注意：reward是负的成本）
            current_discount *= discount
            total_cost += current_discount * reward
            
            step_count += 1
            
            # 可选：如果感染率达到边界，提前终止
            if env.x >= 0.95 or env.x <= 0.05:
                break
        
        episode_costs.append(total_cost)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Total Cost: {total_cost:.4f}")
    
    # 计算平均总成本
    averaged_total_cost = np.mean(episode_costs)
    
    return averaged_total_cost, episode_costs

# 使用示例
if __name__ == "__main__":
    # 创建环境
    env = PandemicControlEnvironment()
    
    print("计算平均总成本...")
    averaged_cost, all_costs = calculate_averaged_total_cost(
        env, 
        num_episodes=500, 
        max_steps_per_episode = 500
    )
    
    print(f"\n平均总成本: {averaged_cost:.4f}")
    print(f"成本标准差: {np.std(all_costs):.4f}")
    print(f"最小成本: {np.min(all_costs):.4f}")
    print(f"最大成本: {np.max(all_costs):.4f}")