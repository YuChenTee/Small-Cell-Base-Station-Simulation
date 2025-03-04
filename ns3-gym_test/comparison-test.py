import gym
import ns3gym
from ns3gym import ns3env
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import ttest_rel

N_AGENTS = 3

class SingleAgent:
    def __init__(self, model_path, action_size=9):
        self.model = load_model(model_path)
        self.action_size = action_size
        self.power_options = [20, 30, 40]
        self.cio_options = [-10, 0, 10]

    def decode_action(self, action_idx):
        power_idx = action_idx // 3
        cio_idx = action_idx % 3
        return self.power_options[power_idx], self.cio_options[cio_idx]

    def act(self, state):
        state_tensor = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state_tensor, verbose=0)
        action_idx = np.argmax(q_values[0])
        power, cio = self.decode_action(action_idx)
        return [float(power), float(cio)]

def run_simulation(agents=None, fixed_action=None, seed=None, num_steps=100):
    env = ns3env.Ns3Env(port=5555, simSeed=seed)
    state = env.reset()
    total_reward = 0

    for step in range(num_steps):
        if fixed_action:
            actions = fixed_action * N_AGENTS  # Apply same action to all eNBs
        else:
            actions = []
            for agent in agents:
                action = agent.act(state)
                actions.extend(action)

        next_state, reward, done, info = env.step(actions)
        total_reward += reward
        state = next_state

        if done:
            break

    env.close()
    return total_reward / num_steps  # Return average reward per step

def main():
    model_paths = [
        "saved_models/enb_0_main_model_episode_899",
        "saved_models/enb_1_main_model_episode_899",
        "saved_models/enb_2_main_model_episode_899"
    ]
    
    agents = [SingleAgent(model_path) for model_path in model_paths]
    fixed_action = [30, 0]  # power=30, cio=0
    
    num_iterations = 100
    avg_rewards_model = []
    avg_rewards_fixed = []

    for iteration in range(num_iterations):
        seed = iteration + 10000  
        
        avg_reward_model = run_simulation(agents=agents, seed=seed)
        avg_reward_fixed = run_simulation(fixed_action=fixed_action, seed=seed)
        
        avg_rewards_model.append(avg_reward_model)
        avg_rewards_fixed.append(avg_reward_fixed)
        
        print(f"Iteration {iteration + 1}: Model Avg Reward = {avg_reward_model}, Fixed Avg Reward = {avg_reward_fixed}")

        # Plot comparison graph
        plt.figure(figsize=(10, 6))
        plt.plot(avg_rewards_model, label="Trained Model", color="blue")
        plt.plot(avg_rewards_fixed, label="Fixed Action (Power=30, CIO=0)", color="green")
        plt.xlabel("Iterations")
        plt.ylabel("Average Reward")
        plt.title("Comparison of Trained Model vs Fixed Action")
        plt.legend()
        plt.grid()
        image_path = "avg_reward_comparison.png"
        plt.savefig(image_path)
        print(f"Graph saved as {image_path}")

    # Perform paired t-test
    t_stat, p_value = ttest_rel(avg_rewards_model, avg_rewards_fixed)

    print("\nPaired t-test Results:")
    print(f"T-Statistic: {t_stat:.4f}")
    print(f"P-Value: {p_value:.4f}")

    if p_value < 0.05:
        print("Conclusion: The trained model performs significantly better than the fixed action strategy (p < 0.05).")
    else:
        print("Conclusion: There is no significant difference between the trained model and the fixed action strategy.")

if __name__ == "__main__":
    main()
