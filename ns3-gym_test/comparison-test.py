import gym
import ns3gym
from ns3gym import ns3env
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import ttest_rel

N_AGENTS = 3
POWER_EXCELLENT_THRESHOLD = None # Will pass in from ns3gym extra info
POWER_POOR_THRESHOLD = None
QOS_EXCELLENT_THRESHOLD = None
QOS_POOR_THRESHOLD = None
PRB_EXCELLENT_THRESHOLD = None
PRB_POOR_THRESHOLD = None

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
    cumulative_power = 0
    cumulative_qos = 0
    cumulative_prb = 0

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
        parsed = dict(item.split(":") for item in info.split(","))
        cumulative_power += float(parsed["Power"])
        cumulative_qos += float(parsed["QoS"])
        cumulative_prb += float(parsed["PRB"])
        state = next_state

        global POWER_EXCELLENT_THRESHOLD, POWER_POOR_THRESHOLD, QOS_EXCELLENT_THRESHOLD, QOS_POOR_THRESHOLD, PRB_EXCELLENT_THRESHOLD, PRB_POOR_THRESHOLD
        if POWER_EXCELLENT_THRESHOLD is None:
            POWER_EXCELLENT_THRESHOLD = float(parsed["Power_excellent_threshold"])
            POWER_POOR_THRESHOLD = float(parsed["Power_poor_threshold"])
            QOS_EXCELLENT_THRESHOLD = float(parsed["QoS_excellent_threshold"])
            QOS_POOR_THRESHOLD = float(parsed["QoS_poor_threshold"])
            PRB_EXCELLENT_THRESHOLD = float(parsed["PRB_excellent_threshold"])
            PRB_POOR_THRESHOLD = float(parsed["PRB_poor_threshold"])

        if done:
            break

    env.close()

    avg_total_reward = total_reward / num_steps
    avg_power = cumulative_power / num_steps
    avg_qos = cumulative_qos / num_steps
    avg_prb = cumulative_prb / num_steps

    return avg_total_reward, avg_power, avg_qos, avg_prb

def main():
    model_paths = [
        "saved_models/enb_0_main_model_episode_899",
        "saved_models/enb_1_main_model_episode_899",
        "saved_models/enb_2_main_model_episode_899"
    ]
    
    agents = [SingleAgent(model_path) for model_path in model_paths]
    fixed_action_low = [20, -10] # power=20, cio=-10
    fixed_action_mid = [30, 0]  
    fixed_action_high = [40, 10]
    
    num_iterations = 50
    avg_rewards_history_model = []
    avg_rewards_history_low = []
    avg_rewards_history_mid = []
    avg_rewards_history_high = []

    power_history_model = []
    qos_history_model = []
    prb_history_model = []

    power_history_low = []
    qos_history_low = []
    prb_history_low = []

    power_history_mid = []
    qos_history_mid = []
    prb_history_mid = []

    power_history_high = []
    qos_history_high = []
    prb_history_high = []

    for iteration in range(num_iterations):
        seed = iteration + 10000  
        
        avg_reward_model, avg_power_model, avg_qos_model, avg_prb_model = run_simulation(agents=agents, seed=seed)
        avg_reward_low, avg_power_low, avg_qos_low, avg_prb_low = run_simulation(fixed_action=fixed_action_low, seed=seed)
        avg_reward_mid, avg_power_mid, avg_qos_mid, avg_prb_mid = run_simulation(fixed_action=fixed_action_mid, seed=seed)
        avg_reward_high, avg_power_high, avg_qos_high, avg_prb_high = run_simulation(fixed_action=fixed_action_high, seed=seed)
        
        avg_rewards_history_model.append(avg_reward_model)
        power_history_model.append(avg_power_model)
        qos_history_model.append(avg_qos_model)
        prb_history_model.append(avg_prb_model)

        avg_rewards_history_low.append(avg_reward_low)
        power_history_low.append(avg_power_low)
        qos_history_low.append(avg_qos_low)
        prb_history_low.append(avg_prb_low)

        avg_rewards_history_mid.append(avg_reward_mid)
        power_history_mid.append(avg_power_mid)
        qos_history_mid.append(avg_qos_mid)
        prb_history_mid.append(avg_prb_mid)

        avg_rewards_history_high.append(avg_reward_high)
        power_history_high.append(avg_power_high)
        qos_history_high.append(avg_qos_high)
        prb_history_high.append(avg_prb_high)

        # Plot comparison graph
        plt.figure(figsize=(10, 6))
        plt.plot(avg_rewards_history_model, label="Trained Model", color="blue")
        plt.plot(avg_rewards_history_low, label="Fixed Action (Power=20, CIO=-10)", color="yellow")
        plt.plot(avg_rewards_history_mid, label="Fixed Action (Power=30, CIO=0)", color="orange")
        plt.plot(avg_rewards_history_high, label="Fixed Action (Power=40, CIO=10)", color="purple")
        plt.xlabel("Iterations")
        plt.ylabel("Average Reward")
        plt.title("Comparison of Trained Model vs Fixed Action")
        plt.legend()
        plt.grid()
        image_path = "avg_reward_comparison.png"
        plt.savefig(image_path)
        print(f"Graph saved as {image_path}")

        # Plot power comparison graph
        plt.figure(figsize=(10, 6))
        plt.plot(power_history_model, label="Trained Model Power Consumption", color="blue")
        plt.plot(power_history_low, label="Fixed Action Power Consumption (Power=20, CIO=-10)", color="yellow")
        plt.plot(power_history_mid, label="Fixed Action Power Consumption (Power=30, CIO=0)", color="orange")
        plt.plot(power_history_high, label="Fixed Action Power Consumption (Power=40, CIO=10)", color="purple")
        plt.axhline(y=POWER_EXCELLENT_THRESHOLD, color='green', linestyle='--', label='Excellent Threshold')
        plt.axhline(y=POWER_POOR_THRESHOLD, color='red', linestyle='--', label='Poor Threshold')
        plt.xlabel("Iterations")
        plt.ylabel("Average Power Consumption (dBm)")
        plt.title("Comparison of Power Consumption: Trained Model vs Fixed Action")
        plt.legend()
        plt.grid()
        image_path = "power_comparison.png"
        plt.savefig(image_path)
        print(f"Graph saved as {image_path}")

        # Plot QoS comparison graph
        plt.figure(figsize=(10, 6))
        plt.plot(qos_history_model, label="Trained Model QoS", color="blue")
        plt.plot(qos_history_low, label="Fixed Action QoS (Power=20, CIO=-10)", color="yellow")
        plt.plot(qos_history_mid, label="Fixed Action QoS (Power=30, CIO=0)", color="orange")
        plt.plot(qos_history_high, label="Fixed Action QoS (Power=40, CIO=10)", color="purple")
        plt.xlabel("Iterations")
        plt.ylabel("Average QoS (RSRP)")
        plt.axhline(y=QOS_EXCELLENT_THRESHOLD, color='green', linestyle='--', label='Excellent Threshold')
        plt.axhline(y=QOS_POOR_THRESHOLD, color='red', linestyle='--', label='Poor Threshold')
        plt.title("Comparison of QoS: Trained Model vs Fixed Action")
        plt.legend()
        plt.grid()
        image_path = "qos_comparison.png"
        plt.savefig(image_path)
        print(f"Graph saved as {image_path}")

        # Plot PRB comparison graph
        plt.figure(figsize=(10, 6))
        plt.plot(prb_history_model, label="Trained Model PRB", color="blue")
        plt.plot(prb_history_low, label="Fixed Action PRB (Power=20, CIO=-10)", color="yellow")
        plt.plot(prb_history_mid, label="Fixed Action PRB (Power=30, CIO=0)", color="orange")
        plt.plot(prb_history_high, label="Fixed Action PRB (Power=40, CIO=10)", color="purple")
        plt.axhline(y=PRB_EXCELLENT_THRESHOLD, color='green', linestyle='--', label='Excellent Threshold')
        plt.axhline(y=PRB_POOR_THRESHOLD, color='red', linestyle='--', label='Poor Threshold')
        plt.xlabel("Iterations")
        plt.ylabel("Average PRB Deviation (%)")
        plt.title("Comparison of PRB Deviation: Trained Model vs Fixed Action")
        plt.legend()
        plt.grid()
        image_path = "prb_comparison.png"
        plt.savefig(image_path)
        print(f"Graph saved as {image_path}")

if __name__ == "__main__":
    main()
