import gym
import ns3gym
from ns3gym import ns3env
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import ttest_rel
from scipy.ndimage import gaussian_filter1d

N_AGENTS = 3
POWER_EXCELLENT_THRESHOLD = None # Will pass in from ns3gym extra info
POWER_POOR_THRESHOLD = None
QOS_EXCELLENT_THRESHOLD = None
QOS_POOR_THRESHOLD = None
PRB_EXCELLENT_THRESHOLD = None
PRB_POOR_THRESHOLD = None

# Energy efficiency calculation parameters
MAX_THROUGHPUT_MBPS = 100  # Maximum achievable throughput in Mbps
REFERENCE_RSRP_DBM = -70   # Reference RSRP for maximum throughput
NOISE_FLOOR_DBM = -110     # Typical noise floor

def smooth_data(data, sigma=2):
    """Apply Gaussian smoothing to the data."""
    return gaussian_filter1d(data, sigma=sigma)

def calculate_energy_efficiency(power_dbm, rsrp_dbm, prb_deviation_percent):
    """
    Calculate energy efficiency based on power consumption, QoS (RSRP), and PRB deviation.
    
    Args:
        power_dbm: Transmission power in dBm
        rsrp_dbm: Received Signal Reference Power in dBm (QoS metric)
        prb_deviation_percent: PRB deviation as percentage
    
    Returns:
        Energy efficiency in Mbps/Watt
    """
    # Convert power from dBm to Watts
    power_watts = 10**((power_dbm - 30) / 10)
    
    # Calculate SNR-based throughput estimation
    # Higher RSRP (less negative) indicates better signal quality
    snr_db = rsrp_dbm - NOISE_FLOOR_DBM
    snr_linear = 10**(snr_db / 10)
    
    # Shannon capacity-based spectral efficiency
    spectral_efficiency = np.log2(1 + snr_linear)
    
    # Estimate throughput based on RSRP quality
    rsrp_quality_factor = (rsrp_dbm - REFERENCE_RSRP_DBM) / 10
    estimated_throughput = MAX_THROUGHPUT_MBPS * 10**(rsrp_quality_factor / 10)
    
    # Apply spectral efficiency scaling
    throughput_with_se = estimated_throughput * (spectral_efficiency / 10)  # Normalize SE
    
    # Apply PRB deviation penalty
    prb_efficiency = 1 - (prb_deviation_percent / 100)
    effective_throughput = throughput_with_se * max(0.1, prb_efficiency)  # Min 10% efficiency
    
    # Ensure positive throughput
    effective_throughput = max(0.1, effective_throughput)
    
    # Calculate energy efficiency (Mbps/Watt)
    energy_efficiency = effective_throughput / max(0.001, power_watts)  # Avoid division by zero
    
    return energy_efficiency

def calculate_composite_energy_efficiency(power_dbm, rsrp_dbm, prb_deviation_percent):
    """
    Alternative energy efficiency calculation using weighted composite score.
    
    Returns:
        Composite energy efficiency score
    """
    # Normalize RSRP (assuming typical range -120 to -60 dBm)
    rsrp_normalized = (rsrp_dbm + 120) / 60  # Maps -120 to 0, -60 to 1
    rsrp_normalized = np.clip(rsrp_normalized, 0, 1)
    
    # Normalize PRB efficiency
    prb_efficiency = 1 - (prb_deviation_percent / 100)
    prb_efficiency = np.clip(prb_efficiency, 0, 1)
    
    # Weighted QoS score (70% RSRP, 30% PRB efficiency)
    qos_composite = 0.7 * rsrp_normalized + 0.3 * prb_efficiency
    
    # Convert power to watts and calculate efficiency
    power_watts = 10**((power_dbm - 30) / 10)
    composite_ee = qos_composite / max(0.001, power_watts)
    
    return composite_ee

def summarize(label, rewards, powers, qos, prb, energy_efficiency):
    print(f"\n--- Summary for {label} ---")
    print(f"Average Reward: {np.mean(rewards):.4f}")
    print(f"Average Power:  {np.mean(powers):.4f} dBm")
    print(f"Average QoS:    {np.mean(qos):.4f} dBm")
    print(f"Average PRB:    {np.mean(prb):.4f}%")
    print(f"Average Energy Efficiency: {np.mean(energy_efficiency):.4f} Mbps/W")

class DDQNAgent:
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

class PPOAgent:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.power_options = [20, 30, 40]
        self.cio_options = [-10, 0, 10]

    def decode_action(self, action_idx):
        power_idx = action_idx // 3
        cio_idx = action_idx % 3
        return self.power_options[power_idx], self.cio_options[cio_idx]

    def act(self, state):
        state_tensor = np.expand_dims(state, axis=0)
        action_probs = self.model.predict(state_tensor, verbose=0)
        action_idx = np.argmax(action_probs[0])
        power, cio = self.decode_action(action_idx)
        return [float(power), float(cio)]

def run_simulation(ddqn_agents=None, ppo_agents=None, fixed_action=None, seed=None, num_steps=100):
    env = ns3env.Ns3Env(port=5555, simSeed=seed)
    state = env.reset()
    total_reward = 0
    cumulative_power = 0
    cumulative_qos = 0
    cumulative_prb = 0
    cumulative_energy_efficiency = 0

    for step in range(num_steps):
        if fixed_action:
            actions = fixed_action * N_AGENTS  # Apply same action to all eNBs
        elif ddqn_agents:
            actions = []
            for agent in ddqn_agents:
                action = agent.act(state)
                actions.extend(action)
        elif ppo_agents:
            actions = []
            for agent in ppo_agents:
                action = agent.act(state)
                actions.extend(action)

        next_state, reward, done, info = env.step(actions)
        total_reward += reward
        parsed = dict(item.split(":") for item in info.split(","))
        
        power_val = float(parsed["Power"])
        qos_val = float(parsed["QoS"])
        prb_val = float(parsed["PRB"])
        
        cumulative_power += power_val
        cumulative_qos += qos_val
        cumulative_prb += prb_val
        
        # Calculate energy efficiency for this step
        step_energy_efficiency = calculate_energy_efficiency(power_val, qos_val, prb_val)
        cumulative_energy_efficiency += step_energy_efficiency
        
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
    avg_energy_efficiency = cumulative_energy_efficiency / num_steps

    return avg_total_reward, avg_power, avg_qos, avg_prb, avg_energy_efficiency

def main():
    # DDQN model paths
    ddqn_model_paths = [
        "saved_models/enb_0_main_model_episode_300",
        "saved_models/enb_1_main_model_episode_300",
        "saved_models/enb_2_main_model_episode_300"
    ]
    
    # PPO model paths
    ppo_model_paths = [
        "saved_models/enb_0_policy_model_episode_900",
        "saved_models/enb_1_policy_model_episode_900", 
        "saved_models/enb_2_policy_model_episode_900"
    ]
    
    ddqn_agents = [DDQNAgent(model_path) for model_path in ddqn_model_paths]
    ppo_agents = [PPOAgent(model_path) for model_path in ppo_model_paths]
    fixed_action = [40, 0]  # power=40, cio=0
    
    num_iterations = 50
    
    # History arrays for DDQN
    avg_rewards_history_ddqn = []
    power_history_ddqn = []
    qos_history_ddqn = []
    prb_history_ddqn = []
    energy_efficiency_history_ddqn = []
    
    # History arrays for PPO
    avg_rewards_history_ppo = []
    power_history_ppo = []
    qos_history_ppo = []
    prb_history_ppo = []
    energy_efficiency_history_ppo = []

    # History arrays for Fixed Action
    avg_rewards_history_fixed = []
    power_history_fixed = []
    qos_history_fixed = []
    prb_history_fixed = []
    energy_efficiency_history_fixed = []

    for iteration in range(num_iterations):
        seed = iteration + 10000  
        
        # Run simulations
        avg_reward_ddqn, avg_power_ddqn, avg_qos_ddqn, avg_prb_ddqn, avg_ee_ddqn = run_simulation(ddqn_agents=ddqn_agents, seed=seed)
        avg_reward_ppo, avg_power_ppo, avg_qos_ppo, avg_prb_ppo, avg_ee_ppo = run_simulation(ppo_agents=ppo_agents, seed=seed)
        avg_reward_fixed, avg_power_fixed, avg_qos_fixed, avg_prb_fixed, avg_ee_fixed = run_simulation(fixed_action=fixed_action, seed=seed)
        
        # Store DDQN results
        avg_rewards_history_ddqn.append(avg_reward_ddqn)
        power_history_ddqn.append(avg_power_ddqn)
        qos_history_ddqn.append(avg_qos_ddqn)
        prb_history_ddqn.append(avg_prb_ddqn)
        energy_efficiency_history_ddqn.append(avg_ee_ddqn)
        
        # Store PPO results
        avg_rewards_history_ppo.append(avg_reward_ppo)
        power_history_ppo.append(avg_power_ppo)
        qos_history_ppo.append(avg_qos_ppo)
        prb_history_ppo.append(avg_prb_ppo)
        energy_efficiency_history_ppo.append(avg_ee_ppo)

        # Store Fixed Action results
        avg_rewards_history_fixed.append(avg_reward_fixed)
        power_history_fixed.append(avg_power_fixed)
        qos_history_fixed.append(avg_qos_fixed)
        prb_history_fixed.append(avg_prb_fixed)
        energy_efficiency_history_fixed.append(avg_ee_fixed)

        # Plot comparison graphs

        # --- Average Reward Comparison ---
        plt.figure(figsize=(12, 6))
        plt.plot(avg_rewards_history_ddqn, color="lightblue", alpha=0.3)
        plt.plot(smooth_data(avg_rewards_history_ddqn), label="DDQN Model", color="blue", linewidth=2)

        plt.plot(avg_rewards_history_ppo, color="lightgreen", alpha=0.3)
        plt.plot(smooth_data(avg_rewards_history_ppo), label="PPO Model", color="green", linewidth=2)

        plt.plot(avg_rewards_history_fixed, color="lightcoral", alpha=0.3)
        plt.plot(smooth_data(avg_rewards_history_fixed), label="Baseline", color="red", linewidth=2)

        plt.xlabel("Iterations")
        plt.ylabel("Average Reward")
        plt.title("Performance Comparison: DDQN vs PPO vs Baseline")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("avg_reward_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # --- Power Consumption Comparison ---
        plt.figure(figsize=(12, 6))
        plt.plot(power_history_ddqn, color="lightblue", alpha=0.3)
        plt.plot(smooth_data(power_history_ddqn), label="DDQN Power Consumption", color="blue", linewidth=2)

        plt.plot(power_history_ppo, color="lightgreen", alpha=0.3)
        plt.plot(smooth_data(power_history_ppo), label="PPO Power Consumption", color="green", linewidth=2)

        plt.plot(power_history_fixed, color="lightcoral", alpha=0.3)
        plt.plot(smooth_data(power_history_fixed), label="Baseline Power Consumption", color="red", linewidth=2)

        if POWER_EXCELLENT_THRESHOLD is not None:
            plt.axhline(y=POWER_EXCELLENT_THRESHOLD, color='darkgreen', linestyle='--', label='Excellent Threshold', alpha=0.7)
            plt.axhline(y=POWER_POOR_THRESHOLD, color='darkred', linestyle='--', label='Poor Threshold', alpha=0.7)
        
        plt.xlabel("Iterations")
        plt.ylabel("Average Power Consumption (dBm)")
        plt.title("Power Consumption Comparison: DDQN vs PPO vs Baseline")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("power_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # --- QoS Comparison ---
        plt.figure(figsize=(12, 6))
        plt.plot(qos_history_ddqn, color="lightblue", alpha=0.3)
        plt.plot(smooth_data(qos_history_ddqn), label="DDQN QoS", color="blue", linewidth=2)

        plt.plot(qos_history_ppo, color="lightgreen", alpha=0.3)
        plt.plot(smooth_data(qos_history_ppo), label="PPO QoS", color="green", linewidth=2)

        plt.plot(qos_history_fixed, color="lightcoral", alpha=0.3)
        plt.plot(smooth_data(qos_history_fixed), label="Baseline QoS", color="red", linewidth=2)

        if QOS_EXCELLENT_THRESHOLD is not None:
            plt.axhline(y=QOS_EXCELLENT_THRESHOLD, color='darkgreen', linestyle='--', label='Excellent Threshold', alpha=0.7)
            plt.axhline(y=QOS_POOR_THRESHOLD, color='darkred', linestyle='--', label='Poor Threshold', alpha=0.7)
        
        plt.xlabel("Iterations")
        plt.ylabel("Average QoS (RSRP)")
        plt.title("QoS Comparison: DDQN vs PPO vs Baseline")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("qos_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # --- PRB Deviation Comparison ---
        plt.figure(figsize=(12, 6))
        plt.plot(prb_history_ddqn, color="lightblue", alpha=0.3)
        plt.plot(smooth_data(prb_history_ddqn), label="DDQN PRB", color="blue", linewidth=2)

        plt.plot(prb_history_ppo, color="lightgreen", alpha=0.3)
        plt.plot(smooth_data(prb_history_ppo), label="PPO PRB", color="green", linewidth=2)

        plt.plot(prb_history_fixed, color="lightcoral", alpha=0.3)
        plt.plot(smooth_data(prb_history_fixed), label="Baseline PRB", color="red", linewidth=2)

        if PRB_EXCELLENT_THRESHOLD is not None:
            plt.axhline(y=PRB_EXCELLENT_THRESHOLD, color='darkgreen', linestyle='--', label='Excellent Threshold', alpha=0.7)
            plt.axhline(y=PRB_POOR_THRESHOLD, color='darkred', linestyle='--', label='Poor Threshold', alpha=0.7)
        
        plt.xlabel("Iterations")
        plt.ylabel("Average PRB Deviation (%)")
        plt.title("PRB Deviation Comparison: DDQN vs PPO vs Baseline")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("prb_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # --- NEW: Energy Efficiency Comparison ---
        plt.figure(figsize=(12, 6))
        plt.plot(energy_efficiency_history_ddqn, color="lightblue", alpha=0.3)
        plt.plot(smooth_data(energy_efficiency_history_ddqn), label="DDQN Energy Efficiency", color="blue", linewidth=2)

        plt.plot(energy_efficiency_history_ppo, color="lightgreen", alpha=0.3)
        plt.plot(smooth_data(energy_efficiency_history_ppo), label="PPO Energy Efficiency", color="green", linewidth=2)

        plt.plot(energy_efficiency_history_fixed, color="lightcoral", alpha=0.3)
        plt.plot(smooth_data(energy_efficiency_history_fixed), label="Baseline Energy Efficiency", color="red", linewidth=2)
        
        plt.xlabel("Iterations")
        plt.ylabel("Energy Efficiency (Mbps/Watt)")
        plt.title("Energy Efficiency Comparison: DDQN vs PPO vs Baseline")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("energy_efficiency_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Print progress
        if (iteration + 1) % 10 == 0:
            print(f"Completed {iteration + 1}/{num_iterations} iterations")

    # Print final summaries
    summarize("DDQN Model", avg_rewards_history_ddqn, power_history_ddqn, qos_history_ddqn, prb_history_ddqn, energy_efficiency_history_ddqn)
    summarize("PPO Model", avg_rewards_history_ppo, power_history_ppo, qos_history_ppo, prb_history_ppo, energy_efficiency_history_ppo)
    summarize("Baseline", avg_rewards_history_fixed, power_history_fixed, qos_history_fixed, prb_history_fixed, energy_efficiency_history_fixed)
    
    # Create a comprehensive comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Reward plot
    axes[0,0].plot(smooth_data(avg_rewards_history_ddqn), label="DDQN", color="blue", linewidth=2)
    axes[0,0].plot(smooth_data(avg_rewards_history_ppo), label="PPO", color="green", linewidth=2)
    axes[0,0].plot(smooth_data(avg_rewards_history_fixed), label="Fixed", color="red", linewidth=2)
    axes[0,0].set_title("Average Reward")
    axes[0,0].set_ylabel("Reward")
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Power plot
    axes[0,1].plot(smooth_data(power_history_ddqn), label="DDQN", color="blue", linewidth=2)
    axes[0,1].plot(smooth_data(power_history_ppo), label="PPO", color="green", linewidth=2)
    axes[0,1].plot(smooth_data(power_history_fixed), label="Baseline", color="red", linewidth=2)
    axes[0,1].set_title("Power Consumption")
    axes[0,1].set_ylabel("Power (dBm)")
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # QoS plot
    axes[0,2].plot(smooth_data(qos_history_ddqn), label="DDQN", color="blue", linewidth=2)
    axes[0,2].plot(smooth_data(qos_history_ppo), label="PPO", color="green", linewidth=2)
    axes[0,2].plot(smooth_data(qos_history_fixed), label="Baseline", color="red", linewidth=2)
    axes[0,2].set_title("QoS (RSRP)")
    axes[0,2].set_ylabel("RSRP (dBm)")
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # PRB plot
    axes[1,0].plot(smooth_data(prb_history_ddqn), label="DDQN", color="blue", linewidth=2)
    axes[1,0].plot(smooth_data(prb_history_ppo), label="PPO", color="green", linewidth=2)
    axes[1,0].plot(smooth_data(prb_history_fixed), label="Baseline", color="red", linewidth=2)
    axes[1,0].set_title("PRB Deviation")
    axes[1,0].set_ylabel("PRB Deviation (%)")
    axes[1,0].set_xlabel("Iterations")
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Energy Efficiency plot
    axes[1,1].plot(smooth_data(energy_efficiency_history_ddqn), label="DDQN", color="blue", linewidth=2)
    axes[1,1].plot(smooth_data(energy_efficiency_history_ppo), label="PPO", color="green", linewidth=2)
    axes[1,1].plot(smooth_data(energy_efficiency_history_fixed), label="Baseline", color="red", linewidth=2)
    axes[1,1].set_title("Energy Efficiency")
    axes[1,1].set_ylabel("Energy Efficiency (Mbps/W)")
    axes[1,1].set_xlabel("Iterations")
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Performance summary bar chart
    algorithms = ['DDQN', 'PPO', 'Baseline']
    ee_means = [np.mean(energy_efficiency_history_ddqn), 
                np.mean(energy_efficiency_history_ppo), 
                np.mean(energy_efficiency_history_fixed)]
    ee_stds = [np.std(energy_efficiency_history_ddqn), 
               np.std(energy_efficiency_history_ppo), 
               np.std(energy_efficiency_history_fixed)]
    
    axes[1,2].bar(algorithms, ee_means, yerr=ee_stds, capsize=5, 
                  color=['blue', 'green', 'red'], alpha=0.7)
    axes[1,2].set_title("Energy Efficiency Summary")
    axes[1,2].set_ylabel("Energy Efficiency (Mbps/W)")
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("comprehensive_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nAll graphs saved successfully!")
    print("- avg_reward_comparison.png")
    print("- power_comparison.png") 
    print("- qos_comparison.png")
    print("- prb_comparison.png")
    print("- energy_efficiency_comparison.png")
    print("- comprehensive_comparison.png")

if __name__ == "__main__":
    main()