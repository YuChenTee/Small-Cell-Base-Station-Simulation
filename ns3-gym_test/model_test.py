import gym
import ns3gym
from ns3gym import ns3env
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

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

def main():
    # Define thresholds for scaling
    rsrp_poor = -60.0
    rsrp_excellent = -40.0
    prb_poor = 30
    prb_excellent = 10

    # Define paths to saved models
    model_paths = [
        "saved_models/enb_0_main_model_episode_899",
        "saved_models/enb_1_main_model_episode_899",
        "saved_models/enb_2_main_model_episode_899"
    ]

    # Create agents for each eNB
    agents = [SingleAgent(model_path) for model_path in model_paths]

    try:
        # Initialize NS-3 environment
        env = ns3env.Ns3Env(port=5555)
        state = env.reset()
        num_steps = 100
        
        # Initialize tracking lists
        total_reward = 0
        reward_list = []

        for step in range(num_steps):
            # Get actions from all agents
            actions = []
            for agent in agents:
                action = agent.act(state)
                actions.extend(action)

            # Take action in the environment
            next_state, reward, done, info = env.step(actions)
            total_reward += reward
            reward_list.append(reward)
            state = next_state

            if done:
                print("Simulation completed before reaching maximum steps.")
                break

        print(f"Total Reward: {total_reward}")
        print(f"Avg. Reward: {total_reward / num_steps}")

        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.plot(reward_list, label="Reward", color="red")
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.title("Reward Over Time")
        plt.legend()
        plt.grid()
        image_path = "rsrp_prb_reward_plot_DDQN.png"
        plt.savefig(image_path)
        print(f"Graph saved as {image_path}")

    except Exception as e:
        print(f"Error during simulation: {e}")

    finally:
        try:
            env.close()
        except:
            pass
        print("Simulation terminated.")

if __name__ == "__main__":
    main()