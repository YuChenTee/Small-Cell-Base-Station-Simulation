import gym
import ns3gym
from ns3gym import ns3env
import numpy as np
import matplotlib.pyplot as plt

constant_actions = [30, 0, 30, 0, 30, 0]

def main():
    try:
        # Initialize NS-3 environment
        env = ns3env.Ns3Env(port=5555)
        num_steps = 100
        
        # Initialize tracking lists
        total_reward = 0
        reward_list = []

        for step in range(num_steps):
            # Get actions from all agents
            actions = constant_actions

            # Take action in the environment
            next_state, reward, done, info = env.step(actions)
            total_reward += reward
            reward_list.append(reward)
            

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
        image_path = "rsrp_prb_reward_plot_without_DDQN.png"
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