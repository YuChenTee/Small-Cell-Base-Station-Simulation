import gym
import ns3gym
from ns3gym import ns3env
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
import gc
from collections import deque
import random
import time
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def smooth_data(data, sigma=2):
    """Apply Gaussian smoothing to the data."""
    return gaussian_filter1d(data, sigma=sigma)

class SinglePPOAgent:
    def __init__(self, state_size):
        self.state_size = state_size
        self.action_size = 9
        self.power_options = [20, 30, 40]
        self.cio_options = [-10, 0, 10]
        
        # PPO Hyperparameters
        self.gamma = 0.99
        self.learning_rate = 1e-4
        self.clip_ratio = 0.2
        self.policy_epochs = 4
        self.value_epochs = 80
        self.target_kl = 0.01
        self.entropy_coef = 0.01
        self.entropy_decay = 0.995
        self.value_coef = 0.5
        self.max_grad_norm = 0.3
        self.batch_size = 256
        self.buffer_size = 2048
        
        # Build networks
        self.policy_network = self._build_policy_network()
        self.value_network = self._build_value_network()
        
        # Optimizers
        self.policy_optimizer = Adam(learning_rate=self.learning_rate)
        self.value_optimizer = Adam(learning_rate=self.learning_rate)
        
        # Experience buffer
        self.clear_buffer()
        
        # Training metrics
        self.policy_loss_history = []
        self.value_loss_history = []
        self.entropy_history = []
        self.kl_divergence_history = []

    def _build_policy_network(self):
        """Build policy network that outputs action probabilities."""
        inputs = Input(shape=(self.state_size,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(self.action_size, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def _build_value_network(self):
        """Build value network that estimates state values."""
        inputs = Input(shape=(self.state_size,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(1, activation='linear')(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def clear_buffer(self):
        """Clear the experience buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def decode_action(self, action_idx):
        power_idx = action_idx // 3
        cio_idx = action_idx % 3
        return self.power_options[power_idx], self.cio_options[cio_idx]

    def act(self, state):
        """Select action using the policy network."""
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        
        # Get action probabilities and state value
        action_probs = self.policy_network(state_tensor)[0]
        value = self.value_network(state_tensor)[0][0]
        
        # Sample action from the probability distribution
        action_idx = tf.random.categorical(tf.math.log(action_probs.numpy().reshape(1, -1)), 1)[0][0]
        action_idx = int(action_idx)
        
        # Calculate log probability of the selected action
        log_prob = tf.math.log(action_probs[action_idx])
        
        power, cio = self.decode_action(action_idx)
        return [float(power), float(cio)], action_idx, float(log_prob), float(value)

    def store_transition(self, state, action_idx, reward, log_prob, value, done):
        """Store a transition in the buffer."""
        self.states.append(state)
        self.actions.append(action_idx)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, next_value=0):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        returns = []
        gae = 0
        
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[i]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - self.dones[i]
                next_val = self.values[i + 1]
            
            delta = self.rewards[i] + self.gamma * next_val * next_non_terminal - self.values[i]
            gae = delta + self.gamma * 0.95 * next_non_terminal * gae  # lambda = 0.95
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[i])
        
        return np.array(advantages), np.array(returns)

    @tf.function
    def train_policy(self, states, actions, old_log_probs, advantages):
        """Train the policy network using PPO loss."""
        with tf.GradientTape() as tape:
            # Get current action probabilities
            action_probs = self.policy_network(states)
            
            # Calculate log probabilities for the taken actions
            indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
            log_probs = tf.math.log(tf.gather_nd(action_probs, indices))
            
            # Calculate ratio for PPO
            ratio = tf.exp(log_probs - old_log_probs)
            
            # Calculate surrogate losses
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            
            # Policy loss (negative because we want to maximize)
            policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            
            # Entropy bonus for exploration
            entropy = -tf.reduce_mean(tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=1))
            
            # Total loss
            total_loss = policy_loss - self.entropy_coef * entropy
        
        # Calculate KL divergence for early stopping
        kl_div = tf.reduce_mean(old_log_probs - log_probs)
        
        # Apply gradients
        gradients = tape.gradient(total_loss, self.policy_network.trainable_variables)
        gradients = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in gradients]
        self.policy_optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))
        
        return policy_loss, entropy, kl_div

    @tf.function
    def train_value(self, states, returns):
        """Train the value network."""
        with tf.GradientTape() as tape:
            values = tf.squeeze(self.value_network(states))
            value_loss = tf.reduce_mean(tf.square(returns - values))
        
        gradients = tape.gradient(value_loss, self.value_network.trainable_variables)
        gradients = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in gradients]
        self.value_optimizer.apply_gradients(zip(gradients, self.value_network.trainable_variables))
        
        return value_loss

    def update(self, next_value=0):
        """Update the policy and value networks using collected experiences."""
        if len(self.states) < self.batch_size:
            return None, None, None, None
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Convert to tensors
        states = tf.convert_to_tensor(np.array(self.states), dtype=tf.float32)
        actions = tf.convert_to_tensor(np.array(self.actions), dtype=tf.int32)
        old_log_probs = tf.convert_to_tensor(np.array(self.log_probs), dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        
        # Train policy
        for _ in range(self.policy_epochs):
            policy_loss, entropy, kl_div = self.train_policy(states, actions, old_log_probs, advantages)
            
            # Early stopping if KL divergence is too high
            if kl_div > self.target_kl:
                break
        
        # Train value function
        for _ in range(self.value_epochs):
            value_loss = self.train_value(states, returns)
        
        # Store metrics
        self.policy_loss_history.append(float(policy_loss))
        self.value_loss_history.append(float(value_loss))
        self.entropy_history.append(float(entropy))
        self.kl_divergence_history.append(float(kl_div))

        self.entropy_coef *= self.entropy_decay
        
        # Clear buffer
        self.clear_buffer()
        
        return float(policy_loss), float(value_loss), float(entropy), float(kl_div)


class IndependentPPOAgent:
    def __init__(self, state_size, num_enbs):
        self.num_enbs = num_enbs
        self.agents = [SinglePPOAgent(state_size) for _ in range(num_enbs)]
        self.policy_loss_history = []
        self.value_loss_history = []
        self.entropy_history = []
        self.kl_divergence_history = []

    def act(self, state):
        """Get actions from all agents."""
        actions = []
        self.current_action_indices = []
        self.current_log_probs = []
        self.current_values = []
        
        for agent in self.agents:
            action, action_idx, log_prob, value = agent.act(state)
            actions.extend([float(action[0]), float(action[1])])
            self.current_action_indices.append(action_idx)
            self.current_log_probs.append(log_prob)
            self.current_values.append(value)
        
        return actions

    def store_transition(self, state, action, reward, done):
        """Store transitions for all agents."""
        for i, agent in enumerate(self.agents):
            agent.store_transition(
                state, 
                self.current_action_indices[i], 
                reward, 
                self.current_log_probs[i], 
                self.current_values[i], 
                done
            )

    def update(self, next_state=None):
        """Update all agents."""
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl_div = 0
        valid_updates = 0
        
        for agent in self.agents:
            # Get next value for GAE computation
            next_value = 0
            if next_state is not None:
                next_state_tensor = tf.convert_to_tensor(next_state.reshape(1, -1), dtype=tf.float32)
                next_value = float(agent.value_network(next_state_tensor)[0][0])
            
            policy_loss, value_loss, entropy, kl_div = agent.update(next_value)
            
            if policy_loss is not None:
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_entropy += entropy
                total_kl_div += kl_div
                valid_updates += 1
        
        if valid_updates > 0:
            avg_policy_loss = total_policy_loss / valid_updates
            avg_value_loss = total_value_loss / valid_updates
            avg_entropy = total_entropy / valid_updates
            avg_kl_div = total_kl_div / valid_updates
            
            self.policy_loss_history.append(avg_policy_loss)
            self.value_loss_history.append(avg_value_loss)
            self.entropy_history.append(avg_entropy)
            self.kl_divergence_history.append(avg_kl_div)
            
            return avg_policy_loss, avg_value_loss, avg_entropy, avg_kl_div
        
        return None, None, None, None


def save_models(agent, episode_num, save_dir='saved_models'):
    """Save models for each eNB agent."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    for i, single_agent in enumerate(agent.agents):
        # Save policy network
        policy_model_path = f'{save_dir}/enb_{i}_policy_model_episode_{episode_num}'
        single_agent.policy_network.save(policy_model_path)
        
        # Save value network
        value_model_path = f'{save_dir}/enb_{i}_value_model_episode_{episode_num}'
        single_agent.value_network.save(value_model_path)


def main():
    try:
        env = ns3env.Ns3Env(port=5555, simSeed=random.randint(1, 10000))
        state = env.reset()
        state_size = len(state)
        num_enbs = 3
        agent = IndependentPPOAgent(state_size, num_enbs)
        
        n_episodes = 900
        max_steps = 100
        
        # Tracking variables
        reward_history = []
        power_history = []
        qos_history = []
        prb_history = []

        for episode in range(n_episodes):
            env.simSeed = random.randint(1, 10000)
            state = env.reset()
            total_reward = 0
            actions_taken = []
            
            # Cumulative sums for metrics in each episode
            cumulative_power = 0
            cumulative_qos = 0
            cumulative_prb = 0
            num_steps = 0

            for step in range(max_steps):
                action = agent.act(state)
                actions_taken.append(action)
                
                next_state, reward, done, info = env.step(action)
                parsed = dict(item.split(":") for item in info.split(","))
                
                # Accumulate power, QoS, and PRB values
                cumulative_power += float(parsed["Power(Scaled)"])
                cumulative_qos += float(parsed["QoS(Scaled)"])
                cumulative_prb += float(parsed["PRB(Scaled)"])
                num_steps += 1
                
                # Store transition
                agent.store_transition(state, action, reward, done)
                
                total_reward += reward
                state = next_state
                
                if done:
                    break

            # Update networks at the end of each episode
            policy_loss, value_loss, entropy, kl_div = agent.update(next_state if not done else None)

            # Compute averages over the episode
            avg_power = cumulative_power / num_steps if num_steps > 0 else 0
            avg_qos = cumulative_qos / num_steps if num_steps > 0 else 0
            avg_prb = cumulative_prb / num_steps if num_steps > 0 else 0

            # Append averages to history
            power_history.append(avg_power)
            qos_history.append(avg_qos)
            prb_history.append(avg_prb)
            reward_history.append(total_reward)

            print(f"Episode: {episode + 1}/{n_episodes}")
            print(f"Total Reward: {total_reward:.2f}")
            if policy_loss is not None:
                print(f"Policy Loss: {policy_loss:.4f}")
                print(f"Value Loss: {value_loss:.4f}")
                print(f"Entropy: {entropy:.4f}")
                print(f"KL Divergence: {kl_div:.4f}")
            print("------------------------")

            # Plotting every episode
            # Plotting reward trend
            plt.figure(figsize=(10, 6))
            plt.plot(reward_history, "lightgray", alpha=0.3)
            smoothed_reward = smooth_data(reward_history)
            plt.plot(smoothed_reward, label="Smoothed Total Reward")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.title("Reward Trend Over Episodes")
            plt.legend()
            plt.grid()
            plt.savefig("reward_trend_ppo.png")
            plt.close()

            # Plot loss trends
            if agent.policy_loss_history:
                plt.figure(figsize=(15, 10))
                
                plt.subplot(2, 2, 1)
                plt.plot(agent.policy_loss_history, label="Policy Loss")
                plt.xlabel("Update Step")
                plt.ylabel("Policy Loss")
                plt.title("Policy Loss Over Training")
                plt.legend()
                plt.grid()
                
                plt.subplot(2, 2, 2)
                plt.plot(agent.value_loss_history, label="Value Loss")
                plt.xlabel("Update Step")
                plt.ylabel("Value Loss")
                plt.title("Value Loss Over Training")
                plt.legend()
                plt.grid()
                
                plt.subplot(2, 2, 3)
                plt.plot(agent.entropy_history, label="Entropy")
                plt.xlabel("Update Step")
                plt.ylabel("Entropy")
                plt.title("Policy Entropy Over Training")
                plt.legend()
                plt.grid()
                
                plt.subplot(2, 2, 4)
                plt.plot(agent.kl_divergence_history, label="KL Divergence")
                plt.xlabel("Update Step")
                plt.ylabel("KL Divergence")
                plt.title("KL Divergence Over Training")
                plt.legend()
                plt.grid()
                
                plt.tight_layout()
                plt.savefig("ppo_training_metrics.png")
                plt.close()

            # Plot power, QoS, PRB breakdown
            plt.figure(figsize=(10, 6))
            plt.plot(power_history, "lightgray", alpha=0.3)
            plt.plot(qos_history, "lightgray", alpha=0.3)
            plt.plot(prb_history, "lightgray", alpha=0.3)
            smoothed_avg_power = smooth_data(power_history)
            smoothed_avg_qos = smooth_data(qos_history)
            smoothed_avg_prb = smooth_data(prb_history)
            plt.plot(smoothed_avg_power, label="Smoothed Average Power")
            plt.plot(smoothed_avg_qos, label="Smoothed Average QoS")
            plt.plot(smoothed_avg_prb, label="Smoothed Average PRB")
            plt.xlabel("Episode")
            plt.ylabel("Average Value")
            plt.title("Power, QoS, PRB Breakdown Over Episodes")
            plt.legend()
            plt.grid()
            plt.savefig("reward_breakdown_ppo.png")
            plt.close()

            if (episode + 1) % 100 == 0:
                print("Saving models...") 
                save_models(agent, episode + 1)  # Save with current episode number
                print("Models saved successfully.")

        env.close()
        time.sleep(2.0)
        
    except Exception as e:
        print(f"Simulation error: {str(e)}")
        
    finally:
        try:
            env.close()
        except:
            pass
        print("Simulation completed.")

if __name__ == "__main__":
    main()