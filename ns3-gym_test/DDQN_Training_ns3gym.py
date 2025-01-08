import gym
import ns3gym
from ns3gym import ns3env
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
import gc
from collections import deque
import random
import time
import matplotlib.pyplot as plt

class SingleDDQNAgent:
    def __init__(self, state_size):
        self.state_size = state_size
        self.action_size = 9
        self.power_options = [20, 30, 40]
        self.cio_options = [-10, 0, 10]
        
        # Hyperparameters
        self.memory = deque(maxlen=50000) 
        self.gamma = 0.95    
        self.epsilon = 1.0   
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99993 # 0.9995 use (min_epsilon/initial_epsilon)^(1/decay_steps) to find the decay rate, better to drop to minimum at 75% of training step
        self.learning_rate = 0.001 # 0.001
        self.update_target_frequency = 500
        self.batch_size = 32
        
        # Create main and target networks with specified input shapes
        self.model = self._build_model()
        self.target_model = self._build_model()
        
        # Create and compile the train step function
        self._create_train_step()
        
        self.update_target_model()
        self.train_step_counter = 0
        self.loss_history = []

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(32, activation='relu'),
            # Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='huber_loss', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def _create_train_step(self):
        """Create a compiled training step function to avoid retracing."""
        @tf.function(reduce_retracing=True)
        def train_step(states, target_q_values):
            with tf.GradientTape() as tape:
                q_values = self.model(states, training=True)
                loss = tf.reduce_mean(tf.square(target_q_values - q_values))
            
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return loss
        
        self.train_step = train_step

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def decode_action(self, action_idx):
        power_idx = action_idx // 3
        cio_idx = action_idx % 3
        return self.power_options[power_idx], self.cio_options[cio_idx]

    @tf.function(reduce_retracing=True)
    def _predict_action(self, state):
        return self.model(state)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action_idx = np.random.randint(self.action_size)
        else:
            state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
            act_values = self._predict_action(state_tensor)
            action_idx = np.argmax(act_values[0])
        
        power, cio = self.decode_action(action_idx)
        return [float(power), float(cio)], action_idx

    def remember(self, state, action_idx, reward, next_state, done):
        self.memory.append((state, action_idx, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample and prepare batch
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([transition[0] for transition in minibatch])
        action_indices = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])
        
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        
        # Get current Q values and next Q values
        current_q_values = self.model(states)
        next_q_values = self.target_model(next_states)
        
        # Calculate target Q values
        target_q_values = current_q_values.numpy()
        next_actions = np.argmax(self.model(next_states), axis=1)
        
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i][action_indices[i]] = rewards[i]
            else:
                target_q_values[i][action_indices[i]] = rewards[i] + self.gamma * next_q_values[i][next_actions[i]]
        
        # Convert target Q values to tensor
        target_q_values = tf.convert_to_tensor(target_q_values, dtype=tf.float32)
        
        # Perform training step
        loss = self.train_step(states, target_q_values)
        self.loss_history.append(float(loss))
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network if needed
        self.train_step_counter += 1
        if self.train_step_counter % self.update_target_frequency == 0:
            self.update_target_model()
            
        return float(loss)


class IndependentDDQNAgent:
    def __init__(self, state_size, num_enbs):
        self.num_enbs = num_enbs
        self.agents = [SingleDDQNAgent(state_size) for _ in range(num_enbs)]
        self.loss_history = []

    def act(self, state):
        actions = []
        self.current_action_indices = []  # Store indices for remember function
        for agent in self.agents:
            action, action_idx = agent.act(state)
            # Convert each action to float
            actions.extend([float(action[0]), float(action[1])])
            self.current_action_indices.append(action_idx)
        # Return as a flat list of floats
        return actions

    def remember(self, state, action, reward, next_state, done):
        for i, agent in enumerate(self.agents):
            agent.remember(state, self.current_action_indices[i], reward, next_state, done)

    def replay(self):
        total_loss = 0
        for agent in self.agents:
            loss = agent.replay()
            if loss is not None:
                total_loss += loss
        avg_loss = total_loss / self.num_enbs if self.num_enbs > 0 else 0
        self.loss_history.append(avg_loss)
        return avg_loss

    @property
    def epsilon(self):
        return self.agents[0].epsilon

def save_models(agent, episode_num, save_dir='saved_models'):
    # Save individual models for each eNB agent
    for i, single_agent in enumerate(agent.agents):
        # Save main model
        main_model_path = f'{save_dir}/enb_{i}_main_model_episode_{episode_num}'
        single_agent.model.save(main_model_path)
        
        # Save target model
        target_model_path = f'{save_dir}/enb_{i}_target_model_episode_{episode_num}'
        single_agent.target_model.save(target_model_path)

def main():
    try:
        env = ns3env.Ns3Env(port=5555)
        
        state = env.reset()
        state_size = len(state)
        num_enbs = 3
        agent = IndependentDDQNAgent(state_size, num_enbs)
        
        n_episodes = 500
        max_steps = 100
        reward_history = []
        epsilon_history = []

        total_steps = 0  # Track total steps across episodes

        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            actions_taken = []

            for step in range(max_steps):
                total_steps += 1  # Increment total steps
                action = agent.act(state)
                actions_taken.append(action)
                
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
                if step % 100 == 0:
                    gc.collect()
                    clear_session()

            reward_history.append(total_reward)
            epsilon_history.append(agent.epsilon)

            print(f"Episode: {episode + 1}/{n_episodes}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Epsilon: {agent.epsilon:.4f}")
            print("------------------------")

            plt.figure(figsize=(10, 6))
            plt.plot(reward_history, label="Total Reward per Episode")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.title("Reward Trend Over Episodes")
            plt.legend()
            plt.grid()
            plt.savefig(f"reward_trend.png")
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(epsilon_history, label="Epsilon Decay")
            plt.xlabel("Episode")
            plt.ylabel("Epsilon Value")
            plt.title("Epsilon Decay Over Episodes")
            plt.legend()
            plt.grid()
            plt.savefig(f"epsilon_decay.png")
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(agent.loss_history, label="Loss History")
            plt.xlabel("Training Steps")
            plt.ylabel("Loss")
            plt.title(f"Loss Trend Over Training Steps")
            plt.legend()
            plt.grid()
            plt.savefig(f"loss_trend.png")
            plt.close()

            if (episode + 1) % 100 == 0:  # Save models every 100 episodes
                save_models(agent, episode)

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