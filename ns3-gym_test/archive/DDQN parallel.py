import multiprocessing as mp
from functools import partial
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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress most warnings

class SingleDDQNAgent:
    def __init__(self, state_size):
        self.state_size = state_size
        self.action_size = 9
        self.power_options = [-3, 0, 3]
        self.cio_options = [-3, 0, 3]
        
        # Hyperparameters
        self.memory = deque(maxlen=1100)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.997
        self.learning_rate = 0.001
        self.update_target_frequency = 200
        self.batch_size = 32

        # Create main and target networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        
        self.update_target_model()
        self.train_step_counter = 0
        self.loss_history = []

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def decode_action(self, action_idx):
        power_idx = action_idx // 3
        cio_idx = action_idx % 3
        return self.power_options[power_idx], self.cio_options[cio_idx]

    @tf.function(reduce_retracing=True)
    def train_step(self, states, target_q_values):
        """Moved train_step outside _create_train_step to avoid pickling issues."""
        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)
            loss = tf.reduce_mean(tf.square(target_q_values - q_values))
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action_idx = np.random.randint(self.action_size)
        else:
            state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
            act_values = self.model(state_tensor)
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


class ParallelIndependentDDQNAgent:
    def __init__(self, state_size, num_enbs=10, num_processes=4):
        self.num_enbs = num_enbs
        self.num_processes = min(num_processes, num_enbs)
        self.agents = [SingleDDQNAgent(state_size) for _ in range(num_enbs)]
        self.loss_history = []
        self.pool = mp.Pool(processes=self.num_processes)

    def act(self, state):
        # Parallel action selection
        chunk_size = self.num_enbs // self.num_processes
        state_chunks = [state] * self.num_enbs
        
        results = self.pool.map(partial(self._parallel_act, state=state), 
                              range(self.num_enbs))
        
        actions = []
        self.current_action_indices = []
        for action, action_idx in results:
            actions.extend([float(action[0]), float(action[1])])
            self.current_action_indices.append(action_idx)
        return actions

    def _parallel_act(self, agent_idx, state):
        return self.agents[agent_idx].act(state)

    def remember(self, state, action, reward, next_state, done):
        # Parallel memory storage
        args = [(i, state, self.current_action_indices[i], reward, next_state, done) 
                for i in range(self.num_enbs)]
        self.pool.map(self._parallel_remember, args)

    def _parallel_remember(self, args):
        agent_idx, state, action_idx, reward, next_state, done = args
        self.agents[agent_idx].remember(state, action_idx, reward, next_state, done)

    def replay(self):
        # Parallel replay
        results = self.pool.map(self._parallel_replay, range(self.num_enbs))
        losses = [loss for loss in results if loss is not None]
        avg_loss = sum(losses) / len(losses) if losses else 0
        self.loss_history.append(avg_loss)
        return avg_loss

    def _parallel_replay(self, agent_idx):
        return self.agents[agent_idx].replay()

    @property
    def epsilon(self):
        return self.agents[0].epsilon

    def close(self):
        self.pool.close()
        self.pool.join()

def main():
    try:
        env = ns3env.Ns3Env(port=5555)
        
        state = env.reset()
        state_size = len(state)
        num_enbs = 10
        num_processes = mp.cpu_count() - 1  # Leave one CPU for system tasks
        
        agent = ParallelIndependentDDQNAgent(state_size, num_enbs, num_processes)
        
        # Rest of the main function remains the same
        n_episodes = 20
        max_steps = 100
        reward_history = []
        reward_per_step = []
        epsilon_history = []

        total_steps = 0

        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            actions_taken = []

            for step in range(max_steps):
                total_steps += 1
                action = agent.act(state)
                actions_taken.append(action)
                
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                
                total_reward += reward
                reward_per_step.append(reward)
                state = next_state
                
                if done:
                    break
            
                if step % 100 == 0:
                    gc.collect()
                    clear_session()

            # [Plotting code remains the same]
            
            reward_history.append(total_reward)
            epsilon_history.append(agent.epsilon)

            print(f"Episode: {episode + 1}/{n_episodes}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Epsilon: {agent.epsilon:.4f}")
            print("------------------------")

            lt.figure(figsize=(10, 6))
            plt.plot(reward_per_step, label="Reward per Step")
            plt.xlabel("Steps")
            plt.ylabel("Reward")
            plt.title("Reward Trend Over Steps")
            plt.legend()
            plt.grid()
            plt.savefig(f"reward_trend_steps_{episode + 1}.png")
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(reward_history, label="Total Reward per Episode")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.title("Reward Trend Over Episodes")
            plt.legend()
            plt.grid()
            plt.savefig(f"reward_trend_episode_{episode + 1}.png")
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(epsilon_history, label="Epsilon Decay")
            plt.xlabel("Episode")
            plt.ylabel("Epsilon Value")
            plt.title("Epsilon Decay Over Episodes")
            plt.legend()
            plt.grid()
            plt.savefig(f"epsilon_decay_episode_{episode + 1}.png")
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(agent.loss_history, label="Loss History")
            plt.xlabel("Training Steps")
            plt.ylabel("Loss")
            plt.title(f"Loss Trend Up to Episode {episode + 1}")
            plt.legend()
            plt.grid()
            plt.savefig(f"loss_trend_episode_{episode + 1}.png")
            plt.close()

        agent.close()
        env.close()
        time.sleep(2.0)
        
    except Exception as e:
        print(f"Simulation error: {str(e)}")
        
    finally:
        try:
            env.close()
            agent.close()
        except:
            pass
        print("Simulation completed.")

if __name__ == "__main__":
    mp.set_start_method('spawn')  # Required for TensorFlow
    main()