import gym
import ns3gym
from ns3gym import ns3env
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
import gc
from collections import deque
import random
import time
import matplotlib.pyplot as plt

class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size*2
        
        # Hyperparameters
        self.memory = deque(maxlen=1000)
        self.gamma = 0.9    # discount rate (0.95)
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995 # 0.9995
        self.learning_rate = 0.01 # 0.001
        self.update_target_frequency = 100 #40
        self.batch_size = 16 #16
        
        # Create main and target networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Training step counter
        self.train_step = 0
        self.loss_history = []  # Store all losses

    def _build_model(self):
        model = Sequential([
            Dense(32, activation='relu', input_shape=(self.state_size,)), #64
            Dense(16, activation='relu'), #64
            Dense(self.action_size, activation='linear')  # Use tanh activation to scale actions to [-1, 1]
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(-3, 3, self.action_size)
        
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        # Add some noise to prevent similar outputs
        print(f"Action Values before clipping: {act_values}")
        return np.clip(act_values[0], -3, 3)
        

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Prepare arrays for batch processing
        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])
        
        # Get current Q-values for all states
        current_q_values = self.model.predict(states, verbose=0)
        
        # Get next Q-values from target model for all next states
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Get Q-values from main model for action selection (DDQN)
        next_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)
        
        # Prepare targets
        targets = current_q_values.copy()
        
        for i in range(self.batch_size):
            if dones[i]:
                # For terminal states, target is just the reward
                targets[i] = rewards[i]
            else:
                # DDQN update
                # Use main network to select action and target network to evaluate it
                selected_action = next_actions[i]
                targets[i] = rewards[i] + self.gamma * next_q_values[i][selected_action]
        
        # Train the model and store the loss
        history = self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)
        avg_loss = history.history['loss'][0]
        self.loss_history.append(avg_loss)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network periodically
        self.train_step += 1
        if self.train_step % self.update_target_frequency == 0:
            self.update_target_model()
        
        print(f"Replay - Average Loss: {avg_loss:.4f}")


def preprocess_state(obs):
    # Flatten and normalize the state if it's multi-dimensional
    state = np.array(obs).flatten()
    return state / np.linalg.norm(state)  # Normalize the state vector

def main():
    try:
        env = ns3env.Ns3Env(port=5555)
        
        state = env.reset()
        state_size = len(state)
        action_size = env.action_space.shape[0]
        agent = DDQNAgent(state_size, action_size)
        
        n_episodes = 100
        max_steps = 100
        reward_history = []
        epsilon_history = []

        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            actions_taken = []

            for step in range(max_steps):
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
                    clear_session()  # Clears the TensorFlow session to free memory

            reward_history.append(total_reward)
            epsilon_history.append(agent.epsilon)

            # Per-episode reporting
            print(f"Episode: {episode + 1}/{n_episodes}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Max Action Taken: {np.max(actions_taken):.2f}")
            print(f"Min Action Taken: {np.min(actions_taken):.2f}")
            print(f"Epsilon: {agent.epsilon:.4f}")
            print("------------------------")

            # Save reward trends
            plt.figure(figsize=(10, 6))
            plt.plot(reward_history, label="Total Reward per Episode")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.title("Reward Trend Over Episodes")
            plt.legend()
            plt.grid()
            plt.savefig(f"reward_trend_episode_{episode + 1}.png")
            plt.close()

            # Save epsilon trends
            plt.figure(figsize=(10, 6))
            plt.plot(epsilon_history, label="Epsilon Decay")
            plt.xlabel("Episode")
            plt.ylabel("Epsilon Value")
            plt.title("Epsilon Decay Over Episodes")
            plt.legend()
            plt.grid()
            plt.savefig(f"epsilon_decay_episode_{episode + 1}.png")
            plt.close()

            # Save loss trends
            plt.figure(figsize=(10, 6))
            plt.plot(agent.loss_history, label="Loss History")
            plt.xlabel("Training Steps")
            plt.ylabel("Loss")
            plt.title(f"Loss Trend Up to Episode {episode + 1}")
            plt.legend()
            plt.grid()
            plt.savefig(f"loss_trend_episode_{episode + 1}.png")
            plt.close()

            # Optionally save the model every 10 episodes
            if (episode + 1) % 10 == 0:
                agent.model.save(f'ddqn_model_episode_{episode + 1}.h5')

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