import gym
import ns3gym
from ns3gym import ns3env
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import time

class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size*2
        
        # Hyperparameters
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.update_target_frequency = 100
        self.batch_size = 32
        
        # Create main and target networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Training step counter
        self.train_step = 0

    def _build_model(self):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.state_size,)),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Random action: generate random power and CIO adjustments
            return np.random.uniform(-3, 3, self.action_size)  # Adjustments between -3 and 3 dB
        
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return act_values[0]

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state.reshape(1, -1), verbose=0)
            
            if done:
                target[0] = reward
            else:
                # DDQN update
                a = np.argmax(self.model.predict(next_state.reshape(1, -1), verbose=0)[0])
                t = self.target_model.predict(next_state.reshape(1, -1), verbose=0)[0]
                target[0] = reward + self.gamma * t[a]
            
            self.model.fit(state.reshape(1, -1), target, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network periodically
        self.train_step += 1
        if self.train_step % self.update_target_frequency == 0:
            self.update_target_model()

def preprocess_state(obs):
    # Flatten the state if it's multi-dimensional
    return np.array(obs).flatten()

def main():
    try:
        # Create and configure the environment
        env = ns3env.Ns3Env(port=5555)
        
        # Get environment dimensions
        state = env.reset()
        state_size = len(state)
        action_size = env.action_space.shape[0]
        
        # Create DDQN agent
        agent = DDQNAgent(state_size, action_size)
        
        # Training parameters
        n_episodes = 10
        max_steps = 100 #limit the number of steps per episode in case of simulation not ending in ns3
        
        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                # Get action from agent
                action = agent.act(state)
                
                # Take action in environment
                next_state, reward, done, info = env.step(action)
                
                # Store experience and train
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                
                total_reward += reward
                state = next_state
                
                # Break if episode is done (game over in ns3)
                if done:
                    break
            
            print(f"Episode: {episode + 1}/{n_episodes}")
            print(f"Total Reward: {total_reward}")
            print(f"Epsilon: {agent.epsilon}")
            print("------------------------")
            
            # Save model periodically
            if (episode + 1) % 100 == 0:
                agent.model.save(f'ddqn_model_episode_{episode + 1}.h5')
                
        # Proper cleanup
        env.close()
        time.sleep(2.0)  # Give time for ns-3 to clean up
        
    except Exception as e:
        print(f"Simulation error: {str(e)}")
        
    finally:
        # Ensure environment is closed even if there's an error
        try:
            env.close()
        except:
            pass
        print("Simulation completed.")

if __name__ == "__main__":
    main()