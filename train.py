import numpy as np
from ddqn_agent import DDQNAgent
from ns3_env import Ns3Env

if __name__ == "__main__":
    env = Ns3Env()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DDQNAgent(state_size, action_size)
    
    episodes = 1000
    batch_size = 32
    
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        print(f"Episode {e + 1}/{episodes} - Initial State: {state}")  # Print initial state
        
        done = False
        step = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            # Print the current state, action taken, reward, and next state
            print(f"Step {step} - State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            step += 1
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        agent.update_target_model()
        print(f"Episode {e + 1}/{episodes} finished.\n")

    # Save the trained model
    agent.model.save("ddqn_ns3_model.h5")