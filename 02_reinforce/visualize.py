import gymnasium as gym
import torch
import time
from policy import PolicyNetwork

def visualize():
    env = gym.make("CartPole-v1", render_mode="human")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy = PolicyNetwork(obs_dim, action_dim)
    
    # Load the saved model
    try:
        policy.load_state_dict(torch.load("02_reinforce/policy.pt"))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: '02_reinforce/policy.pt' not found. Please run train.py first.")
        return

    policy.eval() # Set to evaluation mode
    
    for episode in range(5):
        state, _ = env.reset()
        total_reward = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action, _ = policy.get_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            time.sleep(0.02) # Slow down slightly
            
        print(f"Episode {episode+1}: Total Reward: {total_reward}")
        
    env.close()

if __name__ == "__main__":
    visualize()
