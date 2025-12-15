import gymnasium as gym
import torch
import time
from models import PolicyNetwork

def visualize():
    env = gym.make("CartPole-v1", render_mode="human")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy = PolicyNetwork(obs_dim, action_dim)
    
    try:
        policy.load_state_dict(torch.load("03_reinforce_baseline/policy.pt"))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: '03_reinforce_baseline/policy.pt' not found. Please run train.py first.")
        return

    policy.eval()
    
    for episode in range(5):
        state, _ = env.reset()
        total_reward = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action, _ = policy.get_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            time.sleep(0.02)
            
        print(f"Episode {episode+1}: Total Reward: {total_reward}")
        
    env.close()

if __name__ == "__main__":
    visualize()
