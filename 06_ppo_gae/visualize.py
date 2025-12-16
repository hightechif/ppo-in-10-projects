import gymnasium as gym
import torch
import time
from ppo import PPO

def visualize() -> None:
    env = gym.make("CartPole-v1", render_mode="human")
    obs_dim = env.observation_space.shape[0] # type: ignore
    action_dim = env.action_space.n # type: ignore
    
    # Needs to match training config
    agent = PPO(obs_dim, action_dim)
    
    try:
        agent.policy.load_state_dict(torch.load("06_ppo_gae/model.pt", map_location="cpu"))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: 06_ppo_gae/model.pt not found. Train the agent first.")
        return

    agent.policy.eval()

    for episode in range(5):
        state, _ = env.reset()
        total_reward: float = 0.0
        done = False
        
        while not done:
            action, _, _ = agent.select_action(state)
            
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
            time.sleep(0.02)
            
        print(f"Episode {episode+1}: Total Reward: {total_reward}")
    
    env.close()

if __name__ == "__main__":
    visualize()
