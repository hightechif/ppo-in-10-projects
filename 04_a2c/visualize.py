import gymnasium as gym
import torch
import time
from models import ActorCritic

def visualize() -> None:
    env = gym.make("CartPole-v1", render_mode="human")
    obs_dim = env.observation_space.shape[0] # type: ignore
    action_dim = env.action_space.n # type: ignore
    
    model = ActorCritic(obs_dim, action_dim)
    
    try:
        model.load_state_dict(torch.load("04_a2c/model.pt"))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: '04_a2c/model.pt' not found. Please run train.py first.")
        return

    model.eval()
    
    for episode in range(5):
        state, _ = env.reset()
        total_reward: float = 0.0
        done = False
        
        while not done:
            action, _, _, _ = model.get_action_and_value(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
            time.sleep(0.02)
            
        print(f"Episode {episode+1}: Total Reward: {total_reward}")
        
    env.close()

if __name__ == "__main__":
    visualize()
