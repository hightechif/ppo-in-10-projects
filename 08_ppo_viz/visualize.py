import gymnasium as gym
import torch
import numpy as np
import time
from ppo import PPO

def visualize() -> None:
    num_envs = 4
    # Using sync vector env for visualization to ensure rendering might work better locally, 
    # but 'human' render mode in make_vec often tries to open one window per env or one tile.
    # Gymnasium's make_vec defaults to Async if not specified but we can force Sync.
    try:
        envs = gym.make_vec("CartPole-v1", num_envs=num_envs, render_mode="human", vectorization_mode="sync")
    except Exception as e:
        print(f"Failed to create vectorized env with render_mode='human': {e}")
        print("Falling back to single env visualization.")
        visualize_single()
        return

    obs_dim = envs.observation_space.shape[1] # type: ignore
    action_dim = envs.action_space.nvec[0] # type: ignore
    
    agent = PPO(obs_dim, action_dim)
    
    try:
        agent.policy.load_state_dict(torch.load("08_ppo_viz/model.pt", map_location="cpu"))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: 08_ppo_viz/model.pt not found. Train the agent first.")
        return

    agent.policy.eval()

    obs = envs.reset()[0]
    
    # Run for a few steps
    for _ in range(500):
        actions, _, _ = agent.select_action(obs)
        obs, rewards, dones, truncations, infos = envs.step(actions)
        
        # Slow down slightly for viewing
        time.sleep(0.05)
        
        if any(dones) or any(truncations):
            # VecEnv auto resets
            pass
            
    envs.close()

def visualize_single() -> None:
    env = gym.make("CartPole-v1", render_mode="human")
    obs_dim = env.observation_space.shape[0] # type: ignore
    action_dim = env.action_space.n # type: ignore
    
    agent = PPO(obs_dim, action_dim)
    agent.policy.load_state_dict(torch.load("08_ppo_viz/model.pt", map_location="cpu"))
    agent.policy.eval()
    
    for _ in range(3):
        state, _ = env.reset()
        done = False
        while not done:
            state = state.reshape(1, -1) # Batch dim
            action, _, _ = agent.select_action(state)
            state, _, term, trunc, _ = env.step(action[0])
            done = term or trunc
            time.sleep(0.02)
    env.close()

if __name__ == "__main__":
    visualize()
