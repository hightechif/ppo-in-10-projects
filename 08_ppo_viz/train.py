import gymnasium as gym
import torch
import numpy as np
import time
from typing import List, Tuple
from collections import deque
from ppo import PPO

def make_env():
    return gym.make("CartPole-v1")

def calculate_gae_vectorized(
    rewards: np.ndarray, 
    values: np.ndarray, 
    dones: np.ndarray, 
    next_value: np.ndarray, 
    gamma: float = 0.99, 
    lam: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate GAE for vectorized environments.
    
    Args:
        rewards: shape (num_steps, num_envs)
        values: shape (num_steps, num_envs)
        dones: shape (num_steps, num_envs) - indicates if step t completed the episode
        next_value: shape (num_envs,) - value at t=num_steps
    """
    num_steps, num_envs = rewards.shape
    advantages = np.zeros_like(rewards)
    last_gae_lam = np.zeros(num_envs)
    
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_non_terminal = 1.0 - dones[t]
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_val = values[t+1]
            
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
        advantages[t] = last_gae_lam
        
    returns = advantages + values
    return returns, advantages

def train() -> None:
    # 1. Setup Vectorized Envs
    num_envs = 8
    # Using async vector env for speedup
    envs = gym.make_vec("CartPole-v1", num_envs=num_envs, vectorization_mode="async")
    
    obs_dim = envs.observation_space.shape[1] # (num_envs, obs_dim) # type: ignore
    action_dim = envs.action_space.nvec[0] # type: ignore (Discrete: nvec array of n's)
    
    print(f"Obs Dim: {obs_dim}, Action Dim: {action_dim}, Num Envs: {num_envs}")
    
    # 2. Hyperparameters
    lr = 2.5e-4
    gamma = 0.99
    gae_lambda = 0.95
    
    num_steps = 128 # Steps per env per update (Total batch = num_envs * num_steps = 8 * 128 = 1024)
    total_timesteps = 100000 
    
    agent = PPO(obs_dim, action_dim, lr, gamma, 0.2, 4, batch_size=256) # Larger batch size for update
    
    # 3. Training Loop
    # Storage buffers shape: (num_steps, num_envs, ...)
    obs = envs.reset()[0] # (num_envs, obs_dim)
    
    num_updates = total_timesteps // (num_envs * num_steps)
    start_time = time.time()
    
    global_scores = deque(maxlen=100)
    # We need to track scores per env manually because vec env auto-resets
    current_episode_rewards = np.zeros(num_envs)
    
    for update in range(1, num_updates + 1):
        # Buffers
        b_obs = []
        b_actions = []
        b_log_probs = []
        b_rewards = []
        b_dones = []
        b_values = []
        
        for step in range(num_steps):
            # Select action
            # Agent outputs numpy arrays (num_envs,)
            actions, log_probs, values = agent.select_action(obs)
            
            # Step
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            dones = terminations | truncations
            
            # Track rewards
            current_episode_rewards += rewards
            # Check for completed episodes
            for i in range(num_envs):
                if dones[i]:
                    # Note: in Gymnasium VecEnv, 'final_info' or similar contains the real terminal state/info
                    # and 'next_obs' is already the reset state.
                    # But for reward tracking:
                    global_scores.append(current_episode_rewards[i])
                    current_episode_rewards[i] = 0
            
            # Save data
            b_obs.append(obs)
            b_actions.append(actions)
            b_log_probs.append(log_probs)
            b_rewards.append(rewards)
            b_dones.append(dones)
            b_values.append(values)
            
            obs = next_obs
            
        # Convert to numpy arrays: (num_steps, num_envs, ...)
        b_obs_np = np.stack(b_obs)
        b_actions_np = np.stack(b_actions)
        b_log_probs_np = np.stack(b_log_probs)
        b_rewards_np = np.stack(b_rewards)
        b_dones_np = np.stack(b_dones)
        b_values_np = np.stack(b_values)
        
        # Calculate N-step returns / GAE
        with torch.no_grad():
            tensor_obs = torch.from_numpy(obs).float()
            next_value = agent.policy.get_value(tensor_obs).numpy().squeeze()
            
        returns, advantages = calculate_gae_vectorized(
            b_rewards_np, b_values_np, b_dones_np, next_value, gamma, gae_lambda
        )
        
        # Flatten for PPO update
        # We need (num_steps * num_envs, ...)
        flat_obs = b_obs_np.reshape(-1, obs_dim)
        flat_actions = b_actions_np.reshape(-1)
        flat_log_probs = b_log_probs_np.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        
        # Update
        loss = agent.update(flat_obs, flat_actions, flat_log_probs, flat_returns, flat_advantages)
        
        if update % 5 == 0:
            avg_score = np.mean(global_scores) if len(global_scores) > 0 else 0.0
            elapsed = time.time() - start_time
            print(f"Update {update}/{num_updates}\tAverage Score: {avg_score:.2f}\tLoss: {loss:.4f}\tTime: {elapsed:.2f}s")
            
            if avg_score >= 195.0: # CartPole-v1 solved threshold is high, but 195 is good for testing
                print(f"Solved in {update} updates!")
                torch.save(agent.policy.state_dict(), "08_ppo_viz/model.pt")
                break
    
    envs.close()

if __name__ == "__main__":
    train()
