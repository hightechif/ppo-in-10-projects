import torch
import numpy as np
from typing import List, Tuple
from collections import deque
from ppo import PPO
from ant_env import AntEnv

def calculate_gae(
    rewards: List[float], 
    values: List[float], 
    dones: List[bool], 
    next_value: float, 
    gamma: float = 0.99, 
    lam: float = 0.95
) -> Tuple[List[float], List[float]]:
    """
    Calculate Generalized Advantage Estimation (GAE).
    """
    advantages = []
    gae_accum = 0.0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t+1]
            
        mask = 1.0 - float(dones[t]) 
        
        delta = rewards[t] + gamma * next_val * mask - values[t]
        gae_accum = delta + gamma * lam * mask * gae_accum
        
        advantages.insert(0, gae_accum)
        
    returns = [adv + val for adv, val in zip(advantages, values)]
    
    return returns, advantages

def train() -> None:
    # Use our Custom AntEnv
    # Important: Ant is much harder than Pendulum. Training might take longer.
    env = AntEnv(render_mode=None) # Direct mode for training
    
    obs_dim = env.observation_space.shape[0] # type: ignore
    action_dim = env.action_space.shape[0] # type: ignore
    
    # Hyperparameters for Ant (Robotics)
    # Typically need larger batch sizes and more epochs per update
    lr = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    update_timestep = 2048 # Increased buffer size from 2000 to 2048 (power of 2 good for batching)
    k_epochs = 10
    clip_ratio = 0.2
    
    # We should perform batch_size updates larger than 64 now maybe?
    # Let's keep 64 or 128.
    
    agent = PPO(obs_dim, action_dim, lr, gamma, clip_ratio, k_epochs, batch_size=64)
    
    max_episodes = 5000 
    max_steps_per_episode = 1000
    print_every = 10
    
    scores_deque: deque[float] = deque(maxlen=100)
    
    timestep = 0 # Updates counter
    
    # Buffers
    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    log_probs: List[float] = []
    rewards: List[float] = []
    dones: List[bool] = []
    values: List[float] = [] 
    
    for i_episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        score = 0.0
        episode_steps = 0
        
        while True:
            timestep += 1
            episode_steps += 1
            
            # 1. Select Action
            action, log_prob, value = agent.select_action(state)
            
            # 2. Step Env
            next_state, reward, terminated, _, _ = env.step(action)
            truncated = episode_steps >= max_steps_per_episode
            done = terminated or truncated
            
            # 3. Save Data
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(float(reward)) # Ant rewards are reasonable scale, usu no need to scale strictly but helps.
            dones.append(done)
            values.append(value)
            
            state = next_state
            score += float(reward)
            
            # 4. Update if buffer is full
            if timestep >= update_timestep:
                with torch.no_grad():
                    next_state_ten = torch.from_numpy(next_state).float().unsqueeze(0)
                    next_value = agent.policy.get_value(next_state_ten).item()
                
                returns, advantages = calculate_gae(rewards, values, dones, next_value, gamma, gae_lambda)
                
                loss = agent.update(states, actions, log_probs, returns, advantages)
                
                # Clear buffers
                states = []
                actions = []
                log_probs = []
                rewards = []
                dones = []
                values = []
                timestep = 0
            
            if done:
                break
                
        scores_deque.append(score)
        
        if i_episode % print_every == 0:
            avg_score = np.mean(scores_deque)
            print(f"Episode {i_episode}\tAverage Score: {avg_score:.2f}\tLast Loss: {agent.last_loss if hasattr(agent, 'last_loss') else 'N/A'}")
            
            # Save checkpoint
            torch.save(agent.policy.state_dict(), "09_ppo_pybullet/model.pt")
            
            # Solved threshold is fuzzy for custom env. 
            # If it walks forward consistently without falling, reward should be positive and accumulated.
            # > 1000 is usually good for Ant over 1000 steps.
            # Our episode length is limited by 'done' (falling).
            # If it learns to survive, score increases.
            
            if avg_score >= 200.0: 
                print(f"Solved (Walking!) in {i_episode} episodes! Score: {avg_score}")
                break
    
    env.close()

if __name__ == "__main__":
    train()
