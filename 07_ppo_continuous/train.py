import gymnasium as gym
import torch
import numpy as np
from typing import List, Tuple
from collections import deque
from ppo import PPO

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
    # Pendulum-v1: Actions are [-2, 2]
    # Reward is roughly between -16.27 and 0. Solved is usually considered -200ish, 
    # but theoretically can be close to 0 (upright).
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0] # type: ignore
    action_dim = env.action_space.shape[0] # type: ignore
    
    # Hyperparameters
    lr = 1e-3 # Increased LR
    gamma = 0.99
    gae_lambda = 0.95
    update_timestep = 2000 
    k_epochs = 10
    clip_ratio = 0.2
    reward_scale = 0.1 # Scale rewards
    
    agent = PPO(obs_dim, action_dim, lr, gamma, clip_ratio, k_epochs)
    
    max_episodes = 5000 
    print_every = 20
    
    scores_deque: deque[float] = deque(maxlen=100)
    
    timestep = 0
    
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
        
        while True:
            timestep += 1
            
            # 1. Select Action
            action, log_prob, value = agent.select_action(state)
            
            # 2. Step Env
            action_clipped = np.clip(action, -2.0, 2.0)
            
            next_state, reward, terminated, truncated, _ = env.step(action_clipped)
            done = terminated or truncated
            
            # 3. Save Data
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(float(reward) * reward_scale) # Scale reward
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
            torch.save(agent.policy.state_dict(), "07_ppo_continuous/model.pt")
            
            # Pendulum is solved around -200 or better over 100 trials, but -150 is very good.
            if avg_score >= -200.0:
                print(f"Solved (or at least decent) in {i_episode} episodes! Score: {avg_score}")
                break

if __name__ == "__main__":
    train()
