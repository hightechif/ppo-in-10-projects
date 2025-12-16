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
    
    Args:
        rewards: List of rewards
        values: List of state values (V(s_t))
        dones: List of done flags
        next_value: Value of the state after the last step (V(s_{t+1}) for the last step)
        gamma: Discount factor
        lam: GAE Lambda parameter
        
    Returns:
        returns: List of returns (for value loss)
        advantages: List of advantages (for policy loss)
    """
    advantages = []
    gae_accum = 0.0
    
    # We need to append next_value to values to handle the t+1 logic easily
    # Or just handle it in loop
    
    # Iterate backwards
    for t in reversed(range(len(rewards))):
        # If t is the last step, next_val is passed in
        # If not, next_val is values[t+1]
        
        if t == len(rewards) - 1:
            next_val = next_value
            next_done = dones[-1] # This might be wrong logic generally if dones[-1] implies episode end, but we use it as mask
            # But the 'dones' list corresponds to 'is step t terminal?'
        else:
            next_val = values[t+1]
            next_done = dones[t] 
            
        # Delta calculation: r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
        # Note: 'dones[t]' indicates if step t was terminal. If so, next value should be 0 (masked).
        # Actually gae calculation usually masks with next_non_terminal.
        
        # Correct logic:
        # If step t is terminal, then V(s_{t+1}) is not relevant for this step's return (it's 0), 
        # and gae_accum should be reset.
        
        mask = 1.0 - float(dones[t]) 
        
        delta = rewards[t] + gamma * next_val * mask - values[t]
        gae_accum = delta + gamma * lam * mask * gae_accum
        
        advantages.insert(0, gae_accum)
        
    # Returns = Advantage + Value
    returns = [adv + val for adv, val in zip(advantages, values)]
    
    return returns, advantages

def train() -> None:
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0] # type: ignore
    action_dim = env.action_space.n # type: ignore
    
    # Hyperparameters
    lr = 1e-3
    gamma = 0.99
    gae_lambda = 0.95
    update_timestep = 2000 
    k_epochs = 4
    clip_ratio = 0.2
    
    agent = PPO(obs_dim, action_dim, lr, gamma, clip_ratio, k_epochs)
    
    max_episodes = 2000
    print_every = 20
    
    scores_deque: deque[float] = deque(maxlen=100)
    
    timestep = 0
    
    # Buffers
    states: List[np.ndarray] = []
    actions: List[int] = []
    log_probs: List[float] = []
    rewards: List[float] = []
    dones: List[bool] = []
    values: List[float] = [] 
    
    for i_episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        score = 0.0
        
        while True:
            timestep += 1
            
            # 1. Select Action (and get value)
            action, log_prob, value = agent.select_action(state)
            
            # 2. Step Env
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 3. Save Data
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(float(reward))
            dones.append(done)
            values.append(value)
            
            state = next_state
            score += float(reward)
            
            # 4. Update if buffer is full
            if timestep >= update_timestep:
                # Bootstrap value
                with torch.no_grad():
                    next_state_ten = torch.from_numpy(next_state).float().unsqueeze(0)
                    next_value = agent.policy.get_value(next_state_ten).item()
                
                # Calculate GAE
                returns, advantages = calculate_gae(rewards, values, dones, next_value, gamma, gae_lambda)
                
                # Update
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
            print(f"Episode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}\tLast Loss: {agent.last_loss if hasattr(agent, 'last_loss') else 'N/A'}")
            
        if np.mean(scores_deque) >= 195.0:
            print(f"Solved in {i_episode} episodes!")
            torch.save(agent.policy.state_dict(), "06_ppo_gae/model.pt")
            break

if __name__ == "__main__":
    train()
