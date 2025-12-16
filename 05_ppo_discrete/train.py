import gymnasium as gym
import torch
import numpy as np
from typing import List
from collections import deque
from ppo import PPO

def train() -> None:
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0] # type: ignore
    action_dim = env.action_space.n # type: ignore
    
    # Hyperparameters
    lr = 1e-3
    gamma = 0.99
    update_timestep = 400  # More frequent updates
    k_epochs = 4
    clip_ratio = 0.2
    
    agent = PPO(obs_dim, action_dim, lr, gamma, clip_ratio, k_epochs)
    
    max_episodes = 1000
    print_every = 10
    
    scores_deque: deque[float] = deque(maxlen=100)
    
    timestep = 0
    
    # buffers
    states: List[np.ndarray] = []
    actions: List[int] = []
    log_probs: List[float] = []
    rewards: List[float] = []
    dones: List[bool] = []
    
    for i_episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        score = 0.0
        
        while True:
            timestep += 1
            
            # 1. Select Action
            action, log_prob, _ = agent.select_action(state)
            
            # 2. Step Env
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 3. Save Data
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(float(reward))
            dones.append(done)
            
            state = next_state
            score += float(reward)
            
            # 4. PPO Update if timestep >= update_timestep
            if timestep % update_timestep == 0:
                # Bootstrap value if not done with episode (ignored here for simplicity in vanilla PPO implementation 
                # usually we adhere to episode boundaries or handle bootstrapping carefully)
                # For this specific "update every N steps" logic:
                
                # Compute Returns (Monte-Carlo style or N-step with bootstrapping)
                # Since we have a mix of episodes in the buffer, calculating returns is tricky without GAE buffer logic.
                # To keep Project 5 simple (Pre-GAE), let's strictly calculate returns-to-go 
                # assuming we only update at end of episodes OR we handle the cut-off.
                
                # BETTER APPROACH for Simple PPO:
                # Just collect trajectory segments and bootstrap only at the very end of the batch.
                
                with torch.no_grad():
                    next_state_ten = torch.from_numpy(next_state).float().unsqueeze(0)
                    next_value = agent.policy.get_value(next_state_ten).item()
                    
                batch_returns: List[float] = []
                batch_advantages: List[float] = []
                
                # Calculate Returns and Advantages using Generalized Advantage Estimation (GAE) 
                # OR simple N-step. The plan said "Advantage: Using N-step returns (keeping GAE for Project 6)".
                # But treating a batch of 2000 steps which might cut an episode in half requires some care.
                
                # Let's do simple discounted return from the end of the batch backwards.
                # If the batch ends in the middle of an episode, we use 'next_value' as the bootstrap.
                # If dones[-1] is True, bootstrap is 0.
                
                R = next_value * (1 - int(dones[-1]))
                
                # We need to iterate backwards through the collected batch
                reversed_rewards = reversed(rewards)
                reversed_dones = reversed(dones)
                reversed_states = reversed(states) # We don't need this loops, just indices
                
                # Use plain Python loop with indices
                returns = [0.0] * len(rewards)
                
                for t in reversed(range(len(rewards))):
                    if dones[t]:
                        R = 0.0
                    R = rewards[t] + gamma * R
                    returns[t] = R
                
                # Compute advantages (Return - Value)
                # We need values for all recorded states
                # This is a bit inefficient to run forward pass again but clarity is key
                states_ten = torch.tensor(np.array(states), dtype=torch.float32)
                with torch.no_grad():
                    values_ten = agent.policy.get_value(states_ten).squeeze()
                    values_np = values_ten.numpy()
                    
                advantages = [r - v for r, v in zip(returns, values_np)]
                
                # Update
                loss = agent.update(states, actions, log_probs, returns, advantages)
                
                # Clear buffers
                states = []
                actions = []
                log_probs = []
                rewards = []
                dones = []
            
            if done:
                break
                
        scores_deque.append(score)
        
        if i_episode % print_every == 0:
            print(f"Episode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}\tLast Loss: {agent.last_loss if hasattr(agent, 'last_loss') else 'N/A'}")
            
        if np.mean(scores_deque) >= 195.0:
            print(f"Solved in {i_episode} episodes!")
            torch.save(agent.policy.state_dict(), "05_ppo_discrete/model.pt")
            break

if __name__ == "__main__":
    train()
