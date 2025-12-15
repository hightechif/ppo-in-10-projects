import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
from collections import deque
from typing import List, Any
from policy import PolicyNetwork

def train() -> None:
    env = gym.make("CartPole-v1")
    # Cast to Any to avoid mypy complaining about generic Space not having shape/n
    obs_dim = env.observation_space.shape[0] # type: ignore
    action_dim = env.action_space.n # type: ignore
    
    policy = PolicyNetwork(obs_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    
    n_episodes = 500
    gamma = 0.99
    print_every = 50
    
    # Track scores for average logging
    scores_deque: deque[float] = deque(maxlen=100)
    
    for i_episode in range(1, n_episodes + 1):
        saved_log_probs = []
        rewards: List[float] = []
        state, _ = env.reset()
        
        # 1. Collect Trajectory
        for t in range(1000): # Limit max steps
            action, log_prob = policy.get_action(state)
            saved_log_probs.append(log_prob)
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            if terminated or truncated:
                break
                
        scores_deque.append(sum(rewards))
        
        # 2. Calculate Discounted Returns (Monte-Carlo)
        returns: List[float] = []
        R: float = 0.0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
            
        returns_tensor = torch.tensor(returns)
        # Normalize returns for stability (optional but recommended)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-9)
        
        # 3. Update Policy
        policy_loss = []
        for log_prob, R_val in zip(saved_log_probs, returns_tensor):
            policy_loss.append(-log_prob * R_val)
            
        optimizer.zero_grad()
        policy_loss_tensor = torch.cat(policy_loss).sum()
        policy_loss_tensor.backward()
        optimizer.step()
        
        if i_episode % print_every == 0:
            print(f"Episode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}")
            if np.mean(scores_deque) >= 195.0:
                print(f"Environment solved in {i_episode} episodes!")
                torch.save(policy.state_dict(), "02_reinforce/policy.pt")
                print("Model saved to 02_reinforce/policy.pt")
                break
                
    if i_episode == n_episodes:
        print("Max episodes reached.")
        torch.save(policy.state_dict(), "02_reinforce/policy.pt")

                
    env.close()

if __name__ == "__main__":
    train()
