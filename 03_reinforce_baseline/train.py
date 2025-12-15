import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import List, Any
from models import PolicyNetwork, ValueNetwork

def train() -> None:
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0] # type: ignore
    action_dim = env.action_space.n # type: ignore
    
    policy = PolicyNetwork(obs_dim, action_dim)
    value_net = ValueNetwork(obs_dim)
    
    policy_optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    value_optimizer = optim.Adam(value_net.parameters(), lr=1e-2)
    
    n_episodes = 500
    gamma = 0.99
    print_every = 50
    
    scores_deque: deque[float] = deque(maxlen=100)
    
    for i_episode in range(1, n_episodes + 1):
        saved_log_probs = []
        rewards: List[float] = []
        states = []
        state, _ = env.reset()
        
        for t in range(1000):
            states.append(state)
            action, log_prob = policy.get_action(state)
            saved_log_probs.append(log_prob)
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            if terminated or truncated:
                break
                
        scores_deque.append(sum(rewards))
        
        # Calculate Returns
        returns: List[float] = []
        R: float = 0.0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
            
        returns_tensor = torch.tensor(returns)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-9)
        
        # Update Value Network and Calculate Advantage
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        values = value_net(states_tensor).squeeze()
        
        # Value Loss (MSE)
        value_loss = F.mse_loss(values, returns_tensor)
        
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()
        
        # Re-calculate values (detached) to use as baseline
        # Or simply use the value computed before update.
        # Advantage = Return - Value
        # Note: We detach values because we don't want to backprop through value_net here
        advantages = returns_tensor - values.detach()
        
        # Update Policy Network
        policy_loss = []
        for log_prob, adv in zip(saved_log_probs, advantages):
            policy_loss.append(-log_prob * adv)
            
        policy_optimizer.zero_grad()
        policy_loss_tensor = torch.cat(policy_loss).sum()
        policy_loss_tensor.backward()
        policy_optimizer.step()
        
        if i_episode % print_every == 0:
            print(f"Episode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}")
            if np.mean(scores_deque) >= 195.0:
                print(f"Environment solved in {i_episode} episodes!")
                torch.save(policy.state_dict(), "03_reinforce_baseline/policy.pt")
                torch.save(value_net.state_dict(), "03_reinforce_baseline/value.pt")
                print("Models saved.")
                break

    env.close()

if __name__ == "__main__":
    train()
