import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import List, Any
from models import ActorCritic

def train() -> None:
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0] # type: ignore
    action_dim = env.action_space.n # type: ignore
    
    # Hyperparameters
    n_steps = 5  # Number of steps to unroll before update
    gamma = 0.99
    lr = 1e-3
    max_episodes = 1000
    
    model = ActorCritic(obs_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    scores_deque: deque[float] = deque(maxlen=100)
    
    # Store current state globally for the loop
    state, _ = env.reset()
    
    total_steps = 0
    
    for i_episode in range(1, max_episodes + 1):
        # We don't reset the env every loop iteration in A2C strictly speaking,
        # but to keep it simple and episodic-like for logging, we can, 
        # OR we just run until done like a true continuous loop.
        # Let's do the "run episode until done, but update every n steps" approach.
        
        episode_reward: float = 0.0
        state, _ = env.reset()
        done = False
        
        while not done:
            log_probs: List[torch.Tensor] = []
            values_list: List[torch.Tensor] = []
            rewards: List[float] = []
            entropies: List[torch.Tensor] = []
            
            # 1. Collect n steps
            for _ in range(n_steps):
                action, log_prob, value, entropy = model.get_action_and_value(state)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                log_probs.append(log_prob)
                values_list.append(value)
                rewards.append(float(reward))
                entropies.append(entropy)
                episode_reward += float(reward)
                
                state = next_state
                
                if done:
                    break
            
            # 2. Bootstrap Value
            # If we stopped because of n_steps, R = V(s_next)
            # If we stopped because done, R = 0
            
            R: float = 0.0
            if not done:
                _, _, next_value, _ = model.get_action_and_value(state)
                R = next_value.item()
                
            # 3. Calculate Returns and Advantages
            returns: List[float] = []
            # Reverse order to calculate returns
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
                
            returns_tensor = torch.tensor(returns)
            values_tensor = torch.cat(values_list).view(-1)
            log_probs_tensor = torch.cat(log_probs)
            entropies_tensor = torch.cat(entropies)
            
            # Advantage = Return - Value
            advantage = returns_tensor - values_tensor.detach()
            
            # 4. Update
            actor_loss = -(log_probs_tensor * advantage).mean()
            critic_loss = F.mse_loss(values_tensor, returns_tensor)
            entropy_loss = -entropies_tensor.mean()
            
            loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_steps += len(rewards)

        scores_deque.append(episode_reward)
        
        if i_episode % 10 == 0:
            avg_score = np.mean(scores_deque)
            print(f"Episode {i_episode}\tAverage Score: {avg_score:.2f}")
            if avg_score >= 195.0:
                print(f"Solved in {i_episode} episodes!")
                torch.save(model.state_dict(), "04_a2c/model.pt")
                print("Model saved to 04_a2c/model.pt")
                break
                
    env.close()

if __name__ == "__main__":
    train()
