import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, List, Optional

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128) -> None:
        super(ActorCritic, self).__init__()
        # Shared Feature Extractor
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        
        # Actor Head
        self.actor = nn.Linear(hidden_size, action_dim)
        
        # Critic Head
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return x

    def get_action_and_value(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            action: The selected action (or the one passed in)
            log_prob: Log probability of the action
            entropy: Entropy of the distribution
            value: Value estimate
        """
        features = self.forward(state)
        logits = self.actor(features)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        if action is None:
            action = dist.sample()
            
        return action, dist.log_prob(action), dist.entropy(), self.critic(features)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        features = self.forward(state)
        return self.critic(features)

class PPO:
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: int, 
        lr: float = 3e-4, 
        gamma: float = 0.99, 
        clip_ratio: float = 0.2,
        k_epochs: int = 4,
        batch_size: int = 64  # Mini-batch size for updates, not used strictly if we do full batch update, but good for PPO
    ) -> None:
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        
        self.policy = ActorCritic(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        with torch.no_grad():
            state_ten = torch.from_numpy(state).float().unsqueeze(0)
            action, log_prob, _, value = self.policy.get_action_and_value(state_ten)
        
        return int(action.item()), float(log_prob.item()), float(value.item())

    def update(
        self, 
        states: List[np.ndarray], 
        actions: List[int], 
        log_probs: List[float], 
        returns: List[float], 
        advantages: List[float]
    ) -> float:
        """
        PPO Update Step
        """
        # Convert to tensors
        states_ten = torch.tensor(np.array(states), dtype=torch.float32)
        actions_ten = torch.tensor(actions, dtype=torch.long)
        old_log_probs_ten = torch.tensor(log_probs, dtype=torch.float32)
        returns_ten = torch.tensor(returns, dtype=torch.float32)
        advantages_ten = torch.tensor(advantages, dtype=torch.float32)
        
        # Normalize advantages
        advantages_ten = (advantages_ten - advantages_ten.mean()) / (advantages_ten.std() + 1e-8)
        
        total_loss_val = 0.0
        
        self.last_loss = 0.0
        
        # PPO Epochs
        for _ in range(self.k_epochs):
            # Shuffle indices
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]
                
                # Mini-batch selection
                batch_states = states_ten[batch_indices]
                batch_actions = actions_ten[batch_indices]
                batch_old_log_probs = old_log_probs_ten[batch_indices]
                batch_returns = returns_ten[batch_indices]
                batch_advantages = advantages_ten[batch_indices]
                
                # Get current policy outputs
                _, new_log_probs, entropy, values = self.policy.get_action_and_value(batch_states, batch_actions)
                values = values.squeeze()
                
                # Ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Surrogate Objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy Bonus
                entropy_loss = -torch.mean(entropy)
                
                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss_val += loss.item()
        
        # Approximate average loss per batch (just for logging)
        num_updates = self.k_epochs * (len(states) // self.batch_size)
        self.last_loss = total_loss_val / (num_updates + 1e-8)
        return self.last_loss
