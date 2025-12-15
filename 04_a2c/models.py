import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128) -> None:
        super(ActorCritic, self).__init__()
        # Shared Feature Extractor
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        
        # Actor Head (Policy)
        self.actor = nn.Linear(hidden_size, action_dim)
        
        # Critic Head (Value)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        # In a real shared network, we might want detached gradients for one head to not mess up the other,
        # but for simple CartPole, fully shared is fine.
        return x

    def get_action_and_value(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        state_tens = torch.from_numpy(state).float().unsqueeze(0)
        
        features = self.forward(state_tens)
        
        # Actor
        logits = self.actor(features)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        
        # Critic
        value = self.critic(features)
        
        return action.item(), dist.log_prob(action), value, dist.entropy()
