import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128) -> None:
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

    def get_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        state_tens = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state_tens)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int = 128) -> None:
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
