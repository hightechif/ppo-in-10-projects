# Project 5: PPO Discrete (The Core)
## Goal
Implement Proximal Policy Optimization with the Clipped Surrogate Objective.
## Key Concepts
- Importance Sampling Ratio (`prob / old_prob`)
- Clipping (`clamp(ratio, 1-eps, 1+eps)`)
- Multiple epochs of updates on collected data
## Plan
1. Implement `PPOAgent` class.
2. Implement the PPO Loss function.
3. Train on `CartPole-v1` (should be more stable).
