# Project 2: Vanilla Policy Gradient (REINFORCE)
## Goal
Implement the simplest policy gradient algorithm from scratch.
## Key Concepts
- Policy Network (MLP)
- Categorical Distribution (Action Sampling)
- Monte-Carlo Returns (Sum of discounted rewards)
- Loss Function: `-log_prob * return`
## Plan
1. Implement `PolicyNetwork` in `policy.py`.
2. Implement training loop in `train.py`.
3. Train on `CartPole-v1`.
