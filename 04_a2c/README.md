# Project 4: Advantage Actor-Critic (A2C)
## Goal
Transition from Monte-Carlo (full episode) to Bootstrapping (n-step or 1-step).
## Key Concepts
- Bootstrapping (TD Learning)
- Online updates (update every n steps, don't wait for episode end)
- Actor-Critic Architecture (Shared or Separate)
## Plan
1. Implement `models.py` with Actor and Critic.
2. Implement rollout buffer for n steps.
3. Train on `CartPole-v1`.
