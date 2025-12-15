# Project 8: Vectorized Environments
## Goal
Speed up data collection using multiple parallel environments.
## Key Concepts
- `gymnasium.vector.AsyncVectorEnv`
- Processing batch of observations/rewards
- Syncing updates
## Plan
1. Update PPO to handle `(num_envs, obs_dim)` inputs.
2. Measure speedup vs single environment.
3. Visualize 4 envs at once.
