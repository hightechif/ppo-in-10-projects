# Project 7: PPO Continuous
## Goal
Solve continuous control tasks (e.g., Pendulum, MountainCarContinuous).
## Key Concepts
- Continuous Action Space
- Gaussian Policy (Output Mean and LogStd)
- `Normal` distribution sampling
## Plan
1. Create `ContinuousPolicy` network.
2. Adapt PPO loss for continuous actions.
3. Train on `Pendulum-v1`.
