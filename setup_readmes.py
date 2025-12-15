
import os

projects = {
    "01_hello_rl": """# Project 1: Hello RL
## Goal
Understand the Gym API (observations, actions, rewards) and run a random agent.
## Key Concepts
- Environment Cycle (reset -> step -> loop)
- Observation Space (Box/Discrete)
- Action Space
- Rendering
## Plan
1. Install `gymnasium`.
2. Create `main.py` that loads `CartPole-v1`.
3. Run a loop with `env.action_space.sample()`.
""",
    "02_reinforce": """# Project 2: Vanilla Policy Gradient (REINFORCE)
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
""",
    "03_reinforce_baseline": """# Project 3: REINFORCE with Baseline
## Goal
Reduce variance in gradients by subtracting a baseline (Value Function).
## Key Concepts
- Value Network (Critic)
- Advantage Estimate: `Return - Value`
- Two-network architecture
## Plan
1. Add `ValueNetwork`.
2. Update loss to include value loss.
3. Compare performance with Project 2.
""",
    "04_a2c": """# Project 4: Advantage Actor-Critic (A2C)
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
""",
    "05_ppo_discrete": """# Project 5: PPO Discrete (The Core)
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
""",
    "06_ppo_gae": """# Project 6: PPO with Generalized Advantage Estimation (GAE)
## Goal
Improve advantage estimation using GAE(lambda).
## Key Concepts
- Lambda parameter (trade-off between bias and variance)
- GAE Calculation loop
## Plan
1. Implement `calculate_gae` function.
2. Integrate into PPO buffer.
3. Compare training curves.
""",
    "07_ppo_continuous": """# Project 7: PPO Continuous
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
""",
    "08_ppo_viz": """# Project 8: Vectorized Environments
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
""",
    "09_ppo_pybullet": """# Project 9: Introduction to PyBullet
## Goal
Apply PPO to a 3D Physics simulation (Robotics).
## Key Concepts
- PyBullet Physics Engine
- `AntBulletEnv-v0` or `HopperBulletEnv-v0`
- High-dimensional continuous control
## Plan
1. Install `pybullet`.
2. Tune PPO hyperparameters for robotics (larger batch size, etc.).
3. Train an Ant to walk.
""",
    "10_ppo_advanced": """# Project 10: Advanced PPO Features
## Goal
Implement "Pro-level" tricks that make PPO robust.
## Key Concepts
- Observation Normalization (Running Mean/Std)
- Reward Scaling
- Gradient Clipping
- Learning Rate Annealing
- Orthogonal Weight Initialization
## Plan
1. Implement `RunningMeanStd`.
2. Add all tricks to the PPO agent.
3. Final "Robust" Benchmark.
"""
}

for folder, content in projects.items():
    with open(f"{folder}/README.md", "w") as f:
        f.write(content)

print("READMEs updated.")
