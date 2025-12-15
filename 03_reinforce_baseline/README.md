# Project 3: REINFORCE with Baseline
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
