import gymnasium as gym
import time
from typing import Any

def main() -> None:
    print("Initializing CartPole-v1...")
    # Create the environment. 'render_mode="human"' will open a window.
    # Use 'render_mode="rgb_array"' if you want to capture frames without a window.
    try:
        env = gym.make("CartPole-v1", render_mode="human")
    except Exception as e:
        print(f"Could not open window (headless?): {e}")
        env = gym.make("CartPole-v1") # Fallback

    # Reset the environment
    observation, info = env.reset(seed=42)

    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    print("Starting simulation for 100 steps...")

    total_reward: float = 0.0
    for step in range(100):
        # Sample a random action
        action = env.action_space.sample()

        # Step the environment
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        # Render is handled by the environment in 'human' mode automatically
        # Just adding a small sleep so it's not too fast for the user to see
        time.sleep(0.05)

        if terminated or truncated:
            print(f"Episode finished after {step + 1} steps. Total Reward: {total_reward}")
            observation, info = env.reset()
            total_reward = 0

    print("Simulation finished.")
    env.close()

if __name__ == "__main__":
    main()
