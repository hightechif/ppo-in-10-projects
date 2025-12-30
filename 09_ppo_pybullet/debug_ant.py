from ant_env import AntEnv
import time

def debug():
    print("Creating Env...")
    env = AntEnv(render_mode=None)
    print("Env Created. Resetting...")
    obs, _ = env.reset()
    print(f"Reset Done. Obs shape: {obs.shape}")
    
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        if i % 10 == 0:
            print(f"Step {i}: Reward {reward:.4f}, Terminated {terminated}")
        if terminated or truncated:
            print("Episode done.")
            obs, _ = env.reset()
            
    print("Debug finished.")

if __name__ == "__main__":
    debug()
