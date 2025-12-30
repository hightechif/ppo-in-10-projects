import torch
import numpy as np
import time
import pybullet as p
from ppo import PPO
from ant_env import AntEnv

def visualize() -> None:
    # Use render_mode='human' for GUI
    env = AntEnv(render_mode="human")
    
    obs_dim = env.observation_space.shape[0] # type: ignore
    action_dim = env.action_space.shape[0] # type: ignore
    
    agent = PPO(obs_dim, action_dim)
    
    try:
        agent.policy.load_state_dict(torch.load("09_ppo_pybullet/model.pt", map_location="cpu"))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: 09_ppo_pybullet/model.pt not found. Train the agent first.")
        # Just run random agent if model not found for testing env
        print("Running with random agent for demonstration...")
    
    agent.policy.eval()

    state, _ = env.reset()
    
    print("Starting visualization...")
    print("Use the mouse to rotate/zoom. Camera will follow the Ant.")
    
    # Run for a few steps
    for step_i in range(2000):
        # We need to reshape state for PPO (it expects flattened input usually or handle in select_action)
        # Our select_action handles numpy array input directly.
        action, _, _ = agent.select_action(state)
        
        state, reward, terminated, truncated, _ = env.step(action)
        
        # Camera Follow
        # We need to access the robot position from the env
        # env.robot is the body ID
        if hasattr(env, 'robot') and hasattr(env, 'client'):
             pos, _ = p.getBasePositionAndOrientation(env.robot, physicsClientId=env.client)
             p.resetDebugVisualizerCamera(
                 cameraDistance=2.0, 
                 cameraYaw=50, 
                 cameraPitch=-35, 
                 cameraTargetPosition=pos,
                 physicsClientId=env.client
             )
        
        # Slow down slightly for viewing
        time.sleep(1./60.)
        
        if terminated or truncated:
            print(f"Episode finished. Resetting.")
            state, _ = env.reset()
            
    env.close()

if __name__ == "__main__":
    visualize()
