import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import time
from gymnasium import spaces

class AntEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, render_mode=None):
        super(AntEnv, self).__init__()
        self.render_mode = render_mode
        
        if render_mode == "human":
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Action space: 8 motors (hip, ankle for 4 legs)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        
        # Observation space: 
        # Z (1) + Orn (4) + Joints (8) + JVel (8) + Vel (3) + AngVel (3) = 27
        # We pad to 28 to be safe or just use 28
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)
        
        self.dt = 1./240.
        self.robot = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)
        
        # Load plane
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        
        # Load Ant
        # Usually 'mjcf/ant.xml' works well in pybullet
        # Note: loadMJCF returns a tuple/list of body IDs. We need the first one (the robot).
        self.robot = p.loadMJCF("mjcf/ant.xml", physicsClientId=self.client)[0]

        # Randomize initial state 
        self._obs()
        return self._get_obs(), {}
        
    def step(self, action):
        # Clip actions
        action = np.clip(action, -1.0, 1.0)
        
        force_scale = 30.0
        
        # Apply actions to joints (indices 0..7 likely for Ant XML structure)
        # We apply torque
        p.setJointMotorControlArray(
             self.robot, 
             range(8), 
             p.TORQUE_CONTROL, 
             forces=action * force_scale,
             physicsClientId=self.client
        )
        
        p.stepSimulation(physicsClientId=self.client)
        
        # Get Obs
        obs = self._get_obs()
        
        # Calculate Reward
        # 1. Forward progress (in x direction)
        pos, _ = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client)
        x_pos = pos[0]
        
        if not hasattr(self, 'prev_x'):
             self.prev_x = 0
             
        forward_reward = (x_pos - self.prev_x) / self.dt
        self.prev_x = x_pos
        
        # 2. Survival reward (z height > threshold)
        z_pos = pos[2]
        # Lower survival reward to prevent local optimum of just standing
        survival_reward = 0.05 if z_pos > 0.2 else 0.0
        
        # 3. Control cost
        # Lower control cost to encourage movement
        ctrl_cost = 0.5 * np.square(action).sum()
        
        reward = forward_reward + survival_reward - 1e-4 * ctrl_cost
        
        # Terminal conditions
        terminated = z_pos < 0.2 or z_pos > 1.0 # Fall over
        truncated = False
        
        return obs, reward, terminated, truncated, {}
        
    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot, physicsClientId=self.client)
        
        # Joint states
        joint_states = p.getJointStates(self.robot, range(8), physicsClientId=self.client)
        joint_pos = [s[0] for s in joint_states]
        joint_vel = [s[1] for s in joint_states]
        
        obs_vec = [pos[2]] + list(orn) + joint_pos + joint_vel + list(lin_vel) + list(ang_vel)
        obs_vec = np.array(obs_vec, dtype=np.float32)
        
        # Resize to 28 just in case
        if len(obs_vec) < 28:
             obs_vec = np.pad(obs_vec, (0, 28 - len(obs_vec)))
        else:
             obs_vec = obs_vec[:28]
             
        return obs_vec

    def _obs(self):
        # Reset helper
        self.prev_x = 0
        p.resetBasePositionAndOrientation(self.robot, [0,0,0.75], [0,0,0,1], physicsClientId=self.client)
        
        # Randomize joints to prevent "standing still" local optimum
        # Ant has 8 joints
        for i in range(8):
            # Random position [-0.1, 0.1]
            # Random velocity [-0.1, 0.1]
            pos_noise = np.random.uniform(-0.1, 0.1)
            vel_noise = np.random.uniform(-0.1, 0.1)
            p.resetJointState(self.robot, i, pos_noise, vel_noise, physicsClientId=self.client)
        
    def close(self):
        p.disconnect(self.client)
