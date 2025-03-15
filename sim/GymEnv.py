import gym
from gym import spaces
import numpy as np
from FuzbAISim import FuzbAISim
import time

class FoosballEnv(gym.Env):
    def __init__(self, debug=False):
        super(FoosballEnv, self).__init__()
        self.sim = FuzbAISim(debug=debug)
        self.episode_reward = 0  # Track episode rewards

        # State space: Ball (x, y, vx, vy) + Rods (positions & angles)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -1, -1] + [0]*8 + [-32]*8),
            high=np.array([1210, 700, 1, 1] + [1]*8 + [32]*8),
            dtype=np.float32
        )

        # Continuous action space: (translation, rotation, velocity)
        self.action_space = spaces.Box(low=-1, high=1, shape=(4, 3), dtype=np.float32)

        # Store previous rod positions & angles to prevent shaking
        self.prev_rod_positions = np.zeros(4)
        self.prev_rod_angles = np.zeros(4)

        self.sim.run()

    def _get_obs(self):
        """
        Returns the current state of the environment, ensuring the expected shape.
        """
        try:
            camData = self.sim.getCameraDict(1)

            if "camData" not in camData or not camData["camData"]:
                print("[ERROR] Camera data is missing! Returning default observation.")
                return np.zeros(self.observation_space.shape, dtype=np.float32)

            bx, by = camData["camData"][0]["ball_x"], camData["camData"][0]["ball_y"]
            vx, vy = camData["camData"][0]["ball_vx"], camData["camData"][0]["ball_vy"]
            rod_positions = camData["camData"][0]["rod_position_calib"][:8]  
            rod_angles = camData["camData"][0]["rod_angle"][:8]  

            obs = np.array([bx, by, vx, vy] + rod_positions + rod_angles, dtype=np.float32)

            if obs.shape != self.observation_space.shape:
                print(f"[ERROR] Observation shape mismatch! Expected {self.observation_space.shape}, got {obs.shape}.")
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)

            return obs
        
        except Exception as e:
            print(f"[ERROR] _get_obs() crashed: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)



    def _compute_reward(self):
        """
        Computes the reward for the agent based on gameplay and movement smoothness.
        """
        camData = self.sim.getCameraDict(1)
        bx, by, vx, vy = camData["camData"][0]["ball_x"], camData["camData"][0]["ball_y"], camData["camData"][0]["ball_vx"], camData["camData"][0]["ball_vy"]

        reward = 0.1 # Default reward

        # 1️⃣ **Reward for hitting the ball**
        if abs(vx) > 0.2 or abs(vy) > 0.2:
            reward += 0.1  

        # 2️⃣ **Penalty for excessive shaking**
        current_positions = np.array(camData["camData"][0]["rod_position_calib"][:4])  # First 4 rods
        current_angles = np.array(camData["camData"][0]["rod_angle"][:4])  # First 4 rod angles

        movement_penalty = np.sum(np.abs(current_positions - self.prev_rod_positions))  # How much rods moved
        rotation_penalty = np.sum(np.abs(current_angles - self.prev_rod_angles))  # How much rods rotated

        # If shaking is detected, apply penalty
        if movement_penalty > 0.2:  # Threshold to ignore small movements
            reward -= movement_penalty * 0.5  # Reduce shaking
        if rotation_penalty > 5:  # Too much rod rotation
            reward -= rotation_penalty * 0.1  # Penalize excessive rotation

        # Save previous values for next step comparison
        self.prev_rod_positions = current_positions
        self.prev_rod_angles = current_angles

        # 3️⃣ **Penalty for shooting towards own goal**
        if vx < -0.2:  # Ball moving left (bad for red team)
            reward -= 0.5
        elif vx > 0.2:  # Ball moving right (bad for blue team)
            reward -= 0.5

        # 4️⃣ **Big Penalty if opponent scores a goal**
        if self.sim.score[1] > 0:  # Blue scored against Red
            reward -= 5.0
            self.sim.score = [0, 0]  # Reset the score
        elif self.sim.score[0] > 0:  # Red scored against Blue
            reward -= 5.0
            self.sim.score = [0, 0]  # Reset the score

        # 5️⃣ **Reward for scoring a goal**
        if self.sim.score[0] > 0:  # Red scores
            reward += 10.0
            self.sim.score = [0, 0]  
        elif self.sim.score[1] > 0:  # Blue scores
            reward += 10.0
            self.sim.score = [0, 0]  

        return reward

    def step(self, action):
        """
        Apply continuous actions for smooth movement and shooting.
        """
        commands = []
        for i, rod in enumerate(self.sim.p1.rods):
            trans_dir, rot_dir, velocity = action[i]

            # Smooth movement by limiting sudden jumps
            trans_smooth = np.clip(self.prev_rod_positions[i] + trans_dir * 0.1, 0, 1)
            rot_smooth = np.clip(self.prev_rod_angles[i] + rot_dir * 0.1, -1, 1)

            cmd = {
                # "driveID": i + 1,  
                "driveID": rod + 1,  
                "rotationTargetPosition": rot_smooth * 0.75,  
                "rotationVelocity": np.clip(velocity * 1.5, 0.1, 2.0),  
                "translationTargetPosition": trans_smooth,  
                "translationVelocity": np.clip(velocity * 1.5, 0.1, 2.0)  
            }
            commands.append(cmd)

        # Store previous values for smooth movement
        self.prev_rod_positions = [cmd["translationTargetPosition"] for cmd in commands]
        self.prev_rod_angles = [cmd["rotationTargetPosition"] for cmd in commands]

        self.sim.motorCommandsExternal1 = commands  
        self.sim.motorCommandsExternal2 = commands  

        time.sleep(0.02)

        obs = self._get_obs()
        reward = self._compute_reward()
        self.episode_reward += reward  # Track total rewards in the episode
        return obs, reward, False, {}


    def reset(self):
        """
        Resets the environment and ensures it returns a valid observation.
        """
        if not hasattr(self, "sim"):  # Prevent multiple instances
            print("[DEBUG] Creating a new FuzbAISim instance...")
            self.sim = FuzbAISim()
        # else:
        #     print("[DEBUG] Resetting the existing simulation...")
        #     self.sim.reset()

        obs = self._get_obs()  # Get initial state

        if obs is None:
            print("[ERROR] _get_obs() returned None! Replacing with zeros.")
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)  # Default fallback

        print(f"[DEBUG] Reset() returning observation with shape {obs.shape}: {obs}")

        self.episode_reward = 0  # Reset reward tracking
        
        return obs  # Must return an observation
