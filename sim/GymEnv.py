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

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8, 3), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

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



    def _compute_reward(self, team):
        """
        Computes the reward for the agent based on gameplay and movement smoothness.
        """
        camData = self.sim.getCameraDict(1)
        bx, by, vx, vy = camData["camData"][0]["ball_x"], camData["camData"][0]["ball_y"], camData["camData"][0]["ball_vx"], camData["camData"][0]["ball_vy"]

        reward = 0  

        # Reward for movement (encourages active play)
        avg_velocity = np.mean([abs(vx), abs(vy)])
        reward += avg_velocity * 0.2  

        # Reward for positioning rods near the ball
        for rod in self.sim.p1.rods if team == "red" else self.sim.p2.rods:
            rod_y = 100 * rod
            reward += 0.1 - abs(by - rod_y) / 350  

        # Reward for moving the ball toward the opponent’s goal
        if team == "red" and vx > 0:  
            reward += 0.5  
        elif team == "blue" and vx < 0:
            reward += 0.5  

        # Reward for making contact with the ball
        if self.sim.check_ball_contact():
            reward += 1.0  

        # Small penalty for stopping too much
        if avg_velocity < 0.05:
            reward -= 0.1  

        return float(reward)

    def step(self, action):
        """
        Executes learned continuous actions for both players.
        """
        # print(f"[DEBUG] Received RL action: {action} (Shape: {action.shape})")

        if action is None or not isinstance(action, np.ndarray):
            print("[ERROR] RL action is None or not a NumPy array!")
            action = np.zeros((8, 3))  

        if action.shape != (8, 3):  # Expecting 8 actions (4 for each player)
            print(f"[ERROR] Action shape mismatch! Expected (8,3), got {action.shape}")
            action = np.zeros((8, 3))  

        camera_data = self.sim.getCameraDict(1)

        actions_p1 = action[:4]
        actions_p2 = action[4:]

        commands_p1 = self.sim.p1.process_data(camera_data, actions_p1)
        commands_p2 = self.sim.p2.process_data(camera_data, actions_p2)

        # ✅ Assign the RL-generated motor commands
        self.sim.motorCommandsExternal1 = commands_p1  
        self.sim.motorCommandsExternal2 = commands_p2  

        # print(f"[DEBUG] RL Actions Applied to motorCommandsExternal1: {self.sim.motorCommandsExternal1}")
        # print(f"[DEBUG] RL Actions Applied to motorCommandsExternal2: {self.sim.motorCommandsExternal2}")

        time.sleep(0.02)

        # ✅ Compute separate rewards for both players
        reward_p1 = self._compute_reward("red")
        reward_p2 = self._compute_reward("blue")

        # ✅ Return rewards as a NumPy array instead of a list
        # reward = np.array([reward_p1, reward_p2], dtype=np.float32).flatten()  # Fix for ValueError
        total_reward = float(reward_p1 + reward_p2) 

        done = False  

        return self._get_obs(), total_reward, done, {}


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
