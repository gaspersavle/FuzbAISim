from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from GymEnv import FoosballEnv

class RewardLogger(BaseCallback):
    def __init__(self, check_freq=100, verbose=1):
        super(RewardLogger, self).__init__(verbose)
        self.check_freq = check_freq
        self.rewards = []

    def _on_step(self):
        """
        Logs training progress and checks if the callback is being called.
        """
        print(f"[DEBUG] Step: {self.n_calls} - Callback is running!")  # Debug print

        # Get rewards from the environment
        rewards = self.training_env.get_attr("episode_reward", 0)
        print(f"[DEBUG] Rewards: {rewards}")  # See if rewards are being collected

        self.rewards.append(np.mean(rewards))

        if self.n_calls % self.check_freq == 0:
            mean_reward = np.mean(self.rewards[-10:])  # Average last 10 episodes
            print(f"Step: {self.n_calls}, Avg Reward: {mean_reward:.2f}")

        return True  # Continue training


env = DummyVecEnv([lambda: FoosballEnv(debug=True)])

# Force TensorBoard logging
new_logger = configure("./logs/", ["stdout", "tensorboard"])

# Initialize SAC model
model = SAC("MlpPolicy", env, verbose=1)
model.set_logger(new_logger)  # Force logs

# Attach debugging callback
callback = RewardLogger(check_freq=100)

# Start training
model.learn(total_timesteps=5000, callback=callback, log_interval=10)  # Force logs every 10 steps

# Save model
model.save("foosball_agent_continuous")
