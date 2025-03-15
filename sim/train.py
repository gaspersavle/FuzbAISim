import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from GymEnv import FoosballEnv

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEBUG] Using device: {device}")

# Wrap environment
env = DummyVecEnv([lambda: FoosballEnv(debug=True)])

print(f"[DEBUG] Action space shape: {env.action_space.shape}")
print(f"[DEBUG] Observation space shape: {env.observation_space.shape}")

# Custom callback to monitor training
class TrainingMonitor(BaseCallback):
    def __init__(self, check_freq=100, verbose=1):
        super(TrainingMonitor, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if "rewards" in self.locals:
            self.episode_rewards.append(self.locals["rewards"])

        if self.n_calls % self.check_freq == 0:
            mean_reward = np.mean(self.episode_rewards[-10:])
            print(f"[TRAINING] Step {self.n_calls} | Avg Reward (Last 10): {mean_reward:.2f}")

        return True

# ✅ Use **one** SAC model that learns for **both** players (instead of separate models)
model = SAC("MlpPolicy", env, verbose=1, device=device)

# Attach callback for monitoring
callback = TrainingMonitor(check_freq=100)

# Train model
print("[TRAINING] Starting training...")
model.learn(total_timesteps=200000, callback=callback)

# Save trained model
model.save("foosball_agent")

# ✅ Test the trained model for 100 steps
obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)  # Get actions for both players

    print(f"[DEBUG] RL Model Generated Actions: {action} (Shape: {action.shape})")

    obs, rewards, dones, info = env.step(action)  # Pass combined actions into environment