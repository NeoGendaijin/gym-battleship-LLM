"""
Train a PPO teacher model for Battleship (grid-size=6) using Stable-Baselines3.
The resulting model will be used as the teacher for KL-based distillation.
"""

import os
import gymnasium as gym
import gym_battleship
from stable_baselines3 import PPO

def main():
    grid_size = 6
    env_id = "Battleship-v0"
    total_timesteps = 100_000  # You can increase for better teacher
    save_path = os.path.join(os.path.dirname(__file__), "../checkpoints/teacher_ppo_grid6.zip")

    # Create environment
    env = gym.make(env_id, board_size=(grid_size, grid_size))

    # Instantiate PPO agent
    model = PPO("MlpPolicy", env, verbose=1)

    # Train
    model.learn(total_timesteps=total_timesteps)

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Teacher PPO model saved to: {save_path}")

if __name__ == "__main__":
    main()
