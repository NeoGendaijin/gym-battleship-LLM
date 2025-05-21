#!/usr/bin/env python
"""
Minimal evaluation script for the expert model.
This script is simplified to just check if the model is producing sensible actions.
"""

import numpy as np
from stable_baselines3 import PPO
import gymnasium as gym
import gym_battleship

# Create environment with the correct board size
board_size = 6
env = gym.make('Battleship-v0', board_size=(board_size, board_size))

# Load the expert model
model_path = "expert/best.zip"
print(f"Loading expert model from {model_path}")
model = PPO.load(model_path)

# Print model and environment spaces
print(f"Model observation space: {model.observation_space}")
print(f"Model action space: {model.action_space}")
print(f"Environment observation space: {env.observation_space}")
print(f"Environment action space: {env.action_space}")

# Reset environment
obs, info = env.reset(seed=42)
print(f"Starting evaluation with board size {board_size}x{board_size}")

# Number of ship cells
ship_cells = np.sum(env.unwrapped.board_generated)
print(f"Number of ship cells on the board: {ship_cells}")

# Display the board
print("Ship positions on the board:")
for y in range(board_size):
    row = ""
    for x in range(board_size):
        if env.unwrapped.board_generated[x, y]:
            row += "S "
        else:
            row += ". "
    print(row)

# Take 10 actions
for turn in range(10):
    # Predict action
    action, _ = model.predict(obs, deterministic=True)
    flat_action = int(action)
    x, y = flat_action % board_size, flat_action // board_size
    print(f"Turn {turn+1}: Action ({x}, {y}) [flat: {flat_action}]")
    
    # Take step
    prev_obs = obs.copy()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # Check if it's a hit
    is_hit = (obs[0] != prev_obs[0]).any()
    is_miss = (obs[1] != prev_obs[1]).any()
    
    if is_hit:
        print(f"  HIT at position ({x}, {y})")
    elif is_miss:
        print(f"  MISS at position ({x}, {y})")
    else:
        print(f"  REPEAT action at position ({x}, {y})")
    
    if done:
        print("Game completed!")
        break

print("Evaluation completed.")
