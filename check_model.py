#!/usr/bin/env python
"""
Simple script to check if the expert model was trained with the correct parameters.
"""
from stable_baselines3 import PPO
import gymnasium as gym
import gym_battleship
import numpy as np

def main():
    # Path to the trained expert model
    model_path = "expert/best.zip"
    
    # Load model
    print(f"Loading expert model from {model_path}")
    model = PPO.load(model_path)
    
    # Display model information
    print(f"Model observation space: {model.observation_space}")
    print(f"Model action space: {model.action_space}")
    
    # Create environment with 6x6 board size
    board_size = 6
    env = gym.make('Battleship-v0', board_size=(board_size, board_size))
    
    # Display environment information
    print(f"Environment observation space: {env.observation_space}")
    print(f"Environment action space: {env.action_space}")
    
    # Check if observation spaces match
    obs_match = model.observation_space.shape == env.observation_space.shape
    print(f"Observation spaces match: {obs_match}")
    
    # Check if action spaces match
    action_match = model.action_space.n == env.action_space.n
    print(f"Action spaces match: {action_match}")
    
    # Try a single prediction
    obs, _ = env.reset(seed=42)
    action, _ = model.predict(obs, deterministic=True)
    
    print(f"Model predicted action: {action} (type: {type(action)})")
    
    if isinstance(action, np.ndarray):
        print(f"Action shape: {action.shape}")
        print(f"Action value: {action}")
    else:
        print(f"Action is a scalar value: {action}")
        x, y = action % board_size, action // board_size
        print(f"Corresponding board position: ({x}, {y})")
    
    # Test if the action is valid
    try:
        _, _, _, _, _ = env.step(action)
        print("Action is valid in the environment")
    except Exception as e:
        print(f"Error when taking action: {e}")
    
    print("Model check completed")

if __name__ == "__main__":
    main()
