#!/usr/bin/env python
"""
Simple script to test the trained model on the Battleship environment.
"""

from stable_baselines3 import PPO
import gymnasium as gym
import gym_battleship
import numpy as np

def main():
    # Load model and create environment
    model_path = "expert/quick_test.zip"
    board_size = 6
    
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    
    env = gym.make('Battleship-v0', board_size=(board_size, board_size))
    
    # Reset environment
    obs, _ = env.reset(seed=42)
    
    # Print model and environment info
    print(f"Model info: {model}")
    print(f"Observation shape: {obs.shape}")
    
    # Prepare for tracking actions
    action_counts = {}
    
    # Make predictions and take steps
    for i in range(20):
        action, _ = model.predict(obs, deterministic=True)
        
        # Convert to coordinate
        flat_action = int(action)
        x, y = flat_action % board_size, flat_action // board_size
        coords = (x, y)
        
        # Count repeats
        action_counts[coords] = action_counts.get(coords, 0) + 1
        repeat_marker = " (REPEAT)" if action_counts[coords] > 1 else ""
        
        print(f"Step {i+1}, Action: ({x}, {y}) [flat: {action}]{repeat_marker}")
        
        # Take a step
        prev_obs = obs.copy()
        obs, reward, done, truncated, info = env.step(action)
        
        # Determine outcome
        is_hit = (obs[0] != prev_obs[0]).any()
        is_miss = (obs[1] != prev_obs[1]).any()
        
        if is_hit:
            outcome = "HIT"
        elif is_miss:
            outcome = "MISS"
        else:
            outcome = "REPEAT"
            
        print(f"  Outcome: {outcome}, Reward: {reward}, Done: {done}")
        
        if done:
            print("Game completed!")
            break
    
    # Calculate statistics
    total_steps = i + 1
    unique_actions = len(action_counts)
    repeated_actions = total_steps - unique_actions
    
    print("\nSummary:")
    print(f"Total steps: {total_steps}")
    print(f"Unique actions: {unique_actions}/{total_steps} ({unique_actions/total_steps:.1%})")
    print(f"Repeated actions: {repeated_actions}/{total_steps} ({repeated_actions/total_steps:.1%})")

if __name__ == "__main__":
    main()
