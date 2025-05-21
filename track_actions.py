#!/usr/bin/env python
"""
Script to track actions taken by the expert model over a few episodes.
This will help verify if the model is functioning as expected.
"""
from stable_baselines3 import PPO
import gymnasium as gym
import gym_battleship
import numpy as np
import os
import io
from contextlib import redirect_stdout

def evaluate_one_episode(model, env, episode_num, board_size=6, seed=None):
    """Evaluate the model for one episode and print detailed information."""
    # Capture stdout to a string buffer
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print(f"\n===== Episode {episode_num} =====")
        
        # Reset environment
        if seed is not None:
            obs, info = env.reset(seed=seed+episode_num)
        else:
            obs, info = env.reset()
        
        # Display board information
        ship_cells = np.sum(env.unwrapped.board_generated)
        print(f"Number of ship cells: {ship_cells}")
        
        # Display board
        print("Ship positions on the board:")
        for y in range(board_size):
            row = ""
            for x in range(board_size):
                if env.unwrapped.board_generated[x, y]:
                    row += "S "
                else:
                    row += ". "
            print(row)
        
        # Play the game
        done = False
        episode_turns = 0
        episode_hits = 0
        actions_taken = set()  # Track unique actions
        
        while not done and episode_turns < 100:  # Limit to 100 turns for safety
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            flat_action = int(action)  # Convert to int
            x, y = flat_action % board_size, flat_action // board_size
            
            # Check if action was already taken
            is_repeat = (x, y) in actions_taken
            actions_taken.add((x, y))
            
            # Print action
            print(f"Turn {episode_turns+1}: Action ({x}, {y}) [flat: {flat_action}]")
            
            # Take step in environment
            prev_obs = obs.copy()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_turns += 1
            
            # Check if it's a hit
            is_hit = (obs[0] != prev_obs[0]).any()
            is_miss = (obs[1] != prev_obs[1]).any()
            
            if is_hit:
                episode_hits += 1
                print(f"  HIT at position ({x}, {y})")
            elif is_miss:
                print(f"  MISS at position ({x}, {y})")
            else:
                print(f"  REPEAT action at position ({x}, {y})")
        
        # Print episode summary
        print(f"\nEpisode summary:")
        print(f"  Turns taken: {episode_turns}")
        print(f"  Hits: {episode_hits} / {ship_cells} ship cells")
        print(f"  Unique actions: {len(actions_taken)} / {episode_turns} turns")
        print(f"  Repeated actions: {episode_turns - len(actions_taken)}")
        print(f"  Win: {info.get('win', False)}")
    
    # Return the captured output
    return buffer.getvalue()

def main():
    # Path to the trained expert model
    model_path = "expert/best.zip"
    
    # Load model
    print(f"Loading expert model from {model_path}")
    model = PPO.load(model_path)
    
    # Create environment with 6x6 board size
    board_size = 6
    env = gym.make('Battleship-v0', board_size=(board_size, board_size))
    
    # Check if spaces match
    print(f"Model observation space: {model.observation_space}")
    print(f"Environment observation space: {env.observation_space}")
    print(f"Model action space: {model.action_space}")
    print(f"Environment action space: {env.action_space}")
    
    # Evaluate for 3 episodes
    output_file = "expert_tracking.txt"
    with open(output_file, "w") as f:
        # Write header
        f.write("Expert Model Tracking Report\n")
        f.write("===========================\n\n")
        
        # Evaluate episodes
        for episode in range(3):
            episode_output = evaluate_one_episode(model, env, episode+1, board_size, seed=42)
            f.write(episode_output)
            # Print part of the output to console
            print(f"Episode {episode+1} completed")
        
        # Write footer
        f.write("\nTracking completed. Check for patterns in model behavior.\n")
    
    print(f"\nTracking results written to {output_file}")

if __name__ == "__main__":
    main()
