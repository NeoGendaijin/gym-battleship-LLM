#!/usr/bin/env python
"""
Quick test of the updated expert training script with a small number of timesteps.
This allows us to verify that the improvements address the looping behavior issues.
"""

import os
import logging
import numpy as np
from stable_baselines3 import PPO
import gymnasium as gym
import gym_battleship
import subprocess
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_quick_model(timesteps=10000, board_size=6, seed=42):
    """Train a model with a small number of timesteps for quick testing."""
    
    # Use a different save path to avoid overwriting the main model
    save_path = "expert/quick_test.zip"
    
    logger.info(f"Quick training with {timesteps} timesteps on {board_size}x{board_size} board")
    
    # Run the training command with reduced timesteps
    cmd = [
        "python", "expert/train_expert.py",
        f"--total-timesteps={timesteps}",
        f"--save-path={save_path}",
        f"--board-size={board_size}",
        f"--seed={seed}"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    logger.info(f"Training completed. Model saved to {save_path}")
    return save_path

def evaluate_model(model_path, board_size=6, num_episodes=5):
    """Run a simple evaluation of the trained model."""
    
    logger.info(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    
    # Create environment
    env = gym.make('Battleship-v0', board_size=(board_size, board_size))
    
    # Track metrics
    total_turns = 0
    wins = 0
    episodes_with_repeats = 0
    max_repeats_per_episode = 0
    
    for episode in range(num_episodes):
        logger.info(f"\n===== Episode {episode+1}/{num_episodes} =====")
        
        # Reset environment
        obs, _ = env.reset(seed=episode+100) # Different seeds from training
        done = False
        episode_turns = 0
        action_counts = {}
        consecutive_repeats = 0
        max_consecutive_repeats = 0
        
        # Show ship positions
        ship_cells = np.sum(env.unwrapped.board_generated)
        logger.info(f"Ship cells on board: {ship_cells}")
        
        # Display board
        board_repr = ""
        for y in range(board_size):
            row = ""
            for x in range(board_size):
                if env.unwrapped.board_generated[x, y]:
                    row += "S "
                else:
                    row += ". "
            board_repr += row + "\n"
        logger.info(f"Board layout:\n{board_repr}")
        
        while not done and episode_turns < 100:  # Limit to 100 turns max
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Convert to coordinate
            flat_action = int(action)
            x, y = flat_action % board_size, flat_action // board_size
            coords = (x, y)
            
            # Check for repeats
            action_counts[coords] = action_counts.get(coords, 0) + 1
            
            if action_counts[coords] > 1:
                consecutive_repeats += 1
            else:
                consecutive_repeats = 0
                
            max_consecutive_repeats = max(max_consecutive_repeats, consecutive_repeats)
            
            # Log action
            repeat_marker = " (REPEAT)" if action_counts[coords] > 1 else ""
            logger.info(f"Turn {episode_turns+1}: Action ({x}, {y}){repeat_marker}")
            
            # Take step
            prev_obs = obs.copy()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_turns += 1
            
            # Determine outcome
            is_hit = (obs[0] != prev_obs[0]).any()
            is_miss = (obs[1] != prev_obs[1]).any()
            is_repeat = not (is_hit or is_miss)
            
            # Log outcome
            if is_hit:
                logger.info(f"  HIT! Reward: {reward}")
            elif is_miss:
                logger.info(f"  MISS. Reward: {reward}")
            else:
                logger.info(f"  REPEAT. Reward: {reward}")
        
        # Update metrics
        total_turns += episode_turns
        
        # Count unique vs. repeated actions
        unique_actions = len(action_counts)
        repeated_actions = episode_turns - unique_actions
        repeats_ratio = repeated_actions / episode_turns if episode_turns > 0 else 0
        
        if repeated_actions > 0:
            episodes_with_repeats += 1
            
        max_repeats_per_episode = max(max_repeats_per_episode, max_consecutive_repeats)
        
        # Log episode summary
        logger.info(f"\nEpisode {episode+1} Summary:")
        logger.info(f"  Turns: {episode_turns}")
        logger.info(f"  Unique actions: {unique_actions}/{episode_turns} ({unique_actions/episode_turns:.1%})")
        logger.info(f"  Repeated actions: {repeated_actions}/{episode_turns} ({repeats_ratio:.1%})")
        logger.info(f"  Max consecutive repeats: {max_consecutive_repeats}")
        logger.info(f"  Win: {info.get('win', False)}")
        
        if info.get('win', False):
            wins += 1
    
    # Log overall results
    avg_turns = total_turns / num_episodes
    win_rate = wins / num_episodes
    repeat_episode_rate = episodes_with_repeats / num_episodes
    
    logger.info("\n===== Overall Evaluation Results =====")
    logger.info(f"Win Rate: {win_rate:.2f} ({wins}/{num_episodes})")
    logger.info(f"Average Turns: {avg_turns:.1f}")
    logger.info(f"Episodes with repeated actions: {repeat_episode_rate:.2f} ({episodes_with_repeats}/{num_episodes})")
    logger.info(f"Max consecutive repeats in any episode: {max_repeats_per_episode}")
    
    return {
        "win_rate": win_rate,
        "avg_turns": avg_turns,
        "repeat_episode_rate": repeat_episode_rate,
        "max_repeats": max_repeats_per_episode
    }

def main():
    # Train a quick model with fewer timesteps
    timesteps = 10000  # Small number of training steps for quick testing
    board_size = 6
    
    try:
        # Create necessary directories
        os.makedirs("expert", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
        
        # Train model
        model_path = train_quick_model(timesteps=timesteps, board_size=board_size)
        
        # Evaluate model
        results = evaluate_model(model_path, board_size=board_size, num_episodes=3)
        
        # Return success
        return 0
    
    except Exception as e:
        logger.error(f"Error in quick test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())
