#!/usr/bin/env python
"""
Detailed evaluation script for the expert model with debugging information.
"""

import os
import logging
import numpy as np
from stable_baselines3 import PPO
import gymnasium as gym
import gym_battleship
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_expert_detailed(model_path, num_episodes=10, board_size=6, seed=42):
    """Evaluate the expert model with detailed debugging information."""
    # Load the expert model
    logger.info(f"Loading expert model from {model_path}")
    model = PPO.load(model_path)
    
    # Log model information
    logger.info(f"Model observation space: {model.observation_space}")
    logger.info(f"Model action space: {model.action_space}")
    
    # Create environment with the correct board size
    logger.info(f"Creating Battleship environment with board size {board_size}x{board_size}")
    env = gym.make('Battleship-v0', board_size=(board_size, board_size))
    
    # Log environment information
    logger.info(f"Environment observation space: {env.observation_space}")
    logger.info(f"Environment action space: {env.action_space}")
    
    # Metrics to track
    wins = 0
    episodes_with_hits = 0
    total_hits = 0
    total_turns = 0
    
    # Evaluate over specified number of episodes
    for episode in range(num_episodes):
        logger.info(f"===== Episode {episode+1}/{num_episodes} =====")
        obs, info = env.reset(seed=episode+seed)
        done = False
        episode_turns = 0
        episode_hits = 0
        hit_positions = []
        miss_positions = []
        actions_taken = []
        
        # Log ship positions for this episode
        ship_cells = np.sum(env.unwrapped.board_generated)
        logger.info(f"Number of ship cells in this episode: {ship_cells}")
        
        while not done:
            # Predict action
            action, _ = model.predict(obs, deterministic=True)
            
            # Convert to coordinate format - action is a scalar in Discrete action space
            flat_action = int(action)  # Ensure it's an integer
            x, y = flat_action % board_size, flat_action // board_size
            
            # Store action
            actions_taken.append((x, y))
            
            # Log action
            logger.info(f"Turn {episode_turns+1}: Action taken: ({x}, {y}) [flat: {flat_action}]")
            
            # Take step in environment
            prev_obs = obs.copy()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_turns += 1
            
            # Check if it's a hit by comparing observation changes
            is_hit = (obs[0] != prev_obs[0]).any()
            is_miss = (obs[1] != prev_obs[1]).any()
            
            if is_hit:
                episode_hits += 1
                total_hits += 1
                hit_positions.append((x, y))
                logger.info(f"  HIT! at position ({x}, {y})")
            elif is_miss:
                miss_positions.append((x, y))
                logger.info(f"  MISS at position ({x}, {y})")
            else:
                logger.info(f"  REPEAT action at position ({x}, {y})")
        
        # Update metrics
        total_turns += episode_turns
        if episode_hits > 0:
            episodes_with_hits += 1
        
        # Log episode summary
        logger.info(f"Episode summary:")
        logger.info(f"  Turns taken: {episode_turns}")
        logger.info(f"  Hits: {episode_hits} / {ship_cells} ship cells")
        logger.info(f"  Win: {info.get('win', False)}")
        if info.get('win', False):
            wins += 1
        
        # Analyze action distribution
        x_coords = [a[0] for a in actions_taken]
        y_coords = [a[1] for a in actions_taken]
        logger.info(f"  X-coordinate distribution: min={min(x_coords)}, max={max(x_coords)}, mean={np.mean(x_coords):.1f}")
        logger.info(f"  Y-coordinate distribution: min={min(y_coords)}, max={max(y_coords)}, mean={np.mean(y_coords):.1f}")
    
    # Calculate summary metrics
    win_rate = wins / num_episodes
    hit_rate = total_hits / total_turns if total_turns > 0 else 0
    hit_episode_rate = episodes_with_hits / num_episodes
    avg_turns = total_turns / num_episodes
    
    # Print overall results
    logger.info("\n===== Overall Evaluation Results =====")
    logger.info(f"Win Rate: {win_rate:.4f} ({win_rate * 100:.1f}%)")
    logger.info(f"Episodes with at least one hit: {hit_episode_rate:.4f} ({hit_episode_rate * 100:.1f}%)")
    logger.info(f"Overall Hit Rate: {hit_rate:.4f}")
    logger.info(f"Average Turns per Episode: {avg_turns:.2f}")
    
    return {
        "win_rate": win_rate,
        "hit_episode_rate": hit_episode_rate,
        "hit_rate": hit_rate,
        "avg_turns": avg_turns,
    }

if __name__ == "__main__":
    # Path to the trained expert model
    model_path = "expert/best.zip"
    
    # Evaluate expert model with detailed logging
    results = evaluate_expert_detailed(model_path, num_episodes=5, board_size=6)
