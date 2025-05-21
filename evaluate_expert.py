#!/usr/bin/env python
"""
Custom script to evaluate the expert model with the correct board size.
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

def evaluate_expert(model_path, num_episodes=50, board_size=6, seed=42):
    """Evaluate the expert model on the Battleship environment with the specified board size."""
    # Load the expert model
    logger.info(f"Loading expert model from {model_path}")
    model = PPO.load(model_path)
    
    # Create environment with the correct board size
    logger.info(f"Creating Battleship environment with board size {board_size}x{board_size}")
    env = gym.make('Battleship-v0', board_size=(board_size, board_size))
    
    # Metrics to track
    wins = 0
    total_turns = 0
    turn_counts = []
    hit_rates = []
    
    # Evaluate over specified number of episodes
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        obs, info = env.reset(seed=episode+seed)
        done = False
        episode_turns = 0
        hits = 0
        
        while not done:
            # Predict action
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_turns += 1
            
            # Track metrics
            if info.get("hit", False):
                hits += 1
        
        # Update metrics
        if info.get("win", False):
            wins += 1
        
        total_turns += episode_turns
        turn_counts.append(episode_turns)
        hit_rate = hits / episode_turns if episode_turns > 0 else 0
        hit_rates.append(hit_rate)
    
    # Calculate summary metrics
    win_rate = wins / num_episodes
    avg_turns = np.mean(turn_counts)
    std_turns = np.std(turn_counts)
    avg_hit_rate = np.mean(hit_rates)
    
    # Print results
    logger.info("===== Evaluation Results =====")
    logger.info(f"Win Rate: {win_rate:.4f} ({win_rate * 100:.1f}%)")
    logger.info(f"Average Turns: {avg_turns:.2f} ± {std_turns:.2f}")
    logger.info(f"Average Hit Rate: {avg_hit_rate:.4f}")
    logger.info(f"Min/Max Turns: {min(turn_counts)}/{max(turn_counts)}")
    
    return {
        "win_rate": win_rate,
        "avg_turns": avg_turns,
        "std_turns": std_turns,
        "avg_hit_rate": avg_hit_rate,
        "turn_counts": turn_counts,
    }

if __name__ == "__main__":
    # Path to the trained expert model
    model_path = "expert/best.zip"
    
    # Evaluate expert model
    results = evaluate_expert(model_path, num_episodes=50, board_size=6)
