#!/usr/bin/env python
"""
Evaluate the performance of the pure heuristic expert on the Battleship game.
This script loads and runs the heuristic expert, comparing it to random strategy.
"""

import argparse
import logging
import numpy as np
import gymnasium as gym
import gym_battleship
import random
import pickle
import os
from tqdm import tqdm

from sb3_contrib.common.wrappers import ActionMasker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Battleship expert")
    p.add_argument("--expert-path", type=str, default="expert/pure_heuristic.zip",
                   help="Path to the expert model/pickle")
    p.add_argument("--board-size", type=int, default=4,
                   help="Size of the board (N×N)")
    p.add_argument("--episodes", type=int, default=50,
                   help="Number of episodes to evaluate")
    p.add_argument("--random-episodes", type=int, default=50,
                   help="Number of episodes for random baseline")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true",
                   help="Show detailed output")
    return p.parse_args()

def get_ship_sizes(board_size: int):
    if board_size <= 4:
        return {3: 1, 2: 2, 1: 2}
    if board_size <= 6:
        return {4: 1, 3: 1, 2: 2}
    return {5: 1, 4: 1, 3: 2, 2: 1}

def mask_fn(env):
    """Return a mask of valid (untried) cells."""
    core_env = env.unwrapped if hasattr(env, "unwrapped") else env
    obs = core_env.observation
    hits = obs[0].flatten()
    misses = obs[1].flatten()
    mask = ~((hits == 1) | (misses == 1))
    return mask

def load_expert(expert_path):
    """Load the heuristic expert from a pickle file."""
    try:
        with open(expert_path, 'rb') as f:
            data = pickle.load(f)
            
        if isinstance(data, dict) and 'expert' in data:
            expert = data['expert']
            logger.info(f"Loaded heuristic expert from {expert_path}")
            logger.info(f"Expert metadata: {data.get('metadata', {})}")
            return expert
        else:
            logger.error(f"Invalid expert format in {expert_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading expert: {e}")
        return None

def make_env(board_size):
    """Create the Battleship environment."""
    env = gym.make(
        "Battleship-v0",
        board_size=(board_size, board_size),
        ship_sizes=get_ship_sizes(board_size),
        reward_dictionary={
            "win": 10,
            "missed": -0.1,
            "touched": 1,
            "repeat_missed": 0,
            "repeat_touched": 0,
        },
        episode_steps=board_size**2,
    )
    env = ActionMasker(env, mask_fn)
    return env

def evaluate_random_strategy(board_size, episodes=50):
    """Evaluate a random strategy to establish a baseline."""
    env = make_env(board_size)
    total_hits = 0
    total_wins = 0
    total_steps = 0
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        episode_hits = 0
        steps = 0
        
        while not done and steps < board_size**2:
            # Use mask to get valid actions
            action_mask = mask_fn(env)
            valid_actions = np.where(action_mask)[0]
            
            if len(valid_actions) == 0:
                break
                
            # Choose a random action from valid actions
            action = np.random.choice(valid_actions)
            
            # Take step in environment
            prev_obs = obs.copy()
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
            
            # Count hits by checking if a new hit was registered in observation
            if (obs[0] != prev_obs[0]).any():
                episode_hits += 1
        
        total_hits += episode_hits
        total_steps += steps
        total_wins += int(term and reward > 9)  # Win if terminated with high reward
        
        if (ep + 1) % 10 == 0:
            logger.info(f"Random Episode {ep+1}: Steps={steps}, Hits={episode_hits}")
    
    avg_hits = total_hits / episodes
    avg_steps = total_steps / episodes
    win_rate = total_wins / episodes * 100
    
    logger.info(f"Random strategy: {avg_hits:.2f} average hits per episode")
    logger.info(f"Random strategy: {avg_steps:.2f} average steps per episode")
    logger.info(f"Random strategy: {win_rate:.1f}% win rate")
    
    return avg_hits, win_rate

def evaluate_expert(expert, board_size, random_baseline, episodes=50, verbose=False):
    """Evaluate the expert agent."""
    env = make_env(board_size)
    total_hits = 0
    total_wins = 0
    total_steps = 0
    better_than_random = 0
    
    for ep in range(episodes):
        obs, info = env.reset()
        expert.reset()
        
        done = False
        episode_hits = 0
        total_reward = 0
        steps = 0
        
        while not done and steps < board_size**2:
            # Get valid actions from mask
            action_mask = mask_fn(env)
            valid_actions = np.where(action_mask)[0].tolist()
            
            # Expert decides the action
            action = expert.decide_action(obs, valid_actions)
            
            # Take the action
            prev_obs = obs.copy()
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            total_reward += reward
            steps += 1
            
            # Count hits
            if (obs[0] != prev_obs[0]).any():
                episode_hits += 1
            
            # Update expert's state
            expert.update_state(action, obs, prev_obs, reward, done, info)
            
            if verbose:
                x, y = expert.action_to_coord[action]
                hit = (obs[0] != prev_obs[0]).any()
                print(f"Step {steps}: Action {action} ({x},{y}) - {'HIT!' if hit else 'miss'}")
        
        total_hits += episode_hits
        total_steps += steps
        total_wins += int(term and reward > 9)  # Win if terminated with high reward
        better_than_random += int(episode_hits > random_baseline)
        
        logger.info(f"Expert Episode {ep+1}: Steps={steps}, Hits={episode_hits}, Reward={total_reward:.1f}")
    
    avg_hits = total_hits / episodes
    avg_steps = total_steps / episodes
    win_rate = total_wins / episodes * 100
    better_rate = better_than_random / episodes * 100
    
    logger.info("\nExpert Performance Summary:")
    logger.info(f"Average hits per episode: {avg_hits:.2f} (Random baseline: {random_baseline:.2f})")
    logger.info(f"Average steps per episode: {avg_steps:.2f}")
    logger.info(f"Better than random rate: {better_rate:.1f}% ({better_than_random}/{episodes})")
    logger.info(f"Win rate: {win_rate:.1f}% ({total_wins}/{episodes})")
    
    return avg_hits, win_rate

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Establish random baseline
    logger.info(f"\n=== Evaluating random baseline on {args.board_size}x{args.board_size} board ===")
    random_hits, random_win_rate = evaluate_random_strategy(
        args.board_size, args.random_episodes
    )
    
    # Load and evaluate expert
    logger.info(f"\n=== Loading expert from {args.expert_path} ===")
    expert = load_expert(args.expert_path)
    
    if expert:
        logger.info(f"\n=== Evaluating expert on {args.board_size}x{args.board_size} board ===")
        expert_hits, expert_win_rate = evaluate_expert(
            expert, args.board_size, random_hits, args.episodes, args.verbose
        )
        
        # Print comparison
        logger.info("\n=== Expert vs. Random Comparison ===")
        logger.info(f"Expert hits: {expert_hits:.2f} vs Random hits: {random_hits:.2f}")
        logger.info(f"Expert wins: {expert_win_rate:.1f}% vs Random wins: {random_win_rate:.1f}%")
        logger.info(f"Improvement: {100 * (expert_hits - random_hits) / max(1, random_hits):.1f}% more hits")
    else:
        logger.error("Failed to load expert. Evaluation cancelled.")

if __name__ == "__main__":
    main()
