#!/usr/bin/env python
"""
A simple script to evaluate a trained model against a random baseline.
"""

import argparse
import logging
import numpy as np
import gymnasium as gym
import gym_battleship
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained Battleship model")
    p.add_argument("--model-path", type=str, default="expert/maskable_expert.zip",
                   help="Path to the trained model")
    p.add_argument("--board-size", type=int, default=4,
                   help="Size of the board (N×N)")
    p.add_argument("--episodes", type=int, default=20,
                   help="Number of episodes to evaluate")
    p.add_argument("--baseline-episodes", type=int, default=20,
                   help="Number of episodes for random baseline")
    p.add_argument("--seed", type=int, default=42)
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

def make_env(board_size):
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

def evaluate_random_strategy(board_size, episodes=30):
    """Evaluate a random strategy to establish a baseline."""
    env = make_env(board_size)
    total_hits = 0
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        episode_hits = 0
        steps = 0
        max_steps = board_size * board_size
        
        while not done and steps < max_steps:
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
        logger.info(f"Random Episode {ep+1}: Steps={steps}, Hits={episode_hits}")
    
    avg_hits = total_hits / episodes
    logger.info(f"Random strategy: {avg_hits:.2f} average hits per episode")
    return avg_hits

def evaluate_model(model, board_size, baseline_hits, episodes=10):
    env = make_env(board_size)
    wins = 0
    absolute_wins = 0
    total_hits = 0
    
    for ep in range(episodes):
        obs, info = env.reset()
        done, ep_r, hits = False, 0, 0
        steps = 0
        
        while not done and steps < board_size ** 2:
            action, _ = model.predict(obs, deterministic=True)
            prev_obs = obs.copy()
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            ep_r += r
            steps += 1
            
            # Count hits by checking if a new hit was registered in observation
            if (obs[0] != prev_obs[0]).any():
                hits += 1
        
        # Track absolute wins (all ships sunk)
        absolute_wins += int(term and r > 9)  # Win reward should be >= 10
        
        # Track relative wins (better than random baseline)
        wins += int(hits > baseline_hits)
        total_hits += hits
        
        logger.info(f"Model Episode {ep+1}: Steps={steps}, Hits={hits}, Reward={ep_r:.1f}, " + 
                   f"Better than random={hits > baseline_hits}, All ships sunk={term and r > 9}")
    
    # Log overall statistics
    avg_hits = total_hits / episodes
    logger.info(f"Model evaluation complete: {episodes} episodes")
    logger.info(f"Average hits per episode: {avg_hits:.2f} (Random baseline: {baseline_hits:.2f})")
    logger.info(f"Better than random win rate: {100 * wins / episodes:.1f}% ({wins}/{episodes})")
    logger.info(f"Absolute win rate: {100 * absolute_wins / episodes:.1f}% ({absolute_wins}/{episodes})")

def main():
    args = parse_args()
    
    # Establish random baseline
    logger.info(f"Evaluating random strategy baseline on {args.board_size}x{args.board_size} board...")
    baseline_hits = evaluate_random_strategy(args.board_size, args.baseline_episodes)
    
    # Load and evaluate model
    logger.info(f"Loading model from {args.model_path}...")
    try:
        model = MaskablePPO.load(args.model_path)
        logger.info(f"Evaluating model on {args.board_size}x{args.board_size} board...")
        evaluate_model(model, args.board_size, baseline_hits, args.episodes)
    except Exception as e:
        logger.error(f"Error loading or evaluating model: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
