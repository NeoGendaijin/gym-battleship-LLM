#!/usr/bin/env python
"""
Compare different Battleship strategies:
1. Random strategy (baseline)
2. Heuristic strategy (as implemented in train_expert.py)
3. RL strategy (as trained in train_basic.py)
4. Hybrid strategy (as trained in train_basic.py with heuristic components)

This script shows how the different strategies compare in terms of:
- Average number of hits per episode
- Win rate
- Number of steps needed to complete an episode
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
import time

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# Import the expert implementation
from expert.train_expert import BattleshipExpert, mask_fn, get_ship_sizes

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="Compare Battleship strategies")
    p.add_argument("--board-size", type=int, default=4,
                   help="Size of the board (N×N)")
    p.add_argument("--episodes", type=int, default=50,
                   help="Number of episodes for evaluation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rl-model", type=str, default="expert/maskable_expert.zip",
                   help="Path to the RL model file")
    p.add_argument("--hybrid-model", type=str, default="expert/hybrid_expert.zip",
                   help="Path to the hybrid model file")
    p.add_argument("--verbose", action="store_true",
                   help="Show detailed output")
    return p.parse_args()

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

def evaluate_random_strategy(board_size, episodes=50, verbose=False):
    """Evaluate a random strategy to establish a baseline."""
    env = make_env(board_size)
    total_hits = 0
    total_wins = 0
    total_steps = 0
    total_reward = 0
    episode_rewards = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        episode_hits = 0
        ep_reward = 0
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
            ep_reward += reward
            
            # Count hits by checking if a new hit was registered in observation
            if (obs[0] != prev_obs[0]).any():
                episode_hits += 1
                if verbose:
                    x, y = action % board_size, action // board_size
                    print(f"Random [Ep {ep+1} Step {steps}]: Action ({x},{y}) - HIT!")
            elif verbose:
                x, y = action % board_size, action // board_size
                print(f"Random [Ep {ep+1} Step {steps}]: Action ({x},{y}) - miss")
        
        total_hits += episode_hits
        total_steps += steps
        total_reward += ep_reward
        episode_rewards.append(ep_reward)
        total_wins += int(term and ep_reward > 9)  # Win if terminated with high reward
        
        if verbose or (ep + 1) % 10 == 0:
            logger.info(f"Random Episode {ep+1}: Steps={steps}, Hits={episode_hits}, Reward={ep_reward:.1f}")
    
    avg_hits = total_hits / episodes
    avg_steps = total_steps / episodes
    win_rate = total_wins / episodes * 100
    avg_reward = total_reward / episodes
    
    logger.info(f"Random strategy: {avg_hits:.2f} average hits per episode")
    logger.info(f"Random strategy: {avg_steps:.2f} average steps per episode")
    logger.info(f"Random strategy: {avg_reward:.2f} average reward")
    logger.info(f"Random strategy: {win_rate:.1f}% win rate")
    
    results = {
        "strategy": "Random",
        "avg_hits": avg_hits,
        "avg_steps": avg_steps,
        "avg_reward": avg_reward,
        "win_rate": win_rate,
        "episode_rewards": episode_rewards
    }
    
    return results

def evaluate_heuristic_strategy(board_size, episodes=50, verbose=False):
    """Evaluate the heuristic strategy."""
    env = make_env(board_size)
    expert = BattleshipExpert(board_size, verbose=verbose)
    
    total_hits = 0
    total_wins = 0
    total_steps = 0
    total_reward = 0
    episode_rewards = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        expert.reset()
        
        done = False
        episode_hits = 0
        ep_reward = 0
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
            ep_reward += reward
            steps += 1
            
            # Count hits
            hit_detected = False
            if (obs[0] != prev_obs[0]).any():
                episode_hits += 1
                hit_detected = True
                if verbose:
                    x, y = expert.action_to_coord[action]
                    print(f"Heuristic [Ep {ep+1} Step {steps}]: Action ({x},{y}) - HIT!")
            elif verbose:
                x, y = expert.action_to_coord[action]
                print(f"Heuristic [Ep {ep+1} Step {steps}]: Action ({x},{y}) - miss")
            
            # Update expert's state
            expert.update_state(action, obs, prev_obs, reward, done, info)
        
        total_hits += episode_hits
        total_steps += steps
        total_reward += ep_reward
        episode_rewards.append(ep_reward)
        total_wins += int(term and ep_reward > 9)  # Win if terminated with high reward
        
        if verbose or (ep + 1) % 10 == 0:
            logger.info(f"Heuristic Episode {ep+1}: Steps={steps}, Hits={episode_hits}, Reward={ep_reward:.1f}")
    
    avg_hits = total_hits / episodes
    avg_steps = total_steps / episodes
    win_rate = total_wins / episodes * 100
    avg_reward = total_reward / episodes
    
    logger.info(f"Heuristic strategy: {avg_hits:.2f} average hits per episode")
    logger.info(f"Heuristic strategy: {avg_steps:.2f} average steps per episode")
    logger.info(f"Heuristic strategy: {avg_reward:.2f} average reward")
    logger.info(f"Heuristic strategy: {win_rate:.1f}% win rate")
    
    results = {
        "strategy": "Heuristic",
        "avg_hits": avg_hits,
        "avg_steps": avg_steps,
        "avg_reward": avg_reward,
        "win_rate": win_rate,
        "episode_rewards": episode_rewards
    }
    
    return results

def evaluate_rl_strategy(model_path, board_size, episodes=50, verbose=False):
    """Evaluate the RL strategy."""
    env = make_env(board_size)
    
    try:
        # Try to load the model as RL model
        model = MaskablePPO.load(model_path)
        logger.info(f"Loaded RL model from {model_path}")
    except Exception as e:
        logger.error(f"Error loading RL model: {e}")
        return None
    
    total_hits = 0
    total_wins = 0
    total_steps = 0
    total_reward = 0
    episode_rewards = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        
        done = False
        episode_hits = 0
        ep_reward = 0
        steps = 0
        
        while not done and steps < board_size**2:
            # Get action from the model
            action, _ = model.predict(obs, deterministic=True)
            
            # Take the action
            prev_obs = obs.copy()
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            ep_reward += reward
            steps += 1
            
            # Count hits
            if (obs[0] != prev_obs[0]).any():
                episode_hits += 1
                if verbose:
                    x, y = action % board_size, action // board_size
                    print(f"RL [Ep {ep+1} Step {steps}]: Action ({x},{y}) - HIT!")
            elif verbose:
                x, y = action % board_size, action // board_size
                print(f"RL [Ep {ep+1} Step {steps}]: Action ({x},{y}) - miss")
        
        total_hits += episode_hits
        total_steps += steps
        total_reward += ep_reward
        episode_rewards.append(ep_reward)
        total_wins += int(term and ep_reward > 9)  # Win if terminated with high reward
        
        if verbose or (ep + 1) % 10 == 0:
            logger.info(f"RL Episode {ep+1}: Steps={steps}, Hits={episode_hits}, Reward={ep_reward:.1f}")
    
    avg_hits = total_hits / episodes
    avg_steps = total_steps / episodes
    win_rate = total_wins / episodes * 100
    avg_reward = total_reward / episodes
    
    logger.info(f"RL strategy: {avg_hits:.2f} average hits per episode")
    logger.info(f"RL strategy: {avg_steps:.2f} average steps per episode")
    logger.info(f"RL strategy: {avg_reward:.2f} average reward")
    logger.info(f"RL strategy: {win_rate:.1f}% win rate")
    
    results = {
        "strategy": "RL",
        "avg_hits": avg_hits,
        "avg_steps": avg_steps,
        "avg_reward": avg_reward,
        "win_rate": win_rate,
        "episode_rewards": episode_rewards
    }
    
    return results

def evaluate_hybrid_strategy(model_path, board_size, episodes=50, verbose=False):
    """Evaluate the hybrid strategy."""
    env = make_env(board_size)
    
    try:
        # Try to load the model as RL model
        model = MaskablePPO.load(model_path)
        logger.info(f"Loaded hybrid model from {model_path}")
    except Exception as e:
        logger.error(f"Error loading hybrid model: {e}")
        return None
    
    total_hits = 0
    total_wins = 0
    total_steps = 0
    total_reward = 0
    episode_rewards = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        
        done = False
        episode_hits = 0
        ep_reward = 0
        steps = 0
        
        while not done and steps < board_size**2:
            # Get action from the model
            action, _ = model.predict(obs, deterministic=True)
            
            # Take the action
            prev_obs = obs.copy()
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            ep_reward += reward
            steps += 1
            
            # Count hits
            if (obs[0] != prev_obs[0]).any():
                episode_hits += 1
                if verbose:
                    x, y = action % board_size, action // board_size
                    print(f"Hybrid [Ep {ep+1} Step {steps}]: Action ({x},{y}) - HIT!")
            elif verbose:
                x, y = action % board_size, action // board_size
                print(f"Hybrid [Ep {ep+1} Step {steps}]: Action ({x},{y}) - miss")
        
        total_hits += episode_hits
        total_steps += steps
        total_reward += ep_reward
        episode_rewards.append(ep_reward)
        total_wins += int(term and ep_reward > 9)  # Win if terminated with high reward
        
        if verbose or (ep + 1) % 10 == 0:
            logger.info(f"Hybrid Episode {ep+1}: Steps={steps}, Hits={episode_hits}, Reward={ep_reward:.1f}")
    
    avg_hits = total_hits / episodes
    avg_steps = total_steps / episodes
    win_rate = total_wins / episodes * 100
    avg_reward = total_reward / episodes
    
    logger.info(f"Hybrid strategy: {avg_hits:.2f} average hits per episode")
    logger.info(f"Hybrid strategy: {avg_steps:.2f} average steps per episode")
    logger.info(f"Hybrid strategy: {avg_reward:.2f} average reward")
    logger.info(f"Hybrid strategy: {win_rate:.1f}% win rate")
    
    results = {
        "strategy": "Hybrid",
        "avg_hits": avg_hits,
        "avg_steps": avg_steps,
        "avg_reward": avg_reward,
        "win_rate": win_rate,
        "episode_rewards": episode_rewards
    }
    
    return results

def print_comparison_table(results_list):
    """Print a comparison table of all strategies."""
    header = f"| {'Strategy':15} | {'Avg Hits':10} | {'Avg Steps':10} | {'Avg Reward':10} | {'Win Rate':10} |"
    divider = "-" * len(header)
    
    print("\n" + divider)
    print(header)
    print(divider)
    
    for results in results_list:
        if results:
            print(f"| {results['strategy']:15} | {results['avg_hits']:10.2f} | {results['avg_steps']:10.2f} | {results['avg_reward']:10.2f} | {results['win_rate']:10.1f}% |")
    
    print(divider + "\n")

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    logger.info(f"Evaluating strategies on {args.board_size}x{args.board_size} board for {args.episodes} episodes")
    
    # Run all evaluations
    random_results = evaluate_random_strategy(args.board_size, args.episodes, args.verbose)
    logger.info("\n" + "="*50)
    
    heuristic_results = evaluate_heuristic_strategy(args.board_size, args.episodes, args.verbose)
    logger.info("\n" + "="*50)
    
    rl_results = evaluate_rl_strategy(args.rl_model, args.board_size, args.episodes, args.verbose)
    logger.info("\n" + "="*50)
    
    hybrid_results = evaluate_hybrid_strategy(args.hybrid_model, args.board_size, args.episodes, args.verbose)
    
    # Print comparison table
    print_comparison_table([random_results, heuristic_results, rl_results, hybrid_results])
    
    # Report improvement percentages
    if heuristic_results and random_results:
        heuristic_improvement = 100 * (heuristic_results["avg_hits"] - random_results["avg_hits"]) / random_results["avg_hits"]
        logger.info(f"Heuristic vs Random: {heuristic_improvement:.1f}% more hits")
    
    if rl_results and random_results:
        rl_improvement = 100 * (rl_results["avg_hits"] - random_results["avg_hits"]) / random_results["avg_hits"]
        logger.info(f"RL vs Random: {rl_improvement:.1f}% more hits")
    
    if hybrid_results and random_results:
        hybrid_improvement = 100 * (hybrid_results["avg_hits"] - random_results["avg_hits"]) / random_results["avg_hits"]
        logger.info(f"Hybrid vs Random: {hybrid_improvement:.1f}% more hits")
    
    if heuristic_results and rl_results:
        vs_improvement = 100 * (heuristic_results["avg_hits"] - rl_results["avg_hits"]) / rl_results["avg_hits"]
        logger.info(f"Heuristic vs RL: {vs_improvement:.1f}% more hits")

if __name__ == "__main__":
    main()
