#!/usr/bin/env python
"""
Compare Battleship strategies focusing on heuristic vs LPML-based approaches:
1. Random strategy (baseline)
2. Pure heuristic strategy (as implemented in train_expert.py)
3. LPML-based strategy (trained using LLM-annotated trajectories)

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
import torch

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
    p = argparse.ArgumentParser(description="Compare heuristic vs LPML-based Battleship strategies")
    p.add_argument("--board-size", type=int, default=4,
                   help="Size of the board (N×N)")
    p.add_argument("--episodes", type=int, default=50,
                   help="Number of episodes for evaluation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--heuristic-expert", type=str, default="expert/heuristic_expert.zip",
                   help="Path to the heuristic expert model")
    p.add_argument("--lpml-agent", type=str, default="distill/lpml_student.pth",
                   help="Path to the LPML-based agent model")
    p.add_argument("--transfer-size", type=int, default=None,
                   help="Evaluate on a different board size (transfer task)")
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

def evaluate_heuristic_expert(expert_path, board_size, episodes=50, verbose=False):
    """Evaluate the heuristic expert."""
    env = make_env(board_size)
    
    # Load heuristic expert
    try:
        with open(expert_path, 'rb') as f:
            data = pickle.load(f)
            
        if isinstance(data, dict) and 'expert' in data:
            expert = data['expert']
            logger.info(f"Loaded heuristic expert from {expert_path}")
            logger.info(f"Expert metadata: {data.get('metadata', {})}")
        else:
            logger.error(f"Invalid expert format in {expert_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading expert: {e}")
        return None
    
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
    
    logger.info(f"Heuristic expert: {avg_hits:.2f} average hits per episode")
    logger.info(f"Heuristic expert: {avg_steps:.2f} average steps per episode")
    logger.info(f"Heuristic expert: {avg_reward:.2f} average reward")
    logger.info(f"Heuristic expert: {win_rate:.1f}% win rate")
    
    results = {
        "strategy": "Heuristic Expert",
        "avg_hits": avg_hits,
        "avg_steps": avg_steps,
        "avg_reward": avg_reward,
        "win_rate": win_rate,
        "episode_rewards": episode_rewards
    }
    
    return results

def evaluate_lpml_agent(model_path, board_size, episodes=50, verbose=False):
    """Evaluate the LPML-based agent."""
    env = make_env(board_size)
    
    try:
        # Load the LPML-based student model
        model = torch.load(model_path)
        logger.info(f"Loaded LPML-based agent from {model_path}")
    except Exception as e:
        logger.error(f"Error loading LPML agent: {e}")
        return None
    
    try:
        # Switch model to evaluation mode
        model.eval()
    except:
        logger.warning("Could not set model to eval mode")
    
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
            # Get valid actions from mask
            action_mask = mask_fn(env)
            valid_actions = np.where(action_mask)[0]
            
            if len(valid_actions) == 0:
                break
            
            # Prepare observation for the model
            state_tensor = torch.FloatTensor(obs).unsqueeze(0)
            mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0)
            
            try:
                # Get action from the model
                with torch.no_grad():
                    logits = model(state_tensor)
                    # Apply action mask
                    logits[~mask_tensor] = float('-inf')
                    # Get action probabilities
                    probs = torch.softmax(logits, dim=-1)
                    action = torch.argmax(probs, dim=-1).item()
            except Exception as e:
                logger.error(f"Error predicting action with LPML agent: {e}")
                # Fallback to random action
                action = np.random.choice(valid_actions)
            
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
                    print(f"LPML [Ep {ep+1} Step {steps}]: Action ({x},{y}) - HIT!")
            elif verbose:
                x, y = action % board_size, action // board_size
                print(f"LPML [Ep {ep+1} Step {steps}]: Action ({x},{y}) - miss")
        
        total_hits += episode_hits
        total_steps += steps
        total_reward += ep_reward
        episode_rewards.append(ep_reward)
        total_wins += int(term and ep_reward > 9)  # Win if terminated with high reward
        
        if verbose or (ep + 1) % 10 == 0:
            logger.info(f"LPML Episode {ep+1}: Steps={steps}, Hits={episode_hits}, Reward={ep_reward:.1f}")
    
    avg_hits = total_hits / episodes
    avg_steps = total_steps / episodes
    win_rate = total_wins / episodes * 100
    avg_reward = total_reward / episodes
    
    logger.info(f"LPML-based agent: {avg_hits:.2f} average hits per episode")
    logger.info(f"LPML-based agent: {avg_steps:.2f} average steps per episode")
    logger.info(f"LPML-based agent: {avg_reward:.2f} average reward")
    logger.info(f"LPML-based agent: {win_rate:.1f}% win rate")
    
    results = {
        "strategy": "LPML Agent",
        "avg_hits": avg_hits,
        "avg_steps": avg_steps,
        "avg_reward": avg_reward,
        "win_rate": win_rate,
        "episode_rewards": episode_rewards
    }
    
    return results

def print_comparison_table(results_list):
    """Print a comparison table of all strategies."""
    header = f"| {'Strategy':18} | {'Avg Hits':10} | {'Avg Steps':10} | {'Avg Reward':10} | {'Win Rate':10} |"
    divider = "-" * len(header)
    
    print("\n" + divider)
    print(header)
    print(divider)
    
    for results in results_list:
        if results:
            print(f"| {results['strategy']:18} | {results['avg_hits']:10.2f} | {results['avg_steps']:10.2f} | {results['avg_reward']:10.2f} | {results['win_rate']:10.1f}% |")
    
    print(divider + "\n")

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine board sizes for evaluation
    eval_board_sizes = [args.board_size]
    if args.transfer_size:
        eval_board_sizes.append(args.transfer_size)
    
    for board_size in eval_board_sizes:
        logger.info(f"\n{'='*30}")
        if board_size == args.board_size:
            logger.info(f"Evaluating on original board size: {board_size}x{board_size}")
        else:
            logger.info(f"Evaluating on transfer board size: {board_size}x{board_size}")
        logger.info(f"{'='*30}\n")
        
        # Run evaluations
        random_results = evaluate_random_strategy(board_size, args.episodes, args.verbose)
        logger.info("\n" + "-"*50)
        
        heuristic_results = evaluate_heuristic_expert(args.heuristic_expert, board_size, args.episodes, args.verbose)
        logger.info("\n" + "-"*50)
        
        lpml_results = evaluate_lpml_agent(args.lpml_agent, board_size, args.episodes, args.verbose)
        
        # Print comparison table
        print_comparison_table([random_results, heuristic_results, lpml_results])
        
        # Report improvement percentages
        if heuristic_results and random_results:
            heuristic_improvement = 100 * (heuristic_results["avg_hits"] - random_results["avg_hits"]) / max(0.01, random_results["avg_hits"])
            logger.info(f"Heuristic vs Random: {heuristic_improvement:.1f}% more hits")
        
        if lpml_results and random_results:
            lpml_improvement = 100 * (lpml_results["avg_hits"] - random_results["avg_hits"]) / max(0.01, random_results["avg_hits"])
            logger.info(f"LPML vs Random: {lpml_improvement:.1f}% more hits")
        
        if heuristic_results and lpml_results:
            vs_improvement = 100 * (heuristic_results["avg_hits"] - lpml_results["avg_hits"]) / max(0.01, lpml_results["avg_hits"])
            logger.info(f"Heuristic vs LPML: {vs_improvement:.1f}% difference in hits")
            
            steps_diff = 100 * (heuristic_results["avg_steps"] - lpml_results["avg_steps"]) / max(0.01, lpml_results["avg_steps"])
            logger.info(f"Heuristic vs LPML: {steps_diff:.1f}% difference in steps")

if __name__ == "__main__":
    main()
