#!/usr/bin/env python
"""
Simplified Battleship pipeline using the hybrid model.

This script runs a complete pipeline:
1. Use the hybrid model (trained with heuristic strategies)
2. Collect trajectories using the hybrid model
3. (Optional) Generate LPML annotations
4. Train student policy using hybrid trajectories
5. Evaluate the performance
"""

import os
import argparse
import logging
import numpy as np
import gymnasium as gym
import gym_battleship
import random
import time
import pickle
from tqdm import tqdm

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="Run hybrid Battleship pipeline")
    p.add_argument("--hybrid-model", type=str, default="expert/hybrid_expert.zip",
                   help="Path to the hybrid model file")
    p.add_argument("--board-size", type=int, default=4,
                   help="Size of the board (N×N)")
    p.add_argument("--collect-episodes", type=int, default=100,
                   help="Number of episodes to collect for trajectories")
    p.add_argument("--eval-episodes", type=int, default=50,
                   help="Number of episodes for evaluation")
    p.add_argument("--traj-output", type=str, default="trajectories/hybrid_trajectories.pkl",
                   help="Where to save collected trajectories")
    p.add_argument("--student-output", type=str, default="distill/student_hybrid.pth",
                   help="Where to save the student model")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--skip-collect", action="store_true",
                   help="Skip trajectory collection")
    p.add_argument("--skip-training", action="store_true",
                   help="Skip student training")
    p.add_argument("--verbose", action="store_true",
                   help="Show detailed output")
    return p.parse_args()

def get_ship_sizes(board_size: int):
    """Return appropriate ship configurations based on board size."""
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

def collect_trajectories(model_path, num_episodes, board_size, output_path, verbose=False):
    """Collect trajectories using the provided model."""
    logger.info(f"Collecting {num_episodes} trajectories using {model_path}")
    
    # Load model
    try:
        model = MaskablePPO.load(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False
    
    # Create environment
    env = make_env(board_size)
    
    # Collect trajectories
    trajectories = []
    
    for episode in tqdm(range(num_episodes), desc="Collecting trajectories"):
        obs, info = env.reset()
        done = False
        episode_traj = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "infos": []
        }
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            # Record pre-step data
            episode_traj["observations"].append(obs.copy())
            episode_traj["actions"].append(action)
            
            # Take step
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            
            # Record post-step data
            episode_traj["rewards"].append(reward)
            episode_traj["dones"].append(done)
            episode_traj["infos"].append(info.copy() if info else {})
            
            if verbose and episode % 10 == 0:
                x, y = action % board_size, action // board_size
                hit = "HIT!" if reward > 0 else "miss"
                print(f"Episode {episode}, Step: ({x},{y}) - {hit}")
        
        trajectories.append(episode_traj)
        
        if (episode + 1) % 10 == 0:
            logger.info(f"Collected {episode + 1} trajectories")
    
    # Save trajectories
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(trajectories, f)
    
    logger.info(f"Saved {len(trajectories)} trajectories to {output_path}")
    return True

def train_student(trajectories_path, output_path, board_size, verbose=False):
    """Train a student policy from the collected trajectories."""
    logger.info(f"Training student policy using {trajectories_path}")
    
    # This is a simplified version that would normally call examples/train_student_policy.py
    # For demonstration purposes, we'll just log what would happen
    
    logger.info(f"[SIMULATION] Loading trajectories from {trajectories_path}")
    logger.info(f"[SIMULATION] Training student model with board size {board_size}")
    logger.info(f"[SIMULATION] Saving student model to {output_path}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # In a real implementation, this would train the model using the trajectories
    # For now, we'll just create a dummy file
    with open(output_path, 'w') as f:
        f.write("This is a simulated student model file.")
    
    logger.info(f"Student model saved to {output_path}")
    return True

def evaluate_model(model_path, board_size, num_episodes, verbose=False):
    """Evaluate the model performance."""
    logger.info(f"Evaluating model {model_path}")
    
    # For the hybrid model, we'll load and evaluate it
    # For the student model, we'd normally load it but will skip for now
    
    if os.path.exists(model_path):
        try:
            model = MaskablePPO.load(model_path)
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model for evaluation: {e}")
            return
    else:
        logger.info(f"Model {model_path} doesn't exist, skipping evaluation")
        return
    
    env = make_env(board_size)
    
    total_reward = 0
    total_hits = 0
    total_wins = 0
    
    for episode in tqdm(range(num_episodes), desc="Evaluating model"):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        hits = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            prev_obs = obs.copy()
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            ep_reward += reward
            
            # Count hits
            if (obs[0] != prev_obs[0]).any():
                hits += 1
        
        total_reward += ep_reward
        total_hits += hits
        total_wins += 1 if term and reward > 9 else 0
        
        if verbose or (episode + 1) % 10 == 0:
            logger.info(f"Episode {episode+1}: Reward={ep_reward:.1f}, Hits={hits}")
    
    avg_reward = total_reward / num_episodes
    avg_hits = total_hits / num_episodes
    win_rate = total_wins / num_episodes * 100
    
    logger.info(f"Evaluation results for {model_path}:")
    logger.info(f"Average reward: {avg_reward:.2f}")
    logger.info(f"Average hits: {avg_hits:.2f}")
    logger.info(f"Win rate: {win_rate:.1f}%")
    
    return {
        "avg_reward": avg_reward,
        "avg_hits": avg_hits,
        "win_rate": win_rate
    }

def run_random_baseline(board_size, num_episodes, verbose=False):
    """Run a random baseline evaluation."""
    logger.info("Evaluating random baseline strategy")
    
    env = make_env(board_size)
    
    total_reward = 0
    total_hits = 0
    total_wins = 0
    
    for episode in tqdm(range(num_episodes), desc="Random baseline"):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        hits = 0
        
        while not done:
            # Use mask to get valid actions
            action_mask = mask_fn(env)
            valid_actions = np.where(action_mask)[0]
            
            if len(valid_actions) == 0:
                break
                
            # Choose a random action from valid actions
            action = np.random.choice(valid_actions)
            
            prev_obs = obs.copy()
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            ep_reward += reward
            
            # Count hits
            if (obs[0] != prev_obs[0]).any():
                hits += 1
        
        total_reward += ep_reward
        total_hits += hits
        total_wins += 1 if term and reward > 9 else 0
    
    avg_reward = total_reward / num_episodes
    avg_hits = total_hits / num_episodes
    win_rate = total_wins / num_episodes * 100
    
    logger.info(f"Random baseline results:")
    logger.info(f"Average reward: {avg_reward:.2f}")
    logger.info(f"Average hits: {avg_hits:.2f}")
    logger.info(f"Win rate: {win_rate:.1f}%")
    
    return {
        "avg_reward": avg_reward,
        "avg_hits": avg_hits,
        "win_rate": win_rate
    }

def main():
    args = parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    logger.info(f"Running hybrid model pipeline with board size {args.board_size}")
    
    # Step 1: Collect trajectories using hybrid model
    if not args.skip_collect:
        collect_trajectories(
            args.hybrid_model,
            args.collect_episodes,
            args.board_size,
            args.traj_output,
            args.verbose
        )
    else:
        logger.info("Skipping trajectory collection")
    
    # Step 2: Train student model
    if not args.skip_training:
        train_student(
            args.traj_output,
            args.student_output,
            args.board_size,
            args.verbose
        )
    else:
        logger.info("Skipping student training")
    
    # Step 3: Evaluate models
    logger.info("=== Evaluating models ===")
    
    # Run random baseline for comparison
    random_results = run_random_baseline(
        args.board_size,
        args.eval_episodes,
        args.verbose
    )
    
    # Evaluate hybrid model
    hybrid_results = evaluate_model(
        args.hybrid_model,
        args.board_size,
        args.eval_episodes,
        args.verbose
    )
    
    # Compare results
    if hybrid_results and random_results:
        logger.info("\n=== Performance Comparison ===")
        logger.info(f"Random baseline: {random_results['avg_hits']:.2f} hits, {random_results['win_rate']:.1f}% win rate")
        logger.info(f"Hybrid model: {hybrid_results['avg_hits']:.2f} hits, {hybrid_results['win_rate']:.1f}% win rate")
        
        hit_improvement = 100 * (hybrid_results['avg_hits'] - random_results['avg_hits']) / random_results['avg_hits']
        logger.info(f"Hit improvement: {hit_improvement:.1f}%")
    
    logger.info("Pipeline execution completed!")

if __name__ == "__main__":
    main()
