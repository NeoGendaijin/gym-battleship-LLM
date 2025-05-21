#!/usr/bin/env python
"""
Improved KL distillation script that handles repetitive expert behavior.
This script is a direct replacement for the KL-based distillation step in run_experiment.sh.
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import gymnasium as gym
import gym_battleship
from stable_baselines3 import PPO
from gym_battleship.distill import KLDistiller

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImprovedKLDistiller(KLDistiller):
    """
    Extended KL Distiller with improved trajectory collection that handles repetitive expert behavior.
    """
    
    def generate_trajectory_data(self,
                                expert_model: PPO,
                                env_factory: callable,
                                n_trajectories: int = 100,
                                max_steps: int = 100) -> tuple:
        """
        Generate trajectory data from expert model with better handling of repetitive behavior.
        
        Args:
            expert_model: Expert model (SB3 PPO model).
            env_factory: Callable that creates a new environment instance.
            n_trajectories: Number of trajectories to generate.
            max_steps: Maximum number of steps per trajectory.
            
        Returns:
            Tuple of (observations, expert_logits).
        """
        logger.info(f"Generating {n_trajectories} trajectories (up to {max_steps} steps each)")
        
        # Extract policy from PPO model
        policy = expert_model.policy
        
        # Lists to store data
        all_observations = []
        all_logits = []
        all_actions = []
        
        # Track repetitions across episodes
        total_repetitions = 0
        unique_states_seen = 0
        
        for traj_idx in range(n_trajectories):
            # Create new environment
            env = env_factory()
            obs, _ = env.reset(seed=traj_idx+100)  # Use different seeds
            
            # Track actions in this episode to detect repetition
            action_count = {}
            episode_steps = 0
            unique_steps = 0
            
            for step_idx in range(max_steps):
                episode_steps += 1
                
                # Always store the observation since we need state data
                # even if the expert makes mistakes
                all_observations.append(obs.copy())
                
                # Extract logits (action probabilities) from expert policy
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    features = policy.extract_features(obs_tensor)
                    latent_pi = policy.mlp_extractor.forward_actor(features)
                    action_logits = policy.action_net(latent_pi)
                    logits = action_logits.squeeze(0).numpy()
                    all_logits.append(logits)
                
                # Get action from expert model
                action, _ = expert_model.predict(obs, deterministic=True)
                all_actions.append(action)
                
                # Check for repetitions
                action_int = int(action)
                x, y = action_int % self.grid_size, action_int // self.grid_size
                coords = (x, y)
                
                action_count[coords] = action_count.get(coords, 0) + 1
                is_repeat = action_count[coords] > 1
                
                if not is_repeat:
                    unique_steps += 1
                else:
                    total_repetitions += 1
                
                # Take environment step
                prev_obs = obs.copy()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Check outcome for logging
                is_hit = (obs[0] != prev_obs[0]).any()
                is_miss = (obs[1] != prev_obs[1]).any()
                
                if is_hit:
                    outcome = "HIT"
                elif is_miss:
                    outcome = "MISS"
                else:
                    outcome = "REPEAT"
                
                # Log every 10th trajectory for progress indication
                if traj_idx % 10 == 0 and step_idx % 10 == 0:
                    logger.info(f"Trajectory {traj_idx+1}/{n_trajectories}, Step {step_idx+1}: Action=({x},{y}), Outcome={outcome}")
                
                if done:
                    break
            
            unique_states_seen += unique_steps
            
            # Log statistics for this trajectory
            if traj_idx % 10 == 0:
                logger.info(f"Trajectory {traj_idx+1} complete: {episode_steps} steps, {unique_steps} unique actions, {episode_steps-unique_steps} repeats")
            
            # Close environment
            env.close()
        
        # Log overall statistics
        logger.info(f"Data collection complete: {len(all_observations)} total steps, {unique_states_seen} unique actions, {total_repetitions} repetitions")
        
        # Ensure we have enough data for training
        if len(all_observations) < 100:
            logger.warning("Very few observations collected. Model may not train well.")
        
        # Convert lists to arrays
        observations = np.array(all_observations, dtype=np.float32)
        expert_logits = np.array(all_logits, dtype=np.float32)
        
        return observations, expert_logits

def parse_args():
    parser = argparse.ArgumentParser(description="Train a KL-based student policy from expert Battleship model")
    
    # Input/output arguments
    parser.add_argument(
        "--teacher", 
        type=str, 
        required=True,
        help="Path to the expert model"
    )
    parser.add_argument(
        "--out", 
        type=str, 
        required=True,
        help="Path to save the trained student model"
    )
    
    # Training arguments
    parser.add_argument(
        "--grid-size", 
        type=int, 
        default=6,
        help="Size of the game grid"
    )
    parser.add_argument(
        "--hidden-dim", 
        type=int, 
        default=128,
        help="Dimension of hidden layers in the student policy"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=1.0,
        help="Temperature for KL divergence"
    )
    parser.add_argument(
        "--n-trajectories", 
        type=int, 
        default=100,
        help="Number of trajectories to generate"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        # Set up device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        
        # Create KL distiller
        logger.info(f"Creating improved KL distiller with grid size: {args.grid_size}, hidden dim: {args.hidden_dim}")
        distiller = ImprovedKLDistiller(grid_size=args.grid_size, hidden_dim=args.hidden_dim)
        
        # Load teacher model
        try:
            logger.info(f"Loading teacher model from: {args.teacher}")
            teacher_model = PPO.load(args.teacher)
            
            # Create environment factory for generating trajectories
            def env_factory():
                return gym.make('Battleship-v0', board_size=(args.grid_size, args.grid_size))
            
            logger.info(f"Training student policy for {args.epochs} epochs...")
            model, history = distiller.distill_from_expert(
                expert_model=teacher_model,
                env_factory=env_factory,
                n_trajectories=args.n_trajectories,
                epochs=args.epochs,
                lr=args.lr,
                temperature=args.temperature,
                output_path=args.out
            )
            
            logger.info(f"Final training KL loss: {history['train_loss'][-1]:.6f}")
            logger.info(f"Final validation KL loss: {history['val_loss'][-1]:.6f}")
            
            logger.info(f"Student policy saved to: {args.out}")
            return 0
            
        except ImportError:
            logger.error("Stable-Baselines3 is required for KL-based distillation")
            return 1
            
    except Exception as e:
        logger.error(f"Error training student policy: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
