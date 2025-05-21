#!/usr/bin/env python
"""
Script to fix the "No valid observations were generated" error in KL distillation.
This provides a patched version of the KL distillation process that works with the existing expert model.
"""

import os
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
    
def fix_distillation(expert_model_path="expert/best.zip", output_path="distill/fixed_student.pth", grid_size=6):
    """Run the fixed KL distillation process."""
    logger.info(f"Loading expert model from {expert_model_path}")
    teacher_model = PPO.load(expert_model_path)
    
    # Create improved distiller
    logger.info(f"Creating improved KL distiller with grid size: {grid_size}")
    distiller = ImprovedKLDistiller(grid_size=grid_size, hidden_dim=128)
    
    # Create environment factory
    def env_factory():
        return gym.make('Battleship-v0', board_size=(grid_size, grid_size))
    
    # Train student model
    logger.info("Training student policy (this may take a while)...")
    try:
        model, history = distiller.distill_from_expert(
            expert_model=teacher_model,
            env_factory=env_factory,
            n_trajectories=50,  # Generate 50 trajectories
            epochs=10,
            lr=0.001,
            temperature=1.0,
            output_path=output_path
        )
        
        logger.info("Student model training complete!")
        logger.info(f"Final training loss: {history['train_loss'][-1]:.6f}")
        logger.info(f"Final validation loss: {history['val_loss'][-1]:.6f}")
        return True
    except Exception as e:
        logger.error(f"Error during distillation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    # Make sure output directory exists
    os.makedirs("distill", exist_ok=True)
    
    try:
        # Try with the original expert model first
        success = fix_distillation(
            expert_model_path="expert/best.zip", 
            output_path="distill/fixed_student.pth",
            grid_size=6
        )
        
        if success:
            print("\nDistillation completed successfully!")
            print("The student model has been saved to distill/fixed_student.pth")
            return 0
        else:
            print("\nDistillation failed. Consider training a better expert model first.")
            print("Run 'python run_expert_training.py' to train an improved expert model.")
            return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())
