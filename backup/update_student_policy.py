#!/usr/bin/env python
"""
Script to update the train_student_policy.py file to use the improved KL distillation
without modifying the original file permanently.
"""

import os
import shutil
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path to the original and backup file
ORIGINAL_FILE = "examples/train_student_policy.py"
BACKUP_FILE = "examples/train_student_policy.py.backup"

# Content of the improved KL distiller to patch in
IMPROVED_DISTILLER_CONTENT = """
class ImprovedKLDistiller(KLDistiller):
    \"\"\"
    Extended KL Distiller with improved trajectory collection that handles repetitive expert behavior.
    \"\"\"
    
    def generate_trajectory_data(self,
                                expert_model: PPO,
                                env_factory: callable,
                                n_trajectories: int = 100,
                                max_steps: int = 100) -> tuple:
        \"\"\"
        Generate trajectory data from expert model with better handling of repetitive behavior.
        
        Args:
            expert_model: Expert model (SB3 PPO model).
            env_factory: Callable that creates a new environment instance.
            n_trajectories: Number of trajectories to generate.
            max_steps: Maximum number of steps per trajectory.
            
        Returns:
            Tuple of (observations, expert_logits).
        \"\"\"
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
"""

def update_train_student_policy():
    """Update the train_student_policy.py file with improved KL distiller"""
    if not os.path.exists(ORIGINAL_FILE):
        logger.error(f"Original file not found: {ORIGINAL_FILE}")
        return False
    
    # Create a backup
    if not os.path.exists(BACKUP_FILE):
        logger.info(f"Creating backup: {BACKUP_FILE}")
        shutil.copy2(ORIGINAL_FILE, BACKUP_FILE)
    
    # Read the original file
    with open(ORIGINAL_FILE, 'r') as f:
        content = f.read()
    
    # Find the position to insert our improved KL distiller
    import_pos = content.find("from gym_battleship.distill import StrategyDistiller, KLDistiller")
    if import_pos == -1:
        logger.error("Could not find import statement in the file")
        return False
    
    # Find the train_kl_model function
    train_kl_pos = content.find("def train_kl_model(args):")
    if train_kl_pos == -1:
        logger.error("Could not find train_kl_model function in the file")
        return False
    
    # Insert our improved KL distiller after the imports
    end_of_import = import_pos + len("from gym_battleship.distill import StrategyDistiller, KLDistiller")
    new_content = content[:end_of_import] + "\n\n" + IMPROVED_DISTILLER_CONTENT + content[end_of_import:]
    
    # Modify the train_kl_model function to use our improved distiller
    distiller_creation = new_content.find("distiller = KLDistiller", train_kl_pos)
    if distiller_creation == -1:
        logger.error("Could not find distiller creation in train_kl_model function")
        return False
    
    # Replace KLDistiller with ImprovedKLDistiller
    new_content = new_content.replace("distiller = KLDistiller", "distiller = ImprovedKLDistiller", 1)
    
    # Write the modified content
    with open(ORIGINAL_FILE, 'w') as f:
        f.write(new_content)
    
    logger.info(f"Successfully updated {ORIGINAL_FILE} with improved KL distiller")
    return True

def restore_backup():
    """Restore the original file from backup"""
    if os.path.exists(BACKUP_FILE):
        logger.info(f"Restoring {ORIGINAL_FILE} from backup")
        shutil.copy2(BACKUP_FILE, ORIGINAL_FILE)
        return True
    else:
        logger.error(f"Backup file not found: {BACKUP_FILE}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--restore":
        # Restore from backup
        if restore_backup():
            print("Successfully restored original file from backup")
        else:
            print("Failed to restore original file")
    else:
        # Update the file
        if update_train_student_policy():
            print("Successfully updated train_student_policy.py with improved KL distiller")
            print("To run the experiment with improved distillation, use:")
            print("  ./run_experiment.sh")
            print("\nTo restore the original file after running, use:")
            print("  python update_student_policy.py --restore")
        else:
            print("Failed to update train_student_policy.py")
