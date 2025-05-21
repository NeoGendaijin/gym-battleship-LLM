#!/usr/bin/env python
"""
Script to train an expert model that doesn't get stuck in repetitive behavior.
This uses a higher learning rate, stronger penalties for repetition, and more training steps.
"""

import os
import logging
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym
import gym_battleship
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model(model, board_size=6, num_episodes=3, max_steps=100):
    """Run a quick test of the model to see if it's working correctly."""
    env = gym.make('Battleship-v0', board_size=(board_size, board_size))
    
    total_reward = 0
    total_steps = 0
    total_repeats = 0
    win_count = 0
    
    for episode in range(num_episodes):
        logger.info(f"Test Episode {episode+1}/{num_episodes}")
        obs, _ = env.reset(seed=episode+100)  # Different seeds
        
        episode_reward = 0
        done = False
        step = 0
        action_count = {}
        
        # Display board for debugging
        ship_cells = np.sum(env.unwrapped.board_generated)
        logger.info(f"Ship cells on board: {ship_cells}")
        
        while not done and step < max_steps:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Track repeated actions
            flat_action = int(action)
            x, y = flat_action % board_size, flat_action // board_size
            coords = (x, y)
            action_count[coords] = action_count.get(coords, 0) + 1
            repeated = action_count[coords] > 1
            
            if repeated:
                total_repeats += 1
                
            # Take step
            prev_obs = obs.copy()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Determine outcome for logging
            is_hit = (obs[0] != prev_obs[0]).any()
            is_miss = (obs[1] != prev_obs[1]).any()
            if is_hit:
                outcome = "HIT"
            elif is_miss:
                outcome = "MISS"
            else:
                outcome = "REPEAT"
            
            episode_reward += reward
            logger.info(f"  Step {step+1}: Action ({x},{y}) - {outcome}, Reward: {reward}")
            step += 1
        
        total_reward += episode_reward
        total_steps += step
        
        if info.get('win', False):
            win_count += 1
            
        # Episode summary
        unique_actions = len(action_count)
        repeated_actions = step - unique_actions
        logger.info(f"Episode summary: Steps={step}, Reward={episode_reward}, Unique Actions={unique_actions}, Repeats={repeated_actions}")
    
    # Overall summary
    logger.info(f"Test Summary: Avg Reward={total_reward/num_episodes:.2f}, Avg Steps={total_steps/num_episodes:.1f}, Wins={win_count}/{num_episodes}, Total Repeats={total_repeats}")
    
    return win_count > 0  # Return True if model won at least one episode

def train_expert():
    """Train the expert model with improved parameters."""
    # Parameters
    BOARD_SIZE = 6
    TIMESTEPS = 200000  # More training steps
    SEED = 42
    MODEL_PATH = "expert/improved_expert.zip"
    
    # Create directories
    os.makedirs("expert", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Create environment with very strong penalties for repeated actions
    logger.info(f"Creating environment with board size {BOARD_SIZE}x{BOARD_SIZE}")
    env = gym.make('Battleship-v0', 
                 board_size=(BOARD_SIZE, BOARD_SIZE),
                 reward_dictionary={
                     'win': 500,           # Very high win reward
                     'missed': 0,          # No change
                     'touched': 20,        # Higher reward for hits
                     'repeat_missed': -50, # Extreme penalty for repeated misses
                     'repeat_touched': -30 # Extreme penalty for repeated hits
                 })
    
    # Define network architecture with larger layers
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),  # Deeper network
    )
    
    # Create model with improved parameters
    logger.info("Creating PPO agent with improved parameters")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-3,        # Higher learning rate
        gamma=0.99,               # Discount factor
        n_steps=2048,             # Steps per update
        batch_size=256,           # Large batch size
        n_epochs=20,              # More epochs per update
        ent_coef=0.1,             # Higher entropy to encourage exploration
        clip_range=0.3,           # Higher clip range for more aggressive updates
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=SEED,
        tensorboard_log="./logs/"
    )
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path="./checkpoints/",
        name_prefix="battleship_improved",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    # Train the model
    logger.info(f"Training for {TIMESTEPS} timesteps")
    model.learn(
        total_timesteps=TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save the model
    logger.info(f"Saving model to {MODEL_PATH}")
    model.save(MODEL_PATH)
    
    # Test the model
    logger.info("Testing the trained model")
    model_works = test_model(model, board_size=BOARD_SIZE)
    
    if model_works:
        logger.info("Model successfully completed training and testing!")
    else:
        logger.warning("Model could not win any test games. Consider training longer or adjusting parameters.")
    
    return MODEL_PATH, model_works

if __name__ == "__main__":
    try:
        model_path, success = train_expert()
        if success:
            print(f"\nTraining completed successfully! Model saved to {model_path}")
            print("You can now use this model for distillation or evaluation.")
        else:
            print("\nTraining completed, but the model didn't perform well in testing.")
            print("Consider training with different parameters or for more timesteps.")
    except Exception as e:
        logger.error(f"Error training expert model: {e}")
        import traceback
        logger.error(traceback.format_exc())
