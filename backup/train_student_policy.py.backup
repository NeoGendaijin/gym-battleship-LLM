#!/usr/bin/env python
"""
Example script for training student policies from expert Battleship strategies.

This script demonstrates how to train student policies using either:
1. Strategy supervision from LPML annotations (strategy-based distillation)
2. KL divergence from an expert model (KL-based distillation)
"""

import os
import argparse
import logging
import torch
from gym_battleship.lpml import LPMLParser
from gym_battleship.distill import StrategyDistiller, KLDistiller

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a student policy from expert Battleship strategies")
    
    # Input/output arguments
    parser.add_argument(
        "--lpml", 
        type=str, 
        default=None,
        help="Path to the LPML XML file for strategy-based distillation"
    )
    parser.add_argument(
        "--teacher", 
        type=str, 
        default=None,
        help="Path to the teacher model for KL-based distillation"
    )
    parser.add_argument(
        "--out", 
        type=str, 
        required=True,
        help="Path to save the trained student model"
    )
    
    # Training arguments
    parser.add_argument(
        "--model-type", 
        type=str, 
        choices=["strategy", "kl"],
        required=True,
        help="Type of distillation to perform: 'strategy' or 'kl'"
    )
    parser.add_argument(
        "--grid-size", 
        type=int, 
        default=10,
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
        "--strategy-weight", 
        type=float, 
        default=0.5,
        help="Weight for strategy classification loss (only for strategy-based distillation)"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=1.0,
        help="Temperature for KL divergence (only for KL-based distillation)"
    )
    parser.add_argument(
        "--n-clusters", 
        type=int, 
        default=5,
        help="Number of strategy clusters (only for strategy-based distillation)"
    )
    
    return parser.parse_args()

def train_strategy_model(args):
    """Train a student policy using strategy-based distillation."""
    if not args.lpml:
        raise ValueError("LPML file path must be provided for strategy-based distillation")
    
    logger.info(f"Parsing LPML annotations from: {args.lpml}")
    parser = LPMLParser()
    lpml_data = parser.parse_and_prepare(args.lpml, args.grid_size, args.n_clusters)
    
    logger.info(f"Creating strategy distiller with grid size: {args.grid_size}, hidden dim: {args.hidden_dim}")
    distiller = StrategyDistiller(grid_size=args.grid_size, hidden_dim=args.hidden_dim)
    
    logger.info(f"Training student policy for {args.epochs} epochs...")
    logger.info(f"Strategy clusters: {lpml_data['n_clusters']}")
    model, history = distiller.distill_from_lpml(
        lpml_data=lpml_data,
        epochs=args.epochs,
        lr=args.lr,
        strategy_weight=args.strategy_weight,
        output_path=args.out
    )
    
    logger.info(f"Final training loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    logger.info(f"Final training action accuracy: {history['train_action_acc'][-1]:.4f}")
    logger.info(f"Final validation action accuracy: {history['val_action_acc'][-1]:.4f}")
    
    if "train_strategy_acc" in history:
        logger.info(f"Final training strategy accuracy: {history['train_strategy_acc'][-1]:.4f}")
        logger.info(f"Final validation strategy accuracy: {history['val_strategy_acc'][-1]:.4f}")
    
    logger.info(f"Student policy saved to: {args.out}")
    return model, history

def train_kl_model(args):
    """Train a student policy using KL-based distillation."""
    if not args.teacher:
        raise ValueError("Teacher model path must be provided for KL-based distillation")
    
    logger.info(f"Creating KL distiller with grid size: {args.grid_size}, hidden dim: {args.hidden_dim}")
    distiller = KLDistiller(grid_size=args.grid_size, hidden_dim=args.hidden_dim)
    
    try:
        from stable_baselines3 import PPO
        logger.info(f"Loading teacher model from: {args.teacher}")
        teacher_model = PPO.load(args.teacher)
        
        # Create environment factory for generating trajectories
        def env_factory():
            import gymnasium as gym
            import gym_battleship
            return gym.make('Battleship-v0', board_size=(args.grid_size, args.grid_size))
        
        logger.info(f"Training student policy for {args.epochs} epochs...")
        model, history = distiller.distill_from_expert(
            expert_model=teacher_model,
            env_factory=env_factory,
            n_trajectories=100,  # Generate 100 trajectories
            epochs=args.epochs,
            lr=args.lr,
            temperature=args.temperature,
            output_path=args.out
        )
        
        logger.info(f"Final training KL loss: {history['train_loss'][-1]:.6f}")
        logger.info(f"Final validation KL loss: {history['val_loss'][-1]:.6f}")
        
        logger.info(f"Student policy saved to: {args.out}")
        return model, history
    
    except ImportError:
        logger.error("Stable-Baselines3 is required for KL-based distillation")
        raise
        
def main():
    args = parse_args()
    
    try:
        # Set up device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        
        # Train the appropriate model type
        if args.model_type == "strategy":
            train_strategy_model(args)
        elif args.model_type == "kl":
            train_kl_model(args)
        else:
            raise ValueError(f"Invalid model type: {args.model_type}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error training student policy: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())
