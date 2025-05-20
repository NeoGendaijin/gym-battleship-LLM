#!/usr/bin/env python
"""
Example script for evaluating Battleship agents.

This script evaluates agents (expert or student models) on the Battleship 
environment, calculating metrics such as win rate, average turns, and hit rate.
"""

import os
import argparse
import logging
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a Battleship agent")
    
    # Input/output arguments
    parser.add_argument(
        "--agent", 
        type=str, 
        required=True,
        help="Path to the agent model to evaluate (.zip for SB3, .pth for distilled models)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="eval/results",
        help="Directory to save evaluation results"
    )
    
    # Environment arguments
    parser.add_argument(
        "--variant", 
        type=str, 
        choices=["standard", "large_grid"],
        default="standard",
        help="Environment variant for evaluation"
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=500,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--render", 
        action="store_true",
        help="Render environment during evaluation (slower)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()

def load_agent(agent_path: str):
    """Load agent model based on file extension."""
    if agent_path.endswith('.zip'):
        # Load SB3 model
        try:
            from stable_baselines3 import PPO
            return PPO.load(agent_path)
        except ImportError:
            raise ImportError("Stable-Baselines3 is required to load .zip models")
    elif agent_path.endswith('.pth'):
        # Load PyTorch model
        try:
            # Import required modules locally
            import torch
            import torch.nn as nn
            
            class StudentPolicyWrapper:
                """Wrapper for PyTorch-based student policy models."""
                def __init__(self, model_path):
                    # Load model checkpoint
                    self.checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                    
                    # Determine model type and setup
                    self.model_type = self.checkpoint.get('model_type', 'unknown')
                    self.grid_size = self.checkpoint.get('grid_size', 10)
                    self.strategy_dim = self.checkpoint.get('strategy_dim', None)
                    
                    # Import required modules locally to avoid dependency issues
                    import torch.nn as nn
                    
                    # Define model architecture (must match the saved model)
                    class StudentPolicy(nn.Module):
                        def __init__(self, input_dim, hidden_dim, output_dim, strategy_dim):
                            super(StudentPolicy, self).__init__()
                            
                            self.strategy_dim = strategy_dim
                            
                            # Feature extraction
                            self.features = nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(input_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(),
                            )
                            
                            # Action prediction
                            self.action_head = nn.Sequential(
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, output_dim)
                            )
                            
                            # Strategy classification head (only for strategy model)
                            if strategy_dim is not None and strategy_dim > 0:
                                self.strategy_head = nn.Sequential(
                                    nn.Linear(hidden_dim, hidden_dim // 2),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim // 2, strategy_dim)
                                )
                        
                        def forward(self, x):
                            features = self.features(x)
                            action_logits = self.action_head(features)
                            
                            if self.strategy_dim is not None and self.strategy_dim > 0:
                                strategy_logits = self.strategy_head(features)
                                return action_logits, strategy_logits
                            else:
                                return action_logits
                    
                    # Create model and load state dict
                    hidden_dim = self.checkpoint.get('hidden_dim', 128)
                    self.model = StudentPolicy(
                        input_dim=self.grid_size*self.grid_size*2,
                        hidden_dim=hidden_dim,
                        output_dim=self.grid_size*self.grid_size,
                        strategy_dim=self.strategy_dim
                    )
                    
                    self.model.load_state_dict(self.checkpoint['state_dict'])
                    self.model.eval()  # Set to evaluation mode
                
                def predict(self, observation, deterministic=True):
                    """Predict action from observation."""
                    # Convert observation to tensor
                    obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                    
                    # Get model prediction
                    with torch.no_grad():
                        if self.strategy_dim is not None and self.strategy_dim > 0:
                            action_logits, _ = self.model(obs_tensor)
                        else:
                            action_logits = self.model(obs_tensor)
                        
                        # Convert to action
                        if deterministic:
                            action_idx = torch.argmax(action_logits, dim=1).item()
                        else:
                            # Sample from softmax distribution
                            probs = torch.softmax(action_logits, dim=1)
                            action_idx = torch.multinomial(probs, 1).item()
                        
                        # Convert 1D index to 2D coordinate
                        x = action_idx // self.grid_size
                        y = action_idx % self.grid_size
                        action = np.array([x, y])
                        
                        return action, None  # Return action and empty state (for compatibility)
            
            return StudentPolicyWrapper(agent_path)
        except Exception as e:
            raise RuntimeError(f"Error loading PyTorch model: {e}")
    else:
        raise ValueError(f"Unsupported model format: {agent_path}")

def create_environment(variant: str, seed: int):
    """Create environment based on variant."""
    import gymnasium as gym
    import gym_battleship
    
    if variant == "standard":
        env = gym.make('Battleship-v0')
    elif variant == "large_grid":
        # Create a custom environment with larger grid (12x12)
        env = gym.make('Battleship-LargeGrid-v0')
    else:
        raise ValueError(f"Unknown environment variant: {variant}")
    
    # No need to set seed separately in gymnasium - it's passed to reset
    
    return env

def evaluate_agent(agent, env, num_episodes: int = 500, render: bool = False) -> Dict[str, Any]:
    """Evaluate agent on environment."""
    # Metrics to track
    wins = 0
    total_turns = 0
    turn_counts = []
    hit_rates = []
    ship_hit_sequences = []
    
    for episode in range(num_episodes):
        if episode % 50 == 0:
            logger.info(f"Evaluating episode {episode}/{num_episodes}...")
        
        obs, info = env.reset(seed=episode)
        done = False
        episode_turns = 0
        hits = 0
        ship_hits = []
        
        while not done:
            # Predict action
            action, _ = agent.predict(obs, deterministic=True)
            
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_turns += 1
            
            # Track metrics
            if info.get("hit", False):
                hits += 1
                if info.get("sink", False):
                    ship_hits.append(episode_turns)
            
            # Render if requested
            if render:
                env.render()
        
        # Update metrics
        if info.get("win", False):
            wins += 1
        
        total_turns += episode_turns
        turn_counts.append(episode_turns)
        hit_rate = hits / episode_turns if episode_turns > 0 else 0
        hit_rates.append(hit_rate)
        ship_hit_sequences.append(ship_hits)
    
    # Calculate summary metrics
    win_rate = wins / num_episodes
    avg_turns = total_turns / num_episodes
    avg_hit_rate = sum(hit_rates) / num_episodes
    
    # Calculate distribution of turn counts
    turn_distribution = {}
    for turns in turn_counts:
        if turns not in turn_distribution:
            turn_distribution[turns] = 0
        turn_distribution[turns] += 1
    
    # Sort turn distribution by number of turns
    turn_distribution = {k: v for k, v in sorted(turn_distribution.items())}
    
    # Calculate ship hit patterns
    ship_patterns = {}
    for ship_hits in ship_hit_sequences:
        if len(ship_hits) > 0:
            pattern_key = ','.join(map(str, ship_hits))
            if pattern_key not in ship_patterns:
                ship_patterns[pattern_key] = 0
            ship_patterns[pattern_key] += 1
    
    # Get top ship patterns
    top_patterns = {k: v for k, v in sorted(ship_patterns.items(), key=lambda item: item[1], reverse=True)[:10]}
    
    return {
        "win_rate": win_rate,
        "avg_turns": avg_turns,
        "avg_hit_rate": avg_hit_rate,
        "turn_counts": turn_counts,
        "turn_distribution": turn_distribution,
        "top_patterns": top_patterns
    }

def plot_results(results: Dict[str, Any], output_path: str):
    """Generate plots from evaluation results."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Plot turn distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    turns = list(results["turn_distribution"].keys())
    counts = list(results["turn_distribution"].values())
    ax.bar(turns, counts)
    ax.set_title("Distribution of Game Lengths")
    ax.set_xlabel("Number of Turns")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_path}_turn_dist.png")
    
    # Plot turn count histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(results["turn_counts"], bins=30, alpha=0.7)
    ax.axvline(results["avg_turns"], color='r', linestyle='--', label=f"Average: {results['avg_turns']:.2f} turns")
    ax.set_title("Histogram of Game Lengths")
    ax.set_xlabel("Number of Turns")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_path}_turn_hist.png")
    
    plt.close('all')

def main():
    args = parse_args()
    
    try:
        # Set random seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load agent
        logger.info(f"Loading agent from {args.agent}...")
        agent = load_agent(args.agent)
        
        # Create environment
        logger.info(f"Creating environment variant: {args.variant}...")
        env = create_environment(args.variant, args.seed)
        
        # Evaluate agent
        logger.info(f"Evaluating agent over {args.episodes} episodes...")
        start_time = datetime.now()
        results = evaluate_agent(agent, env, args.episodes, args.render)
        eval_time = (datetime.now() - start_time).total_seconds()
        
        # Add evaluation metadata
        results["metadata"] = {
            "agent_path": args.agent,
            "environment": args.variant,
            "episodes": args.episodes,
            "seed": args.seed,
            "eval_time": eval_time,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Print summary results
        logger.info("===== Evaluation Results =====")
        logger.info(f"Win Rate: {results['win_rate']:.4f} ({results['win_rate'] * 100:.1f}%)")
        logger.info(f"Average Turns: {results['avg_turns']:.2f}")
        logger.info(f"Average Hit Rate: {results['avg_hit_rate']:.4f}")
        logger.info(f"Evaluation Time: {eval_time:.2f} seconds")
        
        # Determine model type from file extension
        model_type = "expert" if args.agent.endswith(".zip") else "student"
        if model_type == "student":
            student_type = "unknown"
            if agent.checkpoint.get("model_type") == "strategy":
                student_type = "strategy"
            elif agent.checkpoint.get("model_type") == "kl":
                student_type = "kl"
            logger.info(f"Student Model Type: {student_type}")
            model_type = f"student_{student_type}"
        
        # Save results
        agent_name = os.path.splitext(os.path.basename(args.agent))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.output_dir, f"{agent_name}_{args.variant}_{timestamp}")
        
        with open(f"{output_path}.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate plots
        plot_results(results, output_path)
        
        logger.info(f"Results saved to {output_path}.json")
        logger.info(f"Plots saved to {output_path}_*.png")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error evaluating agent: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())
