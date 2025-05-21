#!/usr/bin/env python
"""
Compare LPML-guided LLM agent vs vanilla LLM agent in Battleship.

This script evaluates two different approaches:
1. LPML-guided LLM: An agent that has access to annotated strategies from LPML
2. Vanilla LLM: A standard LLM agent that makes decisions without LPML guidance

The comparison highlights the value of structured strategic knowledge (LPML) vs
raw LLM inference in the context of the Battleship game.
"""

import os
import argparse
import gymnasium as gym
import importlib
import gym_battleship
importlib.reload(gym_battleship)
import torch
import numpy as np
from tqdm import tqdm
import asyncio
from openai import AsyncOpenAI
import json
import xml.etree.ElementTree as ET

# --- LPML-guided LLM agent (uses LPML annotations for better decisions) ---
class LPMLGuidedAgent:
    def __init__(self, lpml_file, api_key=None, model="gpt-4o"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be set in OPENAI_API_KEY env or passed to the agent")
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model
        
        # Load and parse LPML strategies
        self.strategies = self.load_lpml_strategies(lpml_file)
        
    def load_lpml_strategies(self, lpml_file):
        """Extract strategies from LPML file."""
        if not os.path.exists(lpml_file):
            raise FileNotFoundError(f"LPML file not found: {lpml_file}")
        
        strategies = []
        try:
            tree = ET.parse(lpml_file)
            root = tree.getroot()
            
            # Extract all strategy elements
            for trajectory in root.findall(".//Trajectory"):
                for annotation in trajectory.findall(".//Annotation"):
                    strategy = {}
                    
                    # Extract the key components
                    condition = annotation.find(".//Condition")
                    if condition is not None:
                        strategy["condition"] = condition.text.strip()
                        
                    thought = annotation.find(".//Thought")
                    if thought is not None:
                        strategy["thought"] = thought.text.strip()
                        
                    execution = annotation.find(".//Execution")
                    if execution is not None:
                        strategy["execution"] = execution.text.strip()
                    
                    if strategy:
                        strategies.append(strategy)
        except Exception as e:
            print(f"Error parsing LPML file: {e}")
        
        print(f"Loaded {len(strategies)} strategies from LPML")
        return strategies

    def obs_to_prompt(self, obs):
        """Convert observation to prompt with LPML strategic guidance."""
        # Create observation visualization
        view_board = np.zeros_like(obs[0])
        if obs.ndim == 3:
            view_board[obs[0] == 1] = 1    # hits
            view_board[obs[1] == 1] = -1   # misses
        else:
            view_board = obs
        board_str = "\n".join(" ".join(str(int(cell)) for cell in row) for row in view_board)
        
        # Create the strategy library portion of the prompt
        strategies_text = ""
        for i, strategy in enumerate(self.strategies[:10]):  # Limit to 10 strategies to keep prompt manageable
            strategy_text = f"Strategy {i+1}:\n"
            if "condition" in strategy:
                strategy_text += f"When: {strategy['condition']}\n"
            if "thought" in strategy:
                strategy_text += f"Think: {strategy['thought']}\n"
            if "execution" in strategy:
                strategy_text += f"Do: {strategy['execution']}\n"
            strategies_text += strategy_text + "\n"
        
        # Build the complete prompt
        prompt = (
            "You are a Battleship strategy agent with access to expert strategies.\n\n"
            "GAME STATE\n"
            "Here is your current observation grid (0: unexplored, 1: hit, -1: miss):\n"
            f"{board_str}\n\n"
            "STRATEGY LIBRARY\n"
            f"{strategies_text}\n"
            "INSTRUCTIONS\n"
            "1. Analyze the current game state\n"
            "2. Select the most appropriate strategy from the library above\n"
            "3. Apply that strategy to determine your next move\n"
            "4. Output only a single Python tuple with your chosen (x, y) coordinate, e.g., (3, 5)\n"
            "No explanation, no other text, no code block, just the coordinate tuple."
        )
        return prompt

    async def act(self, obs):
        prompt = self.obs_to_prompt(obs)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a Battleship strategy agent with access to expert knowledge."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=16,
        )
        text = response.choices[0].message.content.strip()
        import re
        m = re.search(r"\((\d+),\s*(\d+)\)", text)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        else:
            raise ValueError(f"Agent did not return a valid (x, y) tuple: {text}")

# --- Vanilla LLM agent (just basic LLM without LPML guidance) ---
class VanillaLLMAgent:
    def __init__(self, api_key=None, model="gpt-4o"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be set in OPENAI_API_KEY env or passed to the agent")
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model

    def obs_to_prompt(self, obs):
        """Convert observation to prompt WITHOUT strategic guidance."""
        # Create observation visualization
        view_board = np.zeros_like(obs[0])
        if obs.ndim == 3:
            view_board[obs[0] == 1] = 1    # hits
            view_board[obs[1] == 1] = -1   # misses
        else:
            view_board = obs
        board_str = "\n".join(" ".join(str(int(cell)) for cell in row) for row in view_board)
        
        # Basic prompt without LPML strategies
        prompt = (
            "You are playing Battleship.\n\n"
            "Here is your current observation grid (0: unexplored, 1: hit, -1: miss):\n"
            f"{board_str}\n\n"
            "What is the most reasonable (x, y) coordinate to attack next?\n"
            "Output only a single Python tuple, e.g., (3, 5). No explanation, no other text, no code block, no comments."
        )
        return prompt

    async def act(self, obs):
        prompt = self.obs_to_prompt(obs)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a Battleship player."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=16,
        )
        text = response.choices[0].message.content.strip()
        import re
        m = re.search(r"\((\d+),\s*(\d+)\)", text)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        else:
            raise ValueError(f"Agent did not return a valid (x, y) tuple: {text}")

# --- Battle simulation ---
async def run_battle_async(lpml_agent, vanilla_agent, n_episodes=10, results_path=None, board_size=6):
    lpml_wins = 0
    vanilla_wins = 0
    all_results = []
    
    if results_path is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../eval/results")
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f"{os.path.basename(__file__).split('.')[0]}_results.json")
    
    for ep in tqdm(range(n_episodes), desc="Episodes"):
        env = gym.make('Battleship-v0', board_size=(board_size, board_size))
        obs, info = env.reset()
        terminated = False
        truncated = False
        turn = 0
        winner = None
        episode_log = {
            "turns": [],
            "winner": None
        }
        try:
            while not (terminated or truncated):
                if turn % 2 == 0:
                    agent_name = "lpml"
                    action = await lpml_agent.act(obs)
                else:
                    agent_name = "vanilla"
                    action = await vanilla_agent.act(obs)
                
                obs_next, reward, terminated, truncated, info = env.step(action)
                
                # Visualization
                print(f"[Episode {ep+1} | Turn {turn+1} | {agent_name}] Action: {action}, Reward: {reward}")
                print("Board (visualization):")
                view_board = np.zeros_like(obs[0])
                if obs.ndim == 3:
                    view_board[obs[0] == 1] = 1    # hits
                    view_board[obs[1] == 1] = -1   # misses
                else:
                    view_board = obs
                for row in view_board:
                    print(" ".join(f"{int(cell):2d}" for cell in row))
                print("-" * 30)
                
                # Track results
                episode_log["turns"].append({
                    "turn": turn+1,
                    "agent": agent_name,
                    "action": action,
                    "reward": reward,
                    "board": view_board.tolist()
                })
                
                obs = obs_next
                # Determine winner if game ended
                winner = agent_name if terminated else None
                turn += 1
        except Exception as e:
            print(f"Error in episode {ep+1}: {e}")
            if winner is None:
                # Assign win to the agent that didn't cause the error
                winner = "vanilla" if turn % 2 == 0 else "lpml"
                print(f"Assigning win to {winner} due to error")
        
        episode_log["winner"] = winner
        all_results.append(episode_log)
        
        # Count wins
        if winner == 'lpml':
            lpml_wins += 1
        elif winner == 'vanilla':
            vanilla_wins += 1
    
    # Calculate percentages
    lpml_win_pct = lpml_wins / n_episodes * 100
    vanilla_win_pct = vanilla_wins / n_episodes * 100
    draws = n_episodes - lpml_wins - vanilla_wins
    
    print(f'LPML-guided Agent wins: {lpml_wins}/{n_episodes} ({lpml_win_pct:.1f}%)')
    print(f'Vanilla LLM Agent wins: {vanilla_wins}/{n_episodes} ({vanilla_win_pct:.1f}%)')
    print(f'Draws: {draws}/{n_episodes} ({draws/n_episodes*100:.1f}%)')
    
    # Save results
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "lpml_wins": lpml_wins,
            "vanilla_wins": vanilla_wins,
            "draws": draws,
            "lpml_win_percentage": lpml_win_pct,
            "vanilla_win_percentage": vanilla_win_pct,
            "episodes": all_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {results_path}")
    return lpml_wins, vanilla_wins

def parse_args():
    parser = argparse.ArgumentParser(description="Compare LPML-guided LLM vs vanilla LLM in Battleship")
    parser.add_argument(
        "--lpml-file",
        type=str,
        default="lpml/heuristic_battleship.xml",
        help="Path to the LPML annotation file"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes for comparison"
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=6,
        help="Size of the Battleship board"
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use for both agents"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save results JSON (default: eval/results/lpml_vs_llm_results.json)"
    )
    return parser.parse_args()

async def main_async():
    args = parse_args()
    
    # Ensure LPML file exists
    if not os.path.exists(args.lpml_file):
        available_xmls = []
        lpml_dir = os.path.dirname(args.lpml_file)
        if os.path.exists(lpml_dir):
            available_xmls = [f for f in os.listdir(lpml_dir) if f.endswith('.xml')]
        
        if available_xmls:
            print(f"LPML file {args.lpml_file} not found. Available XML files in {lpml_dir}:")
            for xml in available_xmls:
                print(f"  - {os.path.join(lpml_dir, xml)}")
            # Use the first available XML as fallback
            args.lpml_file = os.path.join(lpml_dir, available_xmls[0])
            print(f"Using {args.lpml_file} instead.")
        else:
            print(f"Error: No LPML XML files found in {lpml_dir or '.'}")
            return 1
    
    print(f"=== LPML-guided LLM vs Vanilla LLM comparison ===")
    print(f"LPML file: {args.lpml_file}")
    print(f"OpenAI model: {args.openai_model}")
    print(f"Board size: {args.board_size}x{args.board_size}")
    print(f"Episodes: {args.episodes}")
    
    # Create agents
    try:
        lpml_agent = LPMLGuidedAgent(args.lpml_file, model=args.openai_model)
        vanilla_agent = VanillaLLMAgent(model=args.openai_model)
    except Exception as e:
        print(f"Error creating agents: {e}")
        return 1
    
    # Run comparison
    try:
        lpml_wins, vanilla_wins = await run_battle_async(
            lpml_agent,
            vanilla_agent,
            n_episodes=args.episodes,
            results_path=args.output_file,
            board_size=args.board_size
        )
    except Exception as e:
        print(f"Error during battle simulation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

def main():
    """
    Wrapper function to run the async main function.
    """
    return asyncio.run(main_async())

if __name__ == '__main__':
    exit(main())
