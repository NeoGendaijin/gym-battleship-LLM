#!/usr/bin/env python
"""
Analyze and visualize results from LPML vs vanilla LLM battles.

This script generates figures comparing the performance of LPML-guided LLM agents
versus vanilla LLM agents in the Battleship game. It provides insights into:
- Win rates
- Average reward per turn
- Hit rate comparison
- Board state visualizations
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze LPML vs vanilla LLM Battleship results")
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="Path to specific results JSON file (default: latest in eval/results/)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples/figures",
        help="Directory to save figures to"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figures"
    )
    return parser.parse_args()

def find_latest_results(results_pattern="eval/results/lpml_vs_llm_results*.json"):
    """Find the most recent results file matching the pattern."""
    files = glob.glob(results_pattern)
    if not files:
        # Try looking in the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(script_dir)
        files = glob.glob(os.path.join(base_dir, results_pattern))
        
    if not files:
        return None
    
    # Sort by modification time (most recent first)
    return max(files, key=os.path.getmtime)

def load_results(results_file):
    """Load results from a JSON file."""
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded results from {results_file}")
    return data

def generate_win_rate_chart(data, output_dir, dpi=300):
    """Generate win rate comparison chart."""
    total_episodes = data["lpml_wins"] + data["vanilla_wins"] + data.get("draws", 0)
    
    labels = ['LPML-guided LLM', 'Vanilla LLM', 'Draws']
    values = [
        data["lpml_wins"] / total_episodes * 100,
        data["vanilla_wins"] / total_episodes * 100,
        data.get("draws", 0) / total_episodes * 100
    ]
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    colors = ['#3498db', '#e74c3c', '#95a5a6']
    bars = ax.bar(labels, values, color=colors)
    
    # Add values on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=12)
    
    # Add number of wins in the bars
    ax.annotate(f'({data["lpml_wins"]} wins)',
               xy=(bars[0].get_x() + bars[0].get_width() / 2, bars[0].get_height() / 2),
               ha='center', va='center', color='white', fontweight='bold')
    
    ax.annotate(f'({data["vanilla_wins"]} wins)',
               xy=(bars[1].get_x() + bars[1].get_width() / 2, bars[1].get_height() / 2),
               ha='center', va='center', color='white', fontweight='bold')
    
    if data.get("draws", 0) > 0:
        ax.annotate(f'({data.get("draws", 0)} draws)',
                   xy=(bars[2].get_x() + bars[2].get_width() / 2, bars[2].get_height() / 2),
                   ha='center', va='center', color='white', fontweight='bold')
    
    # Customize the chart
    ax.set_ylabel('Win Rate (%)', fontsize=14)
    ax.set_title('LPML-guided vs Vanilla LLM Win Rates', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 100)
    
    # Add percentage sign to y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x)}%"))
    
    # Add a horizontal line at 50%
    ax.axhline(y=50, color='grey', linestyle='--', alpha=0.7)
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add text showing the advantage
    if data["lpml_wins"] > data["vanilla_wins"]:
        advantage = data["lpml_wins"] / max(1, data["vanilla_wins"])
        ax.text(0.5, 0.02, 
                f'LPML-guided LLM wins {advantage:.1f}x more often than Vanilla LLM',
                horizontalalignment='center',
                verticalalignment='bottom',
                transform=ax.transAxes,
                fontsize=14,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'lpml_vs_llm_win_rates.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    
    print(f"Win rate chart saved to {output_path}")
    return output_path

def analyze_hit_rates(data):
    """Analyze hit rates for both agent types."""
    hit_counts_lpml = []
    hit_counts_vanilla = []
    reward_lpml = []
    reward_vanilla = []
    
    # Process each episode
    for episode in data["episodes"]:
        lpml_hits = 0
        vanilla_hits = 0
        lpml_reward = 0
        vanilla_reward = 0
        
        # Process each turn
        for turn in episode["turns"]:
            agent = turn["agent"]
            reward = turn["reward"]
            
            # Count hits (reward > 0 typically indicates a hit)
            if agent == "lpml":
                lpml_reward += reward
                if reward > 0:  # This is a hit
                    lpml_hits += 1
            elif agent == "vanilla":
                vanilla_reward += reward
                if reward > 0:  # This is a hit
                    vanilla_hits += 1
        
        # Store hit counts and rewards for this episode
        hit_counts_lpml.append(lpml_hits)
        hit_counts_vanilla.append(vanilla_hits)
        reward_lpml.append(lpml_reward)
        reward_vanilla.append(vanilla_reward)
    
    # Calculate overall statistics
    result = {
        "lpml_total_hits": sum(hit_counts_lpml),
        "vanilla_total_hits": sum(hit_counts_vanilla),
        "lpml_avg_hits": np.mean(hit_counts_lpml),
        "vanilla_avg_hits": np.mean(hit_counts_vanilla),
        "lpml_total_reward": sum(reward_lpml),
        "vanilla_total_reward": sum(reward_vanilla),
        "lpml_avg_reward": np.mean(reward_lpml),
        "vanilla_avg_reward": np.mean(reward_vanilla),
        "hit_counts_lpml": hit_counts_lpml,
        "hit_counts_vanilla": hit_counts_vanilla,
        "reward_lpml": reward_lpml,
        "reward_vanilla": reward_vanilla
    }
    
    return result

def generate_hit_rate_chart(hit_data, output_dir, dpi=300):
    """Generate hit rate comparison chart."""
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set up data
    labels = ['LPML-guided LLM', 'Vanilla LLM']
    values = [hit_data["lpml_avg_hits"], hit_data["vanilla_avg_hits"]]
    
    # Create bar chart
    colors = ['#3498db', '#e74c3c']
    bars = ax.bar(labels, values, color=colors)
    
    # Add values on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=12)
    
    # Add total hits in the bars
    ax.annotate(f'(Total: {hit_data["lpml_total_hits"]} hits)',
               xy=(bars[0].get_x() + bars[0].get_width() / 2, bars[0].get_height() / 2),
               ha='center', va='center', color='white', fontweight='bold')
    
    ax.annotate(f'(Total: {hit_data["vanilla_total_hits"]} hits)',
               xy=(bars[1].get_x() + bars[1].get_width() / 2, bars[1].get_height() / 2),
               ha='center', va='center', color='white', fontweight='bold')
    
    # Customize the chart
    ax.set_ylabel('Average Hits per Game', fontsize=14)
    ax.set_title('LPML-guided vs Vanilla LLM Hit Rates', fontsize=16, fontweight='bold')
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add text showing the advantage
    if hit_data["lpml_avg_hits"] > hit_data["vanilla_avg_hits"]:
        advantage = hit_data["lpml_avg_hits"] / max(0.001, hit_data["vanilla_avg_hits"])
        ax.text(0.5, 0.02, 
                f'LPML-guided LLM gets {advantage:.2f}x more hits than Vanilla LLM',
                horizontalalignment='center',
                verticalalignment='bottom',
                transform=ax.transAxes,
                fontsize=14,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'lpml_vs_llm_hit_rates.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    
    print(f"Hit rate chart saved to {output_path}")
    return output_path

def generate_reward_chart(hit_data, output_dir, dpi=300):
    """Generate reward comparison chart."""
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set up data
    labels = ['LPML-guided LLM', 'Vanilla LLM']
    values = [hit_data["lpml_avg_reward"], hit_data["vanilla_avg_reward"]]
    
    # Create bar chart
    colors = ['#3498db', '#e74c3c']
    bars = ax.bar(labels, values, color=colors)
    
    # Add values on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=12)
    
    # Add total rewards in the bars
    ax.annotate(f'(Total: {hit_data["lpml_total_reward"]:.1f})',
               xy=(bars[0].get_x() + bars[0].get_width() / 2, bars[0].get_height() / 2),
               ha='center', va='center', color='white', fontweight='bold')
    
    ax.annotate(f'(Total: {hit_data["vanilla_total_reward"]:.1f})',
               xy=(bars[1].get_x() + bars[1].get_width() / 2, bars[1].get_height() / 2),
               ha='center', va='center', color='white', fontweight='bold')
    
    # Customize the chart
    ax.set_ylabel('Average Reward per Game', fontsize=14)
    ax.set_title('LPML-guided vs Vanilla LLM Rewards', fontsize=16, fontweight='bold')
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add text showing the advantage
    if hit_data["lpml_avg_reward"] > hit_data["vanilla_avg_reward"]:
        advantage = hit_data["lpml_avg_reward"] / max(0.001, hit_data["vanilla_avg_reward"])
        ax.text(0.5, 0.02, 
                f'LPML-guided LLM earns {advantage:.2f}x more reward than Vanilla LLM',
                horizontalalignment='center',
                verticalalignment='bottom',
                transform=ax.transAxes,
                fontsize=14,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'lpml_vs_llm_rewards.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    
    print(f"Reward chart saved to {output_path}")
    return output_path

def generate_board_heatmaps(data, output_dir, dpi=300):
    """Generate heatmaps showing where each agent tends to hit."""
    # Initialize counters for hit locations
    episodes = data["episodes"]
    if not episodes:
        return None
    
    # Determine board dimensions from the first episode
    first_board = np.array(episodes[0]["turns"][0]["board"])
    board_size = first_board.shape[0]
    
    # Initialize hit counters
    lpml_hits = np.zeros((board_size, board_size))
    vanilla_hits = np.zeros((board_size, board_size))
    lpml_misses = np.zeros((board_size, board_size))
    vanilla_misses = np.zeros((board_size, board_size))
    
    # Process each episode
    for episode in episodes:
        current_board = np.zeros((board_size, board_size))
        
        # Process each turn
        for turn in episode["turns"]:
            agent = turn["agent"]
            action = turn["action"]
            reward = turn["reward"]
            board = np.array(turn["board"])
            
            # Extract x, y coordinates from the action
            if isinstance(action, list):
                x, y = action
            else:
                # Handle string representation like "(3, 4)"
                import re
                match = re.search(r"\((\d+),\s*(\d+)\)", str(action))
                if match:
                    x, y = int(match.group(1)), int(match.group(2))
                else:
                    continue
            
            # Skip if out of bounds
            if x >= board_size or y >= board_size:
                continue
            
            # Check for hit or miss
            is_hit = reward > 0
            
            if agent == "lpml":
                if is_hit:
                    lpml_hits[y, x] += 1
                else:
                    lpml_misses[y, x] += 1
            elif agent == "vanilla":
                if is_hit:
                    vanilla_hits[y, x] += 1
                else:
                    vanilla_misses[y, x] += 1
    
    # Calculate hit rates (avoiding division by zero)
    lpml_total = lpml_hits + lpml_misses
    vanilla_total = vanilla_hits + vanilla_misses
    
    lpml_hit_rate = np.zeros((board_size, board_size))
    vanilla_hit_rate = np.zeros((board_size, board_size))
    
    for i in range(board_size):
        for j in range(board_size):
            if lpml_total[i, j] > 0:
                lpml_hit_rate[i, j] = lpml_hits[i, j] / lpml_total[i, j]
            if vanilla_total[i, j] > 0:
                vanilla_hit_rate[i, j] = vanilla_hits[i, j] / vanilla_total[i, j]
    
    # Create heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define custom colormaps
    hit_cmap = LinearSegmentedColormap.from_list('hit_cmap', ['#ffffff', '#3498db'])
    miss_cmap = LinearSegmentedColormap.from_list('miss_cmap', ['#ffffff', '#e74c3c'])
    rate_cmap = LinearSegmentedColormap.from_list('rate_cmap', ['#ffffff', '#2ecc71'])
    
    # Plot hit counts
    im1 = axes[0, 0].imshow(lpml_hits, cmap=hit_cmap)
    axes[0, 0].set_title('LPML-guided LLM Hit Counts', fontsize=14)
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(vanilla_hits, cmap=hit_cmap)
    axes[0, 1].set_title('Vanilla LLM Hit Counts', fontsize=14)
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot hit rates
    im3 = axes[1, 0].imshow(lpml_hit_rate, cmap=rate_cmap, vmin=0, vmax=1)
    axes[1, 0].set_title('LPML-guided LLM Hit Rate', fontsize=14)
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(vanilla_hit_rate, cmap=rate_cmap, vmin=0, vmax=1)
    axes[1, 1].set_title('Vanilla LLM Hit Rate', fontsize=14)
    plt.colorbar(im4, ax=axes[1, 1])
    
    # Add grid lines and labels
    for ax in axes.flat:
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, board_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, board_size, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1, alpha=0.2)
        
        # Add coordinates
        ax.set_xticks(np.arange(board_size))
        ax.set_yticks(np.arange(board_size))
        ax.set_xticklabels(np.arange(board_size))
        ax.set_yticklabels(np.arange(board_size))
    
    plt.suptitle('LPML-guided vs Vanilla LLM Board Analysis', fontsize=16, fontweight='bold')
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'lpml_vs_llm_board_heatmaps.png')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    
    print(f"Board heatmaps saved to {output_path}")
    return output_path

def generate_summary_report(data, hit_data, output_dir):
    """Generate a summary text report with key findings."""
    total_episodes = data["lpml_wins"] + data["vanilla_wins"] + data.get("draws", 0)
    
    # Calculate improvements and advantages
    lpml_win_rate = data["lpml_wins"] / total_episodes * 100
    vanilla_win_rate = data["vanilla_wins"] / total_episodes * 100
    
    if vanilla_win_rate > 0:
        win_improvement = (lpml_win_rate - vanilla_win_rate) / vanilla_win_rate * 100
    else:
        win_improvement = float('inf')
    
    if hit_data["vanilla_avg_hits"] > 0:
        hit_improvement = (hit_data["lpml_avg_hits"] - hit_data["vanilla_avg_hits"]) / hit_data["vanilla_avg_hits"] * 100
    else:
        hit_improvement = float('inf')
    
    if hit_data["vanilla_avg_reward"] > 0:
        reward_improvement = (hit_data["lpml_avg_reward"] - hit_data["vanilla_avg_reward"]) / hit_data["vanilla_avg_reward"] * 100
    else:
        reward_improvement = float('inf')
    
    # Generate report text
    report = f"""# LPML-guided LLM vs Vanilla LLM Analysis

## Overview
- Total Episodes: {total_episodes}
- LPML-guided Agent Wins: {data["lpml_wins"]} ({lpml_win_rate:.1f}%)
- Vanilla LLM Agent Wins: {data["vanilla_wins"]} ({vanilla_win_rate:.1f}%)
- Draws: {data.get("draws", 0)} ({data.get("draws", 0) / total_episodes * 100:.1f}%)

## Performance Metrics

### Hit Statistics
- LPML-guided Agent Average Hits: {hit_data["lpml_avg_hits"]:.2f}
- Vanilla LLM Agent Average Hits: {hit_data["vanilla_avg_hits"]:.2f}
- LPML Improvement: {'+' if hit_improvement > 0 else ''}{hit_improvement:.1f}%

### Reward Statistics
- LPML-guided Agent Average Reward: {hit_data["lpml_avg_reward"]:.2f}
- Vanilla LLM Agent Average Reward: {hit_data["vanilla_avg_reward"]:.2f}
- LPML Improvement: {'+' if reward_improvement > 0 else ''}{reward_improvement:.1f}%

## Conclusion

The analysis demonstrates that **{"LPML-guided LLM significantly outperforms Vanilla LLM" if lpml_win_rate > vanilla_win_rate + 10 else "LPML-guided LLM performs better than Vanilla LLM" if lpml_win_rate > vanilla_win_rate else "Vanilla LLM performs better than LPML-guided LLM" if vanilla_win_rate > lpml_win_rate else "LPML-guided LLM and Vanilla LLM perform similarly"}** in the Battleship environment.

The results suggest that {"structured strategic knowledge from LPML annotations provides a substantial advantage" if lpml_win_rate > vanilla_win_rate + 20 else "LPML annotations contribute to improved decision-making" if lpml_win_rate > vanilla_win_rate else "LPML annotations do not significantly improve performance in this scenario"}.

This highlights the {"importance of strategic guidance in enhancing LLM performance for complex decision-making tasks" if lpml_win_rate > vanilla_win_rate else "need for further refinement of the LPML annotation approach"}.
"""
    
    # Save the report
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'lpml_vs_llm_analysis.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Summary report saved to {output_path}")
    return output_path, report

def main():
    args = parse_args()
    
    # Find results file if not specified
    results_file = args.results_file
    if not results_file:
        results_file = find_latest_results()
        if not results_file:
            print("No results file found. Please specify one with --results-file")
            return 1
    
    # Load and analyze results
    try:
        data = load_results(results_file)
        
        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate figures
        win_chart = generate_win_rate_chart(data, args.output_dir, args.dpi)
        
        # Analyze hit rates
        hit_data = analyze_hit_rates(data)
        hit_chart = generate_hit_rate_chart(hit_data, args.output_dir, args.dpi)
        reward_chart = generate_reward_chart(hit_data, args.output_dir, args.dpi)
        
        # Generate board heatmaps
        heatmaps = generate_board_heatmaps(data, args.output_dir, args.dpi)
        
        # Generate summary report
        report_path, report_text = generate_summary_report(data, hit_data, args.output_dir)
        
        # Print summary to console
        print("\n" + "="*80)
        print(report_text)
        print("="*80)
        
        print(f"\nAll analysis results saved to {args.output_dir}")
        print(f"  - Win rate chart: {win_chart}")
        print(f"  - Hit rate chart: {hit_chart}")
        print(f"  - Reward chart: {reward_chart}")
        if heatmaps:
            print(f"  - Board heatmaps: {heatmaps}")
        print(f"  - Summary report: {report_path}")
        
        return 0
    
    except Exception as e:
        print(f"Error analyzing results: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
