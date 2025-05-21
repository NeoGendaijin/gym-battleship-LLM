import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "rag_vs_student_results.json")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# データ読み込み
with open(RESULTS_PATH, "r") as f:
    results = json.load(f)

# データをpandas DataFrameに変換
records = []
for game_idx, game in enumerate(results):
    for turn in game["turns"]:
        records.append({
            "game": game_idx,
            "turn": turn["turn"],
            "agent": turn["agent"],
            "action_x": turn["action"][0],
            "action_y": turn["action"][1],
            "reward": turn["reward"],
            "winner": game.get("winner"),
        })
df = pd.DataFrame(records)

# 1. Average reward per turn for each agent
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="turn", y="reward", hue="agent", ci="sd")
plt.title("Average Reward per Turn")
plt.xlabel("Turn")
plt.ylabel("Reward")
plt.legend(title="Agent")
plt.savefig(os.path.join(FIG_DIR, "reward_lineplot.png"))
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="agent", y="reward")
plt.title("Reward Distribution (Boxplot)")
plt.xlabel("Agent")
plt.ylabel("Reward")
plt.savefig(os.path.join(FIG_DIR, "reward_boxplot.png"))
plt.close()

plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x="agent", y="reward")
plt.title("Reward Distribution (Violinplot)")
plt.xlabel("Agent")
plt.ylabel("Reward")
plt.savefig(os.path.join(FIG_DIR, "reward_violinplot.png"))
plt.close()

# 2. Action selection heatmap for each agent
for agent in df["agent"].unique():
    agent_df = df[df["agent"] == agent]
    heatmap = np.zeros((6, 6))
    for _, row in agent_df.iterrows():
        heatmap[int(row["action_x"]), int(row["action_y"])] += 1
    plt.figure(figsize=(6, 5))
    sns.heatmap(heatmap, annot=True, fmt=".0f", cmap="Blues")
    plt.title(f"Action Heatmap: {agent.capitalize()} Agent")
    plt.xlabel("Y Coordinate")
    plt.ylabel("X Coordinate")
    plt.savefig(os.path.join(FIG_DIR, f"action_heatmap_{agent}.png"))
    plt.close()

# 3. Hit rate (proportion of actions with reward > 0) for each agent
hit_rates = df.groupby("agent")["reward"].apply(lambda x: np.mean(np.array(x) > 0))
plt.figure(figsize=(6, 5))
sns.barplot(x=hit_rates.index, y=hit_rates.values)
plt.title("Hit Rate per Agent")
plt.xlabel("Agent")
plt.ylabel("Hit Rate")
plt.savefig(os.path.join(FIG_DIR, "hit_rate_barplot.png"))
plt.close()

# 4. Distribution of the number of turns per game
turns_per_game = df.groupby("game")["turn"].max()
plt.figure(figsize=(8, 5))
sns.histplot(turns_per_game, bins=20, kde=True)
plt.title("Distribution of Turns per Game")
plt.xlabel("Number of Turns")
plt.ylabel("Number of Games")
plt.savefig(os.path.join(FIG_DIR, "turns_per_game_hist.png"))
plt.close()

print("All analysis figures have been saved to the figures/ directory.")
