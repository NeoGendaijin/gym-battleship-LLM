import os
import gymnasium as gym
import importlib
import gym_battleship
importlib.reload(gym_battleship)
import torch
import openai
import numpy as np
from tqdm import tqdm
import asyncio
from openai import AsyncOpenAI
import json

# --- 設定 ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o"
STUDENT_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../distill/student_strategy.pth')
N_EPISODES = 10  # 少ないエピソード数で素早く終わらせる
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "rag_vs_student_results.json")
BOARD_SIZE = 6  # サイズを6x6に変更

# --- OpenAI RAGエージェント（async対応, 英語プロンプト） ---
class AsyncOpenAIAgent:
    def __init__(self, api_key=None, model=OPENAI_MODEL):
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key must be set in OPENAI_API_KEY env or passed to OpenAIAgent")
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model

    def obs_to_prompt(self, obs):
        # obs[0]: hit, obs[1]: miss
        view_board = np.zeros_like(obs[0])
        if obs.ndim == 3:
            view_board[obs[0] == 1] = 1    # hits
            view_board[obs[1] == 1] = -1   # misses
        else:
            view_board = obs
        board_str = "\n".join(" ".join(str(int(cell)) for cell in row) for row in view_board)
        prompt = (
            "You are a Battleship strategy agent.\n"
            "Here is your current observation grid (0: unexplored, 1: hit, -1: miss):\n"
            f"{board_str}\n"
            "What is the most reasonable (x, y) coordinate to attack next?\n"
            "Output only a single Python tuple, e.g., (3, 5). No explanation, no other text, no code block, no comments."
        )
        return prompt

    async def act(self, obs):
        prompt = self.obs_to_prompt(obs)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a Battleship strategy agent."},
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
            raise ValueError(f"OpenAI API did not return a valid (x, y) tuple: {text}")

# --- Studentエージェント（同期） ---
class StudentPolicy(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, strategy_dim):
        super().__init__()
        self.strategy_dim = strategy_dim
        self.features = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.action_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
        if strategy_dim > 0:
            self.strategy_head = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim // 2, strategy_dim)
            )

    def forward(self, x):
        features = self.features(x)
        action_logits = self.action_head(features)
        if self.strategy_dim > 0:
            strategy_logits = self.strategy_head(features)
            return action_logits, strategy_logits
        else:
            return action_logits

class StudentAgent:
    def __init__(self, model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        grid_size = checkpoint.get("grid_size", 10)
        hidden_dim = checkpoint.get("hidden_dim", 128)
        strategy_dim = checkpoint.get("strategy_dim", 0)
        input_dim = grid_size * grid_size * 2
        output_dim = grid_size * grid_size
        self.model = StudentPolicy(input_dim, hidden_dim, output_dim, strategy_dim)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def act(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(obs_tensor)
            if isinstance(logits, tuple):
                logits = logits[0]
            action = torch.argmax(logits, dim=1).item()
        size = int(np.sqrt(logits.shape[1]))
        return (action // size, action % size)

# --- 非同期対戦シミュレーション（可視化・保存付き） ---
async def run_battle_async(rag_agent, student_agent, n_episodes=100, results_path=RESULTS_PATH, board_size=BOARD_SIZE):
    rag_wins = 0
    student_wins = 0
    all_results = []
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
        while not (terminated or truncated):
            if turn % 2 == 0:
                agent_name = "rag"
                action = await rag_agent.act(obs)
            else:
                agent_name = "student"
                action = student_agent.act(obs)
            obs_next, reward, terminated, truncated, info = env.step(action)
            # 可視化: 盤面・アクション・reward
            print(f"[Episode {ep+1} | Turn {turn+1} | {agent_name}] Action: {action}, Reward: {reward}")
            print("Board (visualization):")
            # 可視化のため、1:hit, -1:miss のように表示
            view_board = np.zeros_like(obs[0])
            if obs.ndim == 3:
                view_board[obs[0] == 1] = 1    # hits
                view_board[obs[1] == 1] = -1   # misses
            else:
                view_board = obs
            for row in view_board:
                print(" ".join(f"{int(cell):2d}" for cell in row))
            print("-" * 30)
            episode_log["turns"].append({
                "turn": turn+1,
                "agent": agent_name,
                "action": action,
                "reward": reward,
                "board": view_board.tolist()
            })
            obs = obs_next
            winner = info.get('winner', None)
            turn += 1
        episode_log["winner"] = winner
        all_results.append(episode_log)
        if winner == 'rag':
            rag_wins += 1
        elif winner == 'student':
            student_wins += 1
    print(f'OpenAI Agent wins: {rag_wins}/{n_episodes}')
    print(f'Student Agent wins: {student_wins}/{n_episodes}')
    print(f'Draws: {n_episodes - rag_wins - student_wins}')
    # 保存
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {results_path}")

if __name__ == '__main__':
    rag_agent = AsyncOpenAIAgent()
    student_agent = StudentAgent(STUDENT_MODEL_PATH)
    asyncio.run(run_battle_async(rag_agent, student_agent, N_EPISODES, RESULTS_PATH))
