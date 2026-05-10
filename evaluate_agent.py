import sys
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from environment.basketball_env import BasketballEnv
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent


def evaluate_q_learning(episodes=200, seed=123):
    env = BasketballEnv(obs_mode="discrete", seed=seed)

    agent = QLearningAgent(
        n_actions=env.n_actions(),
        alpha=0.1,
        gamma=0.97,
        epsilon=0.0,
    )

    agent.load(ROOT / "models" / "q_learning_agent.pkl")

    rewards, wins = [], []

    for ep in range(episodes):
        env.reset(seed=seed + ep)
        done = False
        total_reward = 0
        final_diff = 0

        while not done:
            state = env.get_obs_for_qtable()
            action = agent.select_action(state, env.available_actions())

            _, reward, term, trunc, info = env.step(action)
            done = term or trunc

            total_reward += reward
            final_diff = info["score_diff"]

        rewards.append(total_reward)
        wins.append(1 if final_diff > 0 else 0)

    env.close()

    return np.mean(rewards), np.mean(wins)


def evaluate_dqn(episodes=200, seed=123):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = BasketballEnv(obs_mode="continuous", seed=seed)

    agent = DQNAgent(
        state_dim=29,
        action_dim=env.n_actions(),
        epsilon=0.0,
        epsilon_min=0.0,
        epsilon_decay=1.0,
        device=device,
    )

    agent.load(ROOT / "models" / "dqn_model.pth")

    rewards, wins = [], []

    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        done = False
        total_reward = 0
        final_diff = 0

        while not done:
            action = agent.select_action(state, env.available_actions())

            state, reward, term, trunc, info = env.step(action)
            done = term or trunc

            total_reward += reward
            final_diff = info["score_diff"]

        rewards.append(total_reward)
        wins.append(1 if final_diff > 0 else 0)

    env.close()

    return np.mean(rewards), np.mean(wins)


if __name__ == "__main__":
    q_reward, q_win = evaluate_q_learning()
    dqn_reward, dqn_win = evaluate_dqn()

    print("\n=== RESULTADOS ===")
    print(f"Q-Learning → Reward: {q_reward:.2f} | Win rate: {q_win:.2%}")
    print(f"DQN        → Reward: {dqn_reward:.2f} | Win rate: {dqn_win:.2%}")