"""
train_dqn.py
============
Entrenamiento del agente DQN para NBA Lineup Optimizer.
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from environment.basketball_env import BasketballEnv
from agents.dqn_agent import DQNAgent


def train_dqn(
    episodes=3000,
    lr=1e-3,
    gamma=0.97,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,
    batch_size=64,
    target_update=200,
    seed=42,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DQN] Usando dispositivo: {device}")

    env = BasketballEnv(
        obs_mode="continuous",
        force_synthetic=False,
        seed=seed,
        render_mode=None,
    )

    agent = DQNAgent(
        state_dim=29,
        action_dim=env.n_actions(),
        lr=lr,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        target_update=target_update,
        device=device,
    )

    rewards_history = []
    win_history = []
    steps_history = []
    epsilon_history = []

    for episode in range(1, episodes + 1):
        state, info = env.reset(seed=seed + episode)

        done = False
        total_reward = 0.0
        steps = 0
        final_score_diff = 0

        while not done:
            available_actions = env.available_actions()

            action = agent.select_action(
                state=state,
                available_actions=available_actions,
            )

            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            agent.store(
                s=state,
                a=action,
                r=reward,
                s2=next_state,
                done=done,
            )

            agent.train_step()

            state = next_state
            total_reward += reward
            steps += 1
            final_score_diff = info["score_diff"]

        agent.decay_epsilon()

        win = 1 if final_score_diff > 0 else 0

        rewards_history.append(float(total_reward))
        win_history.append(int(win))
        steps_history.append(int(steps))
        epsilon_history.append(float(agent.epsilon))

        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            win_rate = np.mean(win_history[-100:])
            avg_steps = np.mean(steps_history[-100:])

            print(
                f"Episode {episode:04d} | "
                f"avg_reward={avg_reward:+.3f} | "
                f"win_rate={win_rate:.2%} | "
                f"avg_steps={avg_steps:.1f} | "
                f"epsilon={agent.epsilon:.3f} | "
                f"buffer={len(agent.memory)}"
            )

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    model_path = models_dir / "dqn_model.pth"
    metrics_path = results_dir / "dqn_metrics.json"

    agent.save(model_path)

    metrics = {
        "algorithm": "DQN",
        "episodes": episodes,
        "gamma": gamma,
        "learning_rate": lr,
        "epsilon_start": epsilon,
        "epsilon_min": epsilon_min,
        "epsilon_decay": epsilon_decay,
        "batch_size": batch_size,
        "target_update": target_update,
        "rewards": rewards_history,
        "wins": win_history,
        "steps": steps_history,
        "epsilon": epsilon_history,
        "final_win_rate_last_100": float(np.mean(win_history[-100:])),
        "final_reward_last_100": float(np.mean(rewards_history[-100:])),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Entrenamiento DQN terminado ===")
    print(f"Modelo guardado en: {model_path}")
    print(f"Métricas guardadas en: {metrics_path}")
    print(f"Win rate últimos 100 episodios: {np.mean(win_history[-100:]):.2%}")
    print(f"Reward promedio últimos 100 episodios: {np.mean(rewards_history[-100:]):+.3f}")

    env.close()

    return agent, rewards_history, win_history


if __name__ == "__main__":
    train_dqn(
        episodes=3000,
        lr=1e-3,
        gamma=0.97,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        target_update=200,
        seed=42,
    )