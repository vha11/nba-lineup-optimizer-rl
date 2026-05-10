"""
train_q_learning.py
===================
Entrenamiento del agente Q-Learning tabular para NBA Lineup Optimizer.
Guarda:
    - modelo en models/q_learning_agent.pkl
    - métricas en results/q_learning_metrics.json
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from environment.basketball_env import BasketballEnv
from agents.q_learning_agent import QLearningAgent


def train_q_learning(
    episodes: int = 3000,
    alpha: float = 0.1,
    gamma: float = 0.97,
    epsilon: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    seed: int = 42,
):
    env = BasketballEnv(
        obs_mode="discrete",
        force_synthetic=False,
        seed=seed,
        render_mode=None,
    )

    agent = QLearningAgent(
        n_actions=env.n_actions(),
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )

    rewards_history = []
    win_history = []
    steps_history = []
    epsilon_history = []
    q_states_history = []

    for episode in range(1, episodes + 1):
        env.reset(seed=seed + episode)

        state = env.get_obs_for_qtable()
        done = False

        total_reward = 0.0
        steps = 0
        final_score_diff = 0

        while not done:
            available_actions = env.available_actions()
            action = agent.select_action(state, available_actions)

            _, reward, terminated, truncated, info = env.step(action)

            next_state = env.get_obs_for_qtable()
            next_available_actions = env.available_actions()

            done = terminated or truncated

            agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                next_available_actions=next_available_actions,
            )

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
        q_states_history.append(int(len(agent.q_table)))

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
                f"Q_states={len(agent.q_table)}"
            )

    models_dir = ROOT / "models"
    results_dir = ROOT / "results"

    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    model_path = models_dir / "q_learning_agent.pkl"
    metrics_path = results_dir / "q_learning_metrics.json"

    agent.save(model_path)

    metrics = {
        "algorithm": "Q-Learning",
        "episodes": episodes,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_start": epsilon,
        "epsilon_min": epsilon_min,
        "epsilon_decay": epsilon_decay,
        "rewards": rewards_history,
        "wins": win_history,
        "steps": steps_history,
        "epsilon": epsilon_history,
        "q_states_history": q_states_history,
        "q_states": len(agent.q_table),
        "final_win_rate_last_100": float(np.mean(win_history[-100:])),
        "final_reward_last_100": float(np.mean(rewards_history[-100:])),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Entrenamiento Q-Learning terminado ===")
    print(f"Modelo guardado en: {model_path}")
    print(f"Métricas guardadas en: {metrics_path}")
    print(f"Estados aprendidos: {len(agent.q_table)}")
    print(f"Win rate últimos 100 episodios: {np.mean(win_history[-100:]):.2%}")
    print(f"Reward promedio últimos 100 episodios: {np.mean(rewards_history[-100:]):+.3f}")

    env.close()

    return agent, rewards_history, win_history


if __name__ == "__main__":
    train_q_learning(
        episodes=3000,
        alpha=0.1,
        gamma=0.97,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        seed=42,
    )