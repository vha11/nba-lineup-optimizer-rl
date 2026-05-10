"""
plot_results.py
===============
Genera gráficas para la memoria:
    - Reward Q-Learning
    - Reward DQN
    - Comparación win rate
    - Comparación reward
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

FIGURES_DIR.mkdir(exist_ok=True)


def load_metrics(filename):
    path = RESULTS_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def moving_average(values, window=100):
    values = np.array(values, dtype=float)

    if len(values) < window:
        return values

    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_reward(metrics, output_name, title):
    rewards = metrics["rewards"]
    avg_rewards = moving_average(rewards, window=100)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label="Reward por episodio")
    plt.plot(
        range(99, 99 + len(avg_rewards)),
        avg_rewards,
        linewidth=2,
        label="Media móvil (100 episodios)",
    )
    plt.xlabel("Episodio")
    plt.ylabel("Reward acumulado")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_name, dpi=300)
    plt.close()


def plot_winrate_comparison(q_metrics, dqn_metrics):
    q_wins = moving_average(q_metrics["wins"], window=100)
    dqn_wins = moving_average(dqn_metrics["wins"], window=100)

    plt.figure(figsize=(10, 5))
    plt.plot(
        range(99, 99 + len(q_wins)),
        q_wins,
        label="Q-Learning",
        linewidth=2,
    )
    plt.plot(
        range(99, 99 + len(dqn_wins)),
        dqn_wins,
        label="DQN",
        linewidth=2,
    )

    plt.xlabel("Episodio")
    plt.ylabel("Win rate")
    plt.title("Comparación de win rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "winrate_comparison.png", dpi=300)
    plt.close()


def plot_reward_comparison(q_metrics, dqn_metrics):
    q_rewards = moving_average(q_metrics["rewards"], window=100)
    dqn_rewards = moving_average(dqn_metrics["rewards"], window=100)

    plt.figure(figsize=(10, 5))
    plt.plot(
        range(99, 99 + len(q_rewards)),
        q_rewards,
        label="Q-Learning",
        linewidth=2,
    )
    plt.plot(
        range(99, 99 + len(dqn_rewards)),
        dqn_rewards,
        label="DQN",
        linewidth=2,
    )

    plt.xlabel("Episodio")
    plt.ylabel("Reward medio móvil")
    plt.title("Comparación de reward promedio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "reward_comparison.png", dpi=300)
    plt.close()


def plot_epsilon(metrics, output_name, title):
    if "epsilon" not in metrics:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(metrics["epsilon"], linewidth=2)
    plt.xlabel("Episodio")
    plt.ylabel(r"$\epsilon$")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_name, dpi=300)
    plt.close()


def main():
    q_metrics = load_metrics("q_learning_metrics.json")
    dqn_metrics = load_metrics("dqn_metrics.json")

    plot_reward(
        q_metrics,
        "reward_q_learning.png",
        "Evolución de reward - Q-Learning",
    )

    plot_reward(
        dqn_metrics,
        "reward_dqn.png",
        "Evolución de reward - DQN",
    )

    plot_winrate_comparison(q_metrics, dqn_metrics)
    plot_reward_comparison(q_metrics, dqn_metrics)

    plot_epsilon(
        dqn_metrics,
        "epsilon_dqn.png",
        "Decaimiento de epsilon - DQN",
    )

    print("Gráficas generadas en la carpeta figures/")


if __name__ == "__main__":
    main()