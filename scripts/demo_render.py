"""
demo_render.py
==============
Demo visual en consola del entorno NBA Lineup Optimizer.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from environment.basketball_env import BasketballEnv


def main():
    env = BasketballEnv(
        obs_mode="continuous",
        force_synthetic=False,
        seed=42,
        render_mode=None,
    )

    obs, info = env.reset(seed=42)

    print("\nESTADO INICIAL")
    print(env.render())

    done = False
    step = 0
    total_reward = 0.0

    while not done:
        input("\nPresiona ENTER para ejecutar la siguiente decisión del agente...")

        available_actions = env.available_actions()
        action = int(np.random.choice(available_actions))

        obs, reward, terminated, truncated, info = env.step(action)

        step += 1
        total_reward += reward
        done = terminated or truncated

        print("\n" + "=" * 90)
        print(f"DECISIÓN {step}")
        print(f"Acción seleccionada: {env.action_to_lineup(action)}")
        print(f"Reward del stint: {reward:+.3f}")
        print(f"Trigger: {info['trigger']}")
        print("=" * 90)
        print(env.render())

        time.sleep(0.2)

    print("\nSIMULACIÓN TERMINADA")
    print(f"Reward total: {total_reward:+.3f}")
    print(f"Resultado: {'VICTORIA' if info['score_diff'] > 0 else 'DERROTA'}")


if __name__ == "__main__":
    main()