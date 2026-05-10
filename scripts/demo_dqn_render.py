import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from environment.basketball_env import BasketballEnv
from agents.dqn_agent import DQNAgent


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = BasketballEnv(
        obs_mode="continuous",
        force_synthetic=False,
        seed=42,
        render_mode=None,
    )

    agent = DQNAgent(
        state_dim=29,
        action_dim=env.n_actions(),
        epsilon=0.0,
        epsilon_min=0.0,
        epsilon_decay=1.0,
        device=device,
    )

    agent.load(ROOT / "models" / "dqn_model.pth")
    agent.epsilon = 0.0
    agent.q_net.eval()

    state, info = env.reset(seed=42)

    print("\nESTADO INICIAL — AGENTE DQN ENTRENADO")
    print(env.render())

    done = False
    step = 0
    total_reward = 0.0

    while not done:
        input("\nPresiona ENTER para ejecutar la siguiente decisión del agente DQN...")

        available_actions = env.available_actions()

        action = agent.select_action(
            state=state,
            available_actions=available_actions,
        )

        next_state, reward, terminated, truncated, info = env.step(action)

        step += 1
        total_reward += reward
        done = terminated or truncated
        state = next_state

        print("\n" + "=" * 90)
        print(f"DECISIÓN {step}")
        print(f"Acción seleccionada: {env.action_to_lineup(action)}")
        print(f"Reward del stint: {reward:+.3f}")
        print(f"Trigger: {info['trigger']}")
        print("=" * 90)
        print(env.render())

        time.sleep(0.2)

    print("\nSIMULACIÓN TERMINADA — DQN")
    print(f"Reward total: {total_reward:+.3f}")
    print(f"Score diff final: {info['score_diff']:+d}")
    print(f"Resultado: {'VICTORIA' if info['score_diff'] > 0 else 'DERROTA'}")

    env.close()


if __name__ == "__main__":
    main()