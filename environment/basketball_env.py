"""
basketball_env.py
=================
Entorno Gymnasium para NBA Lineup Optimizer.

Envuelve GameSimulator y expone:
    reset()
    step(action)
    render()
    close()

Modos:
    continuous → observación Box(29,) para DQN
    discrete   → usa get_obs_for_qtable() para Q-Learning tabular
"""

import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data_loader import load_or_fetch_data, compute_lineup_stats
from environment.game_simulator import GameSimulator

OBS_DIM = 29
MAX_STEPS = 60
DEFAULT_OBS_MODE = "continuous"


class BasketballEnv(gym.Env):
    """
    Entorno Gymnasium para optimización de rotaciones NBA.

    S: vector continuo de 29 variables.
    A: lineups reales filtrados + acción MAINTAIN.
    R: Net Rating + penalización fatiga/faltas + bonus momentum.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        obs_mode: str = DEFAULT_OBS_MODE,
        force_synthetic: bool = False,
        seed: int = None,
        render_mode: str = None,
    ):
        super().__init__()

        assert obs_mode in ("continuous", "discrete")

        self.obs_mode = obs_mode
        self.render_mode = render_mode
        self._seed = seed
        self._step_count = 0

        lineups_raw, players_raw = load_or_fetch_data(
            force_synthetic=force_synthetic
        )

        self.lineup_stats = compute_lineup_stats(lineups_raw)
        self.player_config = players_raw

        self._lineup_ids = sorted(self.lineup_stats.keys()) + ["MAINTAIN"]
        self._n_actions = len(self._lineup_ids)

        self.simulator = GameSimulator(
            lineup_stats=self.lineup_stats,
            player_config=self.player_config,
            seed=seed,
        )

        self.action_space = spaces.Discrete(self._n_actions)

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # API Gymnasium
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:

        super().reset(seed=seed)

        if seed is not None:
            self._seed = seed
            self.simulator = GameSimulator(
                lineup_stats=self.lineup_stats,
                player_config=self.player_config,
                seed=seed,
            )

        self.simulator.reset()
        self._step_count = 0

        obs = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            print("\n" + "=" * 60)
            print("NUEVO PARTIDO — Boston Celtics")
            print("=" * 60)
            print(self.simulator.render())

        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), (
            f"Acción inválida: {action}. "
            f"Debe estar entre 0 y {self._n_actions - 1}."
        )

        lineup_id = self._resolve_action(action)

        _, reward, game_over, sim_info = self.simulator.step(lineup_id)

        self._step_count += 1

        terminated = game_over
        truncated = self._step_count >= MAX_STEPS and not game_over

        obs = self._get_observation()
        info = {**self._get_info(), **sim_info}

        if self.render_mode == "human":
            print(self.simulator.render())

        return obs, float(reward), terminated, truncated, info

    def render(self) -> Optional[str]:
        if self.render_mode == "human":
            print(self.simulator.render())
            return None

        return self.simulator.render()

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Acciones
    # ------------------------------------------------------------------

    def action_to_lineup(self, action: int) -> str:
        return self._lineup_ids[action]

    def lineup_to_action(self, lineup_id: str) -> int:
        return self._lineup_ids.index(lineup_id)

    def available_actions(self) -> list[int]:
        available_ids = self.simulator.available_lineups()

        return [
            self._lineup_ids.index(lineup_id)
            for lineup_id in available_ids
            if lineup_id in self._lineup_ids
        ]

    def n_actions(self) -> int:
        return self._n_actions

    # ------------------------------------------------------------------
    # Observaciones
    # ------------------------------------------------------------------

    def get_obs_for_qtable(self) -> tuple:
        return self.simulator.state.discretized_observation()

    def _get_observation(self) -> np.ndarray:
        obs = self.simulator.state.observation_vector()
        return obs.astype(np.float32)

    def _get_info(self) -> dict:
        s = self.simulator.state

        return {
            "score_diff": s.score_diff,
            "quarter": s.quarter,
            "time_remaining": s.time_remaining_quarter,
            "momentum": s.momentum(),
            "step_count": self._step_count,
            "lineup": s.current_lineup_id,
            "available_actions": self.available_actions(),
        }

    # ------------------------------------------------------------------
    # Resolución de acciones
    # ------------------------------------------------------------------

    def _resolve_action(self, action: int) -> str:
        requested_lineup = self._lineup_ids[action]
        available = self.simulator.available_lineups()

        if requested_lineup in available:
            return requested_lineup

        valid_lineups = [
            lineup_id
            for lineup_id in available
            if lineup_id != "MAINTAIN"
        ]

        if valid_lineups:
            # Escoger el lineup disponible con mayor muestra de minutos.
            return max(
                valid_lineups,
                key=lambda lid: self.lineup_stats[lid].get("minutes", 0),
            )

        return "MAINTAIN"


# ---------------------------------------------------------------------------
# Registro opcional
# ---------------------------------------------------------------------------

def register_env() -> None:
    if "BasketballLineup-v0" not in gym.envs.registry:
        gym.register(
            id="BasketballLineup-v0",
            entry_point="environment.basketball_env:BasketballEnv",
            kwargs={
                "obs_mode": "continuous",
                "force_synthetic": False,
            },
        )

        print("[basketball_env] Entorno BasketballLineup-v0 registrado.")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Smoke test: BasketballEnv ===\n")

    env = BasketballEnv(
        obs_mode="continuous",
        force_synthetic=False,
        seed=42,
        render_mode=None,
    )

    obs, info = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Observation min/max: {obs.min():.3f} / {obs.max():.3f}")
    print(f"Action space: Discrete({env.n_actions()})")
    print(f"Available actions: {env.available_actions()}")
    print(f"Initial lineup: {info['lineup']}")

    total_reward = 0.0
    done = False
    steps = 0

    while not done:
        available = env.available_actions()
        action = int(np.random.choice(available))

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1
        done = terminated or truncated

        print(
            f"Step {steps:02d} | "
            f"lineup={info['lineup']} | "
            f"reward={reward:+.3f} | "
            f"diff={info['score_diff']:+d} | "
            f"Q{info['quarter']} | "
            f"trigger={info['trigger']}"
        )

    print("\n=== Episodio terminado ===")
    print(f"Steps: {steps}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Score diff final: {info['score_diff']:+d}")
    print("Resultado:", "VICTORIA" if info["score_diff"] > 0 else "DERROTA")

    try:
        from gymnasium.utils.env_checker import check_env

        check_env(env)
        print("\ncheck_env: entorno compatible con Gymnasium.")
    except Exception as e:
        print(f"\ncheck_env warning/error: {e}")

    env.close()