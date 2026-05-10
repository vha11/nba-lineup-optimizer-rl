"""
q_learning_agent.py
===================
Agente Q-Learning tabular para NBA Lineup Optimizer.

Compatible con el MDP definido:
    S: estado discretizado del partido
    A: lineups reales + MAINTAIN
    R: recompensa compuesta del entorno
    gamma: 0.97
"""

import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.97,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.rng = random.Random(seed)

        self.q_table = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float32))

    def select_action(self, state: tuple, available_actions: list[int]) -> int:
        """
        Política epsilon-greedy restringida a acciones válidas.
        """
        if not available_actions:
            raise ValueError("No hay acciones disponibles.")

        if self.rng.random() < self.epsilon:
            return self.rng.choice(available_actions)

        q_values = self.q_table[state]
        valid_q_values = [(a, q_values[a]) for a in available_actions]

        max_q = max(q for _, q in valid_q_values)
        best_actions = [a for a, q in valid_q_values if q == max_q]

        return self.rng.choice(best_actions)

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        done: bool,
        next_available_actions: list[int],
    ) -> None:
        """
        Actualización Q-Learning:

            Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
        """

        current_q = self.q_table[state][action]

        if done or not next_available_actions:
            target = reward
        else:
            next_q_values = self.q_table[next_state]
            max_next_q = max(next_q_values[a] for a in next_available_actions)
            target = reward + self.gamma * max_next_q

        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, path: str | Path) -> None:
        path = Path(path)

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.q_table = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float32),
            data,
        )

    def greedy_policy(self) -> dict:
        """
        Devuelve la política aprendida:
            estado -> mejor acción
        """
        policy = {}

        for state, q_values in self.q_table.items():
            policy[state] = int(np.argmax(q_values))

        return policy