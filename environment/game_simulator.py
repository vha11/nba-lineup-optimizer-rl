"""
game_simulator.py
=================
Motor de simulación de partido para NBA Lineup Optimizer.

Responsabilidades:
    - Simular un partido NBA por stints.
    - Aplicar lineups seleccionados por el agente.
    - Generar diferencial de puntos con una distribución normal calibrada.
    - Actualizar marcador, faltas, fatiga, minutos y momentum.
    - Detectar triggers de decisión.
    - Calcular recompensas del MDP.

Este módulo NO conoce Gymnasium ni al agente.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum


# ---------------------------------------------------------------------------
# Constantes del juego
# ---------------------------------------------------------------------------

QUARTERS = 4
QUARTER_MINUTES = 12
OVERTIME_MINUTES = 5

MAX_FOULS_FOULOUT = 6
TEAM_FOULS_BONUS = 5
POSSESSIONS_PER_MIN = 2.1

# Triggers
TRIGGER_FOULS_ALERT = 3
TRIGGER_FOULS_CRITICAL = 5
TRIGGER_MOMENTUM_NEG = -6
TRIGGER_MOMENTUM_WINDOW = 3


# ---------------------------------------------------------------------------
# Tipos de trigger
# ---------------------------------------------------------------------------

class TriggerType(IntEnum):
    QUARTER_START = 0
    TIMEOUT = 1
    FOUL_ALERT = 2
    FOUL_CRITICAL = 3
    FATIGUE = 4
    MOMENTUM_NEG = 5
    FORCED = 6
    GAME_END = 7


# ---------------------------------------------------------------------------
# Estado de jugadores
# ---------------------------------------------------------------------------

@dataclass
class PlayerState:
    name: str
    fouls: int = 0
    total_minutes: float = 0.0
    stint_minutes: float = 0.0
    is_on_court: bool = False
    is_fouled_out: bool = False

    fatigue_threshold_stint: int = 7
    fatigue_threshold_total: int = 30
    foul_out_risk_threshold: int = 4


# ---------------------------------------------------------------------------
# Estado global del partido
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    score_team: int = 0
    score_rival: int = 0

    quarter: int = 1
    minute_in_quarter: float = 0.0
    is_overtime: bool = False
    game_over: bool = False

    team_fouls_quarter: int = 0
    rival_fouls_quarter: int = 0
    foul_bonus_rival: bool = False

    timeouts_remaining: int = 7

    score_history: list = field(default_factory=list)

    current_lineup_id: Optional[str] = None
    current_lineup_players: list = field(default_factory=list)

    players: dict = field(default_factory=dict)

    @property
    def score_diff(self) -> int:
        return self.score_team - self.score_rival

    @property
    def time_remaining_quarter(self) -> float:
        total = OVERTIME_MINUTES if self.is_overtime else QUARTER_MINUTES
        return max(0.0, total - self.minute_in_quarter)

    @property
    def total_minutes_played(self) -> float:
        if self.quarter <= 4:
            return (self.quarter - 1) * QUARTER_MINUTES + self.minute_in_quarter

        # Prórroga
        return 4 * QUARTER_MINUTES + (self.quarter - 5) * OVERTIME_MINUTES + self.minute_in_quarter

    def momentum(self, window_minutes: float = TRIGGER_MOMENTUM_WINDOW) -> int:
        cutoff = self.total_minutes_played - window_minutes
        recent = [delta for t, delta in self.score_history if t >= cutoff]
        raw = int(sum(recent))
        return max(-10, min(10, raw))

    def observation_vector(self) -> np.ndarray:
        """
        Vector continuo de 29 variables:
            5 globales + 8 jugadores * 3 variables.
        """
        global_feats = np.array(
            [
                np.clip(self.score_diff, -30, 30) / 30.0,
                min(self.quarter, 4) / 4.0,
                self.time_remaining_quarter / 12.0,
                self.momentum() / 10.0,
                float(self.foul_bonus_rival),
            ],
            dtype=np.float32,
        )

        player_feats = []

        for player_name in sorted(self.players.keys()):
            p = self.players[player_name]
            player_feats.extend(
                [
                    min(p.stint_minutes, 12.0) / 12.0,
                    min(p.total_minutes, 48.0) / 48.0,
                    min(p.fouls, 5) / 5.0,
                ]
            )

        obs = np.concatenate(
            [global_feats, np.array(player_feats, dtype=np.float32)]
        )

        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def discretized_observation(self) -> tuple:
        """
        Observación discretizada para Q-Learning tabular.
        """

        def bucket3(value: float, low: float, high: float) -> int:
            if value <= low:
                return 0
            elif value <= high:
                return 1
            return 2

        score_bucket = bucket3(self.score_diff, -8, 8)
        time_bucket = bucket3(self.time_remaining_quarter, 4, 9)
        momentum_bucket = bucket3(self.momentum(), -3, 3)

        player_buckets = []

        for player_name in sorted(self.players.keys()):
            p = self.players[player_name]
            player_buckets.append(bucket3(p.stint_minutes, 4, 7))
            player_buckets.append(min(p.fouls, 5))

        return (
            score_bucket,
            min(self.quarter, 4),
            time_bucket,
            momentum_bucket,
            int(self.foul_bonus_rival),
            *player_buckets,
        )


# ---------------------------------------------------------------------------
# Simulador principal
# ---------------------------------------------------------------------------

class GameSimulator:
    """
    Motor del partido.

    step(lineup_id) avanza el partido hasta el siguiente trigger.
    """

    def __init__(self, lineup_stats: dict, player_config: dict, seed: int = None):
        self.lineup_stats = lineup_stats
        self.player_config = player_config
        self.rng = np.random.default_rng(seed)
        self.state: Optional[GameState] = None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> GameState:
        players = {}

        for name, config in self.player_config.items():
            players[name] = PlayerState(
                name=name,
                fatigue_threshold_stint=config.get("fatigue_threshold_stint", 7),
                fatigue_threshold_total=config.get("fatigue_threshold_total", 30),
                foul_out_risk_threshold=config.get("foul_out_risk_threshold", 4),
            )

        self.state = GameState(players=players)

        # Lineup inicial: el de mayor muestra de minutos
        initial_lineup_id = self._initial_lineup()
        self._apply_lineup(initial_lineup_id)

        return self.state

    def _initial_lineup(self) -> str:
        return max(
            self.lineup_stats.keys(),
            key=lambda lid: self.lineup_stats[lid].get("minutes", 0),
        )

    # ------------------------------------------------------------------
    # Step principal
    # ------------------------------------------------------------------

    def step(self, lineup_id: str) -> tuple[GameState, float, bool, dict]:
        assert self.state is not None, "Debes llamar reset() antes de step()."

        if self.state.game_over:
            return self.state, 0.0, True, {"trigger": TriggerType.GAME_END}

        if lineup_id == "MAINTAIN":
            if self.state.current_lineup_id is None:
                lineup_id = self._initial_lineup()
                self._apply_lineup(lineup_id)
        else:
            self._apply_lineup(lineup_id)

        stint_result = self._simulate_stint()
        reward = self._compute_reward(stint_result, lineup_id)

        if self.state.game_over:
            reward += self._terminal_reward()

        info = {
            "trigger": stint_result["trigger"],
            "stint_minutes": stint_result["stint_minutes"],
            "points_scored": stint_result["points_scored"],
            "points_allowed": stint_result["points_allowed"],
            "score_diff": self.state.score_diff,
            "quarter": self.state.quarter,
            "lineup": self.state.current_lineup_id,
        }

        return self.state, float(reward), self.state.game_over, info

    # ------------------------------------------------------------------
    # Aplicar lineup
    # ------------------------------------------------------------------

    def _apply_lineup(self, lineup_id: str) -> None:
        if lineup_id not in self.lineup_stats:
            raise ValueError(f"Lineup no encontrado: {lineup_id}")

        new_players = set(self.lineup_stats[lineup_id]["players"])

        # Validación defensiva: todos deben existir en player_config
        unknown_players = new_players - set(self.state.players.keys())
        if unknown_players:
            raise ValueError(
                f"El lineup {lineup_id} contiene jugadores no registrados: "
                f"{unknown_players}"
            )

        # No permitir jugadores expulsados
        fouled_out = {
            name for name, p in self.state.players.items() if p.is_fouled_out
        }

        if any(player in fouled_out for player in new_players):
            raise ValueError(
                f"El lineup {lineup_id} contiene jugadores expulsados: "
                f"{new_players & fouled_out}"
            )

        for name, player in self.state.players.items():
            was_on = player.is_on_court
            now_on = name in new_players

            player.is_on_court = now_on

            if was_on and not now_on:
                player.stint_minutes = 0.0

        self.state.current_lineup_id = lineup_id
        self.state.current_lineup_players = list(new_players)

    # ------------------------------------------------------------------
    # Simulación del stint
    # ------------------------------------------------------------------

    def _simulate_stint(self) -> dict:
        lineup_id = self.state.current_lineup_id
        stats = self.lineup_stats.get(lineup_id, {"mu": 0.0, "sigma": 6.0})

        mu_per_min = stats["mu"] / 48.0
        sigma_per_min = stats["sigma"] / np.sqrt(48.0)

        points_scored = 0
        points_allowed = 0
        stint_minutes = 0.0
        trigger = None

        quarter_time_left = self.state.time_remaining_quarter

        while trigger is None and quarter_time_left > 0 and not self.state.game_over:
            delta_raw = self.rng.normal(mu_per_min, sigma_per_min)

            base_points = 2.1

            team_pts = max(0, int(round(base_points + delta_raw / 2)))
            rival_pts = max(0, int(round(base_points - delta_raw / 2)))

            self.state.score_team += team_pts
            self.state.score_rival += rival_pts

            points_scored += team_pts
            points_allowed += rival_pts

            self.state.score_history.append(
                (self.state.total_minutes_played, team_pts - rival_pts)
            )

            for player in self.state.players.values():
                if player.is_on_court:
                    player.stint_minutes += 1.0
                    player.total_minutes += 1.0

            trigger = self._simulate_fouls()

            self.state.minute_in_quarter += 1.0
            stint_minutes += 1.0
            quarter_time_left -= 1.0

            if trigger is None:
                trigger = self._check_triggers()

        if trigger is None:
            trigger = self._advance_quarter()

        return {
            "trigger": trigger,
            "stint_minutes": stint_minutes,
            "points_scored": points_scored,
            "points_allowed": points_allowed,
        }

    # ------------------------------------------------------------------
    # Faltas
    # ------------------------------------------------------------------

    def _simulate_fouls(self) -> Optional[TriggerType]:
        """
        Simula faltas propias y faltas del rival.

        Las faltas propias afectan riesgo de foul-out.
        Las faltas del rival activan foul_bonus_rival.
        """

        own_foul_rate_per_player_min = 0.4 / 48.0
        rival_team_foul_rate_per_min = 4.5 / 12.0

        trigger = None

        # Faltas propias por jugador en cancha
        for player in self.state.players.values():
            if not player.is_on_court or player.is_fouled_out:
                continue

            if self.rng.random() < own_foul_rate_per_player_min:
                player.fouls += 1
                self.state.team_fouls_quarter += 1

                if player.fouls >= MAX_FOULS_FOULOUT:
                    player.is_fouled_out = True
                    player.is_on_court = False
                    trigger = TriggerType.FORCED
                elif player.fouls == TRIGGER_FOULS_CRITICAL:
                    trigger = trigger or TriggerType.FOUL_CRITICAL
                elif player.fouls == TRIGGER_FOULS_ALERT:
                    trigger = trigger or TriggerType.FOUL_ALERT

        # Faltas del rival para activar bonus
        if self.rng.random() < rival_team_foul_rate_per_min:
            self.state.rival_fouls_quarter += 1

        self.state.foul_bonus_rival = (
            self.state.rival_fouls_quarter >= TEAM_FOULS_BONUS
        )

        return trigger

    # ------------------------------------------------------------------
    # Triggers
    # ------------------------------------------------------------------

    def _check_triggers(self) -> Optional[TriggerType]:
        for player in self.state.players.values():
            if (
                player.is_on_court
                and player.stint_minutes >= player.fatigue_threshold_stint
            ):
                return TriggerType.FATIGUE

        if self.state.momentum() <= TRIGGER_MOMENTUM_NEG:
            return TriggerType.MOMENTUM_NEG

        return None

    # ------------------------------------------------------------------
    # Cambio de cuarto / fin de partido
    # ------------------------------------------------------------------

    def _advance_quarter(self) -> TriggerType:
        self.state.minute_in_quarter = 0.0
        self.state.team_fouls_quarter = 0
        self.state.rival_fouls_quarter = 0
        self.state.foul_bonus_rival = False

        for player in self.state.players.values():
            player.stint_minutes = 0.0

        if self.state.quarter < QUARTERS:
            self.state.quarter += 1
            return TriggerType.QUARTER_START

        if self.state.score_diff == 0:
            self.state.is_overtime = True
            self.state.quarter += 1
            return TriggerType.QUARTER_START

        self.state.game_over = True
        return TriggerType.GAME_END

    # ------------------------------------------------------------------
    # Recompensa
    # ------------------------------------------------------------------

    def _compute_reward(self, stint_result: dict, action: str) -> float:
        on_court = [
            p for p in self.state.players.values()
            if p.is_on_court
        ]

        possessions = max(1.0, stint_result["stint_minutes"] * POSSESSIONS_PER_MIN)
        net_points = stint_result["points_scored"] - stint_result["points_allowed"]

        r_net = (net_points / possessions) * 10.0

        r_fatigue = -1.5 * sum(
            1 for p in on_court
            if p.stint_minutes > p.fatigue_threshold_stint
        )

        r_fouls = 0.0

        for p in on_court:
            if p.fouls >= 5:
                r_fouls -= 2.0
            elif p.fouls >= 4 and self.state.quarter <= 3:
                r_fouls -= 0.5

        r_momentum = 0.0

        if action == "MAINTAIN" and self.state.momentum() > 3:
            r_momentum = 0.5

        reward = r_net + r_fatigue + r_fouls + r_momentum

        return float(np.clip(reward, -10.0, 10.0))

    def _terminal_reward(self) -> float:
        return 1.0 if self.state.score_diff > 0 else -1.0

    # ------------------------------------------------------------------
    # Acciones disponibles
    # ------------------------------------------------------------------

    def available_lineups(self) -> list[str]:
        assert self.state is not None, "Debes llamar reset() antes."

        fouled_out = {
            name for name, p in self.state.players.items()
            if p.is_fouled_out
        }

        available = []

        for lineup_id, stats in self.lineup_stats.items():
            lineup_players = set(stats["players"])

            if lineup_players & fouled_out:
                continue

            if not lineup_players.issubset(set(self.state.players.keys())):
                continue

            available.append(lineup_id)

        if self.state.current_lineup_id is not None:
            available.append("MAINTAIN")

        return available

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    
    def render(self) -> str:
        """
        Render textual tipo dashboard para visualizar el estado del partido.
        """

        s = self.state

        if s is None:
            return "GameSimulator no inicializado."

        score_line = (
            f"CELTICS {s.score_team:03d} - {s.score_rival:03d} RIVAL"
        )

        quarter_label = f"OT{s.quarter - 4}" if s.quarter > 4 else f"Q{s.quarter}"

        header = [
            "╔" + "═" * 78 + "╗",
            "║" + " NBA LINEUP OPTIMIZER — SIMULACIÓN DEL PARTIDO ".center(78) + "║",
            "╠" + "═" * 78 + "╣",
            f"║ {quarter_label:<5} | Tiempo restante: {s.time_remaining_quarter:>4.1f} min "
            f"| {score_line:<29} | Diff: {s.score_diff:+4d} ║",
            f"║ Momentum: {s.momentum():+3d} | "
            f"Faltas equipo: {s.team_fouls_quarter:<2d} | "
            f"Faltas rival: {s.rival_fouls_quarter:<2d} | "
            f"Bonus rival: {str(s.foul_bonus_rival):<5} ".ljust(79) + "║",
            "╠" + "═" * 78 + "╣",
            "║ LINEUP ACTUAL".ljust(79) + "║",
            "╠" + "═" * 78 + "╣",
        ]

        lineup_players = s.current_lineup_players

        if lineup_players:
            for player in lineup_players:
                p = s.players[player]
                fatigue_level = self._fatigue_label(p)
                foul_level = self._foul_label(p)

                header.append(
                    f"║ ON  {player:<22} | "
                    f"Stint: {p.stint_minutes:>4.1f} | "
                    f"Total: {p.total_minutes:>4.1f} | "
                    f"Faltas: {p.fouls:<1d} | "
                    f"{fatigue_level:<9} | {foul_level:<8} ║"
                )
        else:
            header.append("║ Sin lineup activo".ljust(79) + "║")

        header.extend(
            [
                "╠" + "═" * 78 + "╣",
                "║ BANCA / ESTADO GENERAL".ljust(79) + "║",
                "╠" + "═" * 78 + "╣",
            ]
        )

        for name, p in s.players.items():
            if p.is_on_court:
                continue

            status = "OUT" if p.is_fouled_out else "BENCH"
            fatigue_level = self._fatigue_label(p)
            foul_level = self._foul_label(p)

            header.append(
                f"║ {status:<5} {name:<22} | "
                f"Total: {p.total_minutes:>4.1f} | "
                f"Faltas: {p.fouls:<1d} | "
                f"{fatigue_level:<9} | {foul_level:<8}".ljust(79) + "║"
            )

        header.append("╚" + "═" * 78 + "╝")

        return "\n".join(header)


    def _fatigue_label(self, player) -> str:
        """
        Etiqueta cualitativa de fatiga según minutos del stint.
        """
        if player.stint_minutes >= player.fatigue_threshold_stint:
            return "FATIGA ALTA"
        elif player.stint_minutes >= player.fatigue_threshold_stint * 0.7:
            return "Fatiga med"
        return "Fresco"


    def _foul_label(self, player) -> str:
        """
        Etiqueta cualitativa de riesgo por faltas.
        """
        if player.is_fouled_out:
            return "Expulsado"
        if player.fouls >= 5:
            return "Crítico"
        if player.fouls >= 4:
            return "Riesgo"
        return "Normal"