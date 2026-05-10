"""
data_loader.py
==============
Carga y preprocesamiento de datos para el entorno NBA Lineup Optimizer.

Responsabilidades:
    - Descargar lineups reales de los Boston Celtics usando nba_api.
    - Guardar los datos en JSON local para entrenamiento offline.
    - Cargar datos desde JSON si ya existen.
    - Usar datos sintéticos solo como respaldo si nba_api no está disponible.

Uso:
    python data_loader.py
    python data_loader.py --synthetic

Archivos generados:
    data/celtics_lineups.json
    data/celtics_players.json
"""

import json
import time
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Intentar importar nba_api
# ---------------------------------------------------------------------------

try:
    from nba_api.stats.endpoints import leaguedashlineups, playerdashboardbygeneralsplits
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    print("[data_loader] nba_api no disponible. Se usará fallback sintético.")


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

CELTICS_TEAM_ID = "1610612738"
SEASON = "2024-25"
MIN_LINEUP_MIN = 5

OUTPUT_DIR = Path(__file__).parent / "data"
LINEUPS_FILE = OUTPUT_DIR / "celtics_lineups.json"
PLAYERS_FILE = OUTPUT_DIR / "celtics_players.json"

# Rotación principal Celtics 2024-25
CELTICS_ROTATION = {
    "1628369": "Jayson Tatum",
    "1627759": "Jaylen Brown",
    "1641785": "Kristaps Porzingis",
    "1628389": "Al Horford",
    "1629684": "Payton Pritchard",
    "1629056": "Jrue Holiday",
    "1629654": "Sam Hauser",
    "1628401": "Derrick White",
}


# ---------------------------------------------------------------------------
# Datos sintéticos de respaldo
# ---------------------------------------------------------------------------

SYNTHETIC_LINEUPS = [
    {
        "id": "L00",
        "players": [
            "Jayson Tatum",
            "Jaylen Brown",
            "Jrue Holiday",
            "Al Horford",
            "Kristaps Porzingis",
        ],
        "mu": 8.2,
        "sigma": 4.1,
        "minutes": 187,
    },
    {
        "id": "L01",
        "players": [
            "Jayson Tatum",
            "Jaylen Brown",
            "Jrue Holiday",
            "Sam Hauser",
            "Kristaps Porzingis",
        ],
        "mu": 6.5,
        "sigma": 4.8,
        "minutes": 134,
    },
    {
        "id": "L02",
        "players": [
            "Jayson Tatum",
            "Jaylen Brown",
            "Payton Pritchard",
            "Al Horford",
            "Kristaps Porzingis",
        ],
        "mu": 5.1,
        "sigma": 5.2,
        "minutes": 98,
    },
    {
        "id": "L03",
        "players": [
            "Jayson Tatum",
            "Jaylen Brown",
            "Jrue Holiday",
            "Sam Hauser",
            "Al Horford",
        ],
        "mu": 7.3,
        "sigma": 4.5,
        "minutes": 112,
    },
    {
        "id": "L04",
        "players": [
            "Jayson Tatum",
            "Jaylen Brown",
            "Payton Pritchard",
            "Sam Hauser",
            "Derrick White",
        ],
        "mu": 3.8,
        "sigma": 6.1,
        "minutes": 76,
    },
    {
        "id": "L05",
        "players": [
            "Jaylen Brown",
            "Jrue Holiday",
            "Sam Hauser",
            "Al Horford",
            "Kristaps Porzingis",
        ],
        "mu": 4.2,
        "sigma": 5.5,
        "minutes": 89,
    },
    {
        "id": "L06",
        "players": [
            "Jaylen Brown",
            "Payton Pritchard",
            "Sam Hauser",
            "Al Horford",
            "Derrick White",
        ],
        "mu": 2.1,
        "sigma": 6.3,
        "minutes": 67,
    },
    {
        "id": "L07",
        "players": [
            "Jaylen Brown",
            "Jrue Holiday",
            "Payton Pritchard",
            "Sam Hauser",
            "Kristaps Porzingis",
        ],
        "mu": 3.5,
        "sigma": 5.8,
        "minutes": 72,
    },
    {
        "id": "L08",
        "players": [
            "Jayson Tatum",
            "Jrue Holiday",
            "Sam Hauser",
            "Al Horford",
            "Kristaps Porzingis",
        ],
        "mu": 5.8,
        "sigma": 4.9,
        "minutes": 95,
    },
    {
        "id": "L09",
        "players": [
            "Jayson Tatum",
            "Payton Pritchard",
            "Sam Hauser",
            "Al Horford",
            "Kristaps Porzingis",
        ],
        "mu": 4.4,
        "sigma": 5.6,
        "minutes": 81,
    },
    {
        "id": "L10",
        "players": [
            "Payton Pritchard",
            "Sam Hauser",
            "Jrue Holiday",
            "Derrick White",
            "Al Horford",
        ],
        "mu": 1.5,
        "sigma": 7.2,
        "minutes": 54,
    },
    {
        "id": "L11",
        "players": [
            "Payton Pritchard",
            "Sam Hauser",
            "Derrick White",
            "Al Horford",
            "Kristaps Porzingis",
        ],
        "mu": 0.8,
        "sigma": 7.8,
        "minutes": 48,
    },
    {
        "id": "L12",
        "players": [
            "Jayson Tatum",
            "Jaylen Brown",
            "Jrue Holiday",
            "Al Horford",
            "Sam Hauser",
        ],
        "mu": 9.1,
        "sigma": 3.8,
        "minutes": 143,
    },
    {
        "id": "L13",
        "players": [
            "Jayson Tatum",
            "Jaylen Brown",
            "Jrue Holiday",
            "Derrick White",
            "Sam Hauser",
        ],
        "mu": 5.0,
        "sigma": 5.3,
        "minutes": 61,
    },
    {
        "id": "L14",
        "players": [
            "Jayson Tatum",
            "Jaylen Brown",
            "Payton Pritchard",
            "Jrue Holiday",
            "Al Horford",
        ],
        "mu": 6.8,
        "sigma": 4.6,
        "minutes": 88,
    },
    {
        "id": "L15",
        "players": [
            "Jaylen Brown",
            "Jrue Holiday",
            "Sam Hauser",
            "Al Horford",
            "Derrick White",
        ],
        "mu": 2.9,
        "sigma": 6.0,
        "minutes": 59,
    },
    {
        "id": "L16",
        "players": [
            "Jayson Tatum",
            "Payton Pritchard",
            "Jrue Holiday",
            "Sam Hauser",
            "Kristaps Porzingis",
        ],
        "mu": 4.7,
        "sigma": 5.4,
        "minutes": 70,
    },
    {
        "id": "L17",
        "players": [
            "Jayson Tatum",
            "Jaylen Brown",
            "Sam Hauser",
            "Al Horford",
            "Kristaps Porzingis",
        ],
        "mu": 7.6,
        "sigma": 4.3,
        "minutes": 105,
    },
    {
        "id": "L18",
        "players": [
            "Jaylen Brown",
            "Payton Pritchard",
            "Jrue Holiday",
            "Al Horford",
            "Kristaps Porzingis",
        ],
        "mu": 3.3,
        "sigma": 5.9,
        "minutes": 63,
    },
    {
        "id": "L19",
        "players": [
            "Jayson Tatum",
            "Jaylen Brown",
            "Jrue Holiday",
            "Payton Pritchard",
            "Sam Hauser",
        ],
        "mu": 6.1,
        "sigma": 4.7,
        "minutes": 79,
    },
]


SYNTHETIC_PLAYERS = {
    "Jayson Tatum": {
        "id": "1628369",
        "position": "SF",
        "avg_minutes": 36.5,
        "avg_fouls": 2.1,
        "fatigue_threshold_stint": 8,
        "fatigue_threshold_total": 32,
        "foul_out_risk_threshold": 4,
    },
    "Jaylen Brown": {
        "id": "1627759",
        "position": "SG",
        "avg_minutes": 34.8,
        "avg_fouls": 2.4,
        "fatigue_threshold_stint": 8,
        "fatigue_threshold_total": 30,
        "foul_out_risk_threshold": 4,
    },
    "Kristaps Porzingis": {
        "id": "1641785",
        "position": "C",
        "avg_minutes": 28.3,
        "avg_fouls": 2.8,
        "fatigue_threshold_stint": 6,
        "fatigue_threshold_total": 25,
        "foul_out_risk_threshold": 3,
    },
    "Al Horford": {
        "id": "1628389",
        "position": "PF",
        "avg_minutes": 29.1,
        "avg_fouls": 1.9,
        "fatigue_threshold_stint": 7,
        "fatigue_threshold_total": 26,
        "foul_out_risk_threshold": 4,
    },
    "Payton Pritchard": {
        "id": "1629684",
        "position": "PG",
        "avg_minutes": 26.4,
        "avg_fouls": 1.6,
        "fatigue_threshold_stint": 9,
        "fatigue_threshold_total": 28,
        "foul_out_risk_threshold": 4,
    },
    "Jrue Holiday": {
        "id": "1629056",
        "position": "PG",
        "avg_minutes": 31.2,
        "avg_fouls": 2.0,
        "fatigue_threshold_stint": 8,
        "fatigue_threshold_total": 28,
        "foul_out_risk_threshold": 4,
    },
    "Sam Hauser": {
        "id": "1629654",
        "position": "SF",
        "avg_minutes": 24.7,
        "avg_fouls": 1.4,
        "fatigue_threshold_stint": 9,
        "fatigue_threshold_total": 26,
        "foul_out_risk_threshold": 4,
    },
    "Derrick White": {
        "id": "1628401",
        "position": "SG",
        "avg_minutes": 33.0,
        "avg_fouls": 2.1,
        "fatigue_threshold_stint": 8,
        "fatigue_threshold_total": 30,
        "foul_out_risk_threshold": 4,
    },
}


# ---------------------------------------------------------------------------
# Funciones API
# ---------------------------------------------------------------------------

def normalize_player_name(name: str) -> str:
    mapping = {
        "J. Tatum": "Jayson Tatum",
        "J. Brown": "Jaylen Brown",
        "J. Holiday": "Jrue Holiday",
        "A. Horford": "Al Horford",
        "K. Porziņģis": "Kristaps Porzingis",
        "K. Porzingis": "Kristaps Porzingis",
        "P. Pritchard": "Payton Pritchard",
        "S. Hauser": "Sam Hauser",
        "D. White": "Derrick White",
    }

    return mapping.get(name.strip(), name.strip())

def fetch_lineup_data_from_api() -> list[dict]:
    """
    Descarga lineups reales desde nba_api.

    Nota metodológica:
        nba_api permite obtener Net Rating y minutos de muestra por lineup.
        Sin embargo, leaguedashlineups no proporciona directamente la varianza
        por stint. Por eso, sigma se aproxima heurísticamente en esta versión.
    """
    print(f"[data_loader] Descargando lineups Celtics {SEASON} desde nba_api...")
    time.sleep(1)

    endpoint = leaguedashlineups.LeagueDashLineups(
        team_id_nullable=CELTICS_TEAM_ID,
        season=SEASON,
        season_type_all_star="Regular Season",
        group_quantity=5,
        per_mode_detailed="Per100Possessions",
        measure_type_detailed_defense="Advanced",
    )

    df = endpoint.get_data_frames()[0]

    print("[data_loader] Columnas recibidas desde nba_api:")
    print(df.columns.tolist())

    print("[data_loader] Primeros GROUP_NAME recibidos:")
    print(df[["GROUP_NAME", "MIN", "NET_RATING"]].head(10))

    df = df[df["MIN"] >= MIN_LINEUP_MIN].copy()

    lineups = []

    allowed_players = set(SYNTHETIC_PLAYERS.keys())

    for i, row in df.iterrows():
        players = [
            normalize_player_name(p)
            for p in row["GROUP_NAME"].split(" - ")
        ]

        # Filtrar lineups que tengan jugadores fuera de la rotación de 8
        if not all(player in allowed_players for player in players):
            continue

        lineups.append(
            {
                "id": f"L{len(lineups):02d}",
                "players": players,
                "mu": float(row["NET_RATING"]),
                "sigma": 6.0,
                "minutes": float(row["MIN"]),
            }
        )


    print(f"[data_loader] {len(lineups)} lineups válidos encontrados.")

    if len(lineups) == 0:
        raise RuntimeError("No se encontraron lineups válidos desde nba_api.")

    return lineups


def fetch_player_data_from_api() -> dict:
    """
    Descarga estadísticas individuales básicas desde nba_api.
    """
    players = {}

    for player_id, player_name in CELTICS_ROTATION.items():
        print(f"[data_loader] Descargando stats de {player_name}...")
        time.sleep(0.6)

        endpoint = playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits(
            player_id=player_id,
            season=SEASON,
            season_type_playoffs="Regular Season",
            per_mode_detailed="PerGame",
            measure_type_detailed="Base",
        )

        df = endpoint.get_data_frames()[0]

        if df.empty:
            continue

        row = df.iloc[0]
        avg_minutes = float(row.get("MIN", 25.0))
        avg_fouls = float(row.get("PF", 2.0))

        players[player_name] = {
            "id": player_id,
            "position": "UNK",
            "avg_minutes": avg_minutes,
            "avg_fouls": avg_fouls,
            "fatigue_threshold_stint": max(6, min(9, int(avg_minutes * 0.22))),
            "fatigue_threshold_total": int(avg_minutes * 0.85),
            "foul_out_risk_threshold": 4,
        }

    if len(players) == 0:
        raise RuntimeError("No se pudieron descargar jugadores desde nba_api.")

    return players


# ---------------------------------------------------------------------------
# Carga principal
# ---------------------------------------------------------------------------

def load_or_fetch_data(force_synthetic: bool = False) -> tuple[list, dict]:
    """
    Carga datos desde JSON local, desde nba_api o desde fallback sintético.
    """

    if LINEUPS_FILE.exists() and PLAYERS_FILE.exists() and not force_synthetic:
        print("[data_loader] Cargando datos desde JSON local...")

        with open(LINEUPS_FILE, "r", encoding="utf-8") as f:
            lineups = json.load(f)

        with open(PLAYERS_FILE, "r", encoding="utf-8") as f:
            players = json.load(f)

        print(f"[data_loader] {len(lineups)} lineups y {len(players)} jugadores cargados.")
        return lineups, players

    if NBA_API_AVAILABLE and not force_synthetic:
        try:
            lineups = fetch_lineup_data_from_api()
            players = SYNTHETIC_PLAYERS
        except Exception as e:
            print(f"[data_loader] Error usando nba_api: {e}")
            print("[data_loader] Usando datos sintéticos de respaldo.")
            lineups = SYNTHETIC_LINEUPS
            players = SYNTHETIC_PLAYERS
    else:
        print("[data_loader] Usando datos sintéticos.")
        lineups = SYNTHETIC_LINEUPS
        players = SYNTHETIC_PLAYERS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(LINEUPS_FILE, "w", encoding="utf-8") as f:
        json.dump(lineups, f, indent=2, ensure_ascii=False)

    with open(PLAYERS_FILE, "w", encoding="utf-8") as f:
        json.dump(players, f, indent=2, ensure_ascii=False)

    print(f"[data_loader] Datos guardados en {OUTPUT_DIR}")

    return lineups, players


def compute_lineup_stats(lineups: list[dict]) -> dict:
    """
    Convierte la lista de lineups en un diccionario indexado por lineup_id.
    """
    total_minutes = max(1.0, sum(float(l["minutes"]) for l in lineups))
    stats = {}

    for lineup in lineups:
        stats[lineup["id"]] = {
            "players": lineup["players"],
            "mu": float(lineup["mu"]),
            "sigma": max(float(lineup["sigma"]), 2.0),
            "minutes": float(lineup["minutes"]),
            "sample_weight": float(lineup["minutes"]) / total_minutes,
        }

    return stats


# ---------------------------------------------------------------------------
# Ejecución standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera JSON locales para el entorno NBA Lineup Optimizer."
    )

    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Forzar datos sintéticos sin usar nba_api.",
    )

    args = parser.parse_args()

    lineups, players = load_or_fetch_data(force_synthetic=args.synthetic)
    stats = compute_lineup_stats(lineups)

    print("\n=== Lineups cargados ===")
    for lid, s in list(stats.items())[:10]:
        print(
            f"{lid}: {s['players']} | "
            f"mu={s['mu']:+.1f}, sigma={s['sigma']:.1f}, "
            f"min={s['minutes']:.0f}"
        )

    print("\n=== Jugadores cargados ===")
    for name, p in players.items():
        print(
            f"{name}: {p['avg_minutes']:.1f} min/game, "
            f"fatigue_stint={p['fatigue_threshold_stint']}, "
            f"fatigue_total={p['fatigue_threshold_total']}"
        )
