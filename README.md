# NBA Lineup Optimizer mediante Aprendizaje por Refuerzo

Proyecto de Aprendizaje por Refuerzo aplicado a la optimización de rotaciones en baloncesto. Se construyó un entorno propio compatible con Gymnasium, calibrado con lineups reales de los Boston Celtics mediante `nba_api`, y se entrenaron dos agentes: Q-Learning tabular y Deep Q-Network (DQN).

---

## 1. Estructura del proyecto

```text
P2/
│
├── agents/
│   ├── q_learning_agent.py
│   └── dqn_agent.py
│
├── environment/
│   ├── basketball_env.py
│   └── game_simulator.py
│
├── training/
│   ├── train_q_learning.py
│   └── train_dqn.py
│
├── scripts/
│   ├── demo_render.py
│   └── demo_dqn_render.py
│
├── models/
│   ├── q_learning_agent.pkl
│   └── dqn_model.pth
│
├── results/
│   ├── q_learning_metrics.json
│   └── dqn_metrics.json
│
├── figures/
│   ├── reward_q_learning.png
│   ├── reward_dqn.png
│   ├── reward_comparison.png
│   ├── winrate_comparison.png
│   └── epsilon_dqn.png
│
├── data_loader.py
├── plot_results.py
├── evaluate_agent.py
├── test_gym_make.py
├── requirements.txt
└── README.md
```

---

## 2. Crear entorno virtual

Desde la carpeta raíz del proyecto:

```bash
python -m venv venv
```

Activar en Windows:

```bash
venv\Scripts\activate
```

Actualizar `pip`:

```bash
python -m pip install --upgrade pip
```

---

## 3. Instalar dependencias

Instalar desde `requirements.txt`:

```bash
pip install -r requirements.txt
```

Contenido recomendado para `requirements.txt`:

```text
numpy
pandas
matplotlib
gymnasium
torch
nba_api
requests
```

Si no tienes `requirements.txt`, puedes instalar manualmente:

```bash
pip install numpy pandas matplotlib gymnasium torch nba_api requests
```

---

## 4. Descargar o generar datos

El proyecto usa `nba_api` para obtener lineups reales de los Boston Celtics 2024-25.

Ejecutar:

```bash
python data_loader.py
```

Esto genera o carga:

```text
celtics_lineups.json
celtics_players.json
```

Si se quiere forzar el uso de datos sintéticos:

```bash
python data_loader.py --synthetic
```

---

## 5. Probar el entorno Gymnasium

El entorno se registra como:

```text
BasketballLineup-v0
```

Ejecutar:

```bash
python test_gym_make.py
```

Salida esperada:

```text
Entorno creado con gym.make correctamente.
Observation shape: (29,)
Action space: Discrete(44)
Observation space: Box(-1.0, 1.0, (29,), float32)
```

---

## 6. Probar el entorno directamente

Ejecutar:

```bash
python environment/basketball_env.py
```

Este script realiza un smoke test del entorno, ejecuta un episodio y valida compatibilidad con Gymnasium mediante `check_env`.

---

## 7. Entrenar Q-Learning

```bash
python training/train_q_learning.py
```

Salida esperada:

```text
Modelo guardado en: models/q_learning_agent.pkl
Métricas guardadas en: results/q_learning_metrics.json
Estados aprendidos: 22612
Win rate últimos 100 episodios: 97.00%
Reward promedio últimos 100 episodios: +9.281
```

---

## 8. Entrenar DQN

```bash
python training/train_dqn.py
```

Salida esperada:

```text
Modelo guardado en: models/dqn_model.pth
Métricas guardadas en: results/dqn_metrics.json
Win rate últimos 100 episodios: 98.00%
Reward promedio últimos 100 episodios: +43.670
```

---

## 9. Generar gráficas

Una vez entrenados ambos modelos:

```bash
python plot_results.py
```

Esto genera:

```text
figures/reward_q_learning.png
figures/reward_dqn.png
figures/reward_comparison.png
figures/winrate_comparison.png
figures/epsilon_dqn.png
```

---

## 10. Evaluar agentes entrenados

La evaluación se realiza con `epsilon = 0`, es decir, sin exploración aleatoria.

```bash
python evaluate_agent.py
```

Resultado obtenido:

```text
Q-Learning → Reward: 12.57 | Win rate: 96.00%
DQN        → Reward: 61.80 | Win rate: 100.00%
```

---

## 11. Render del entorno

### Render con acciones aleatorias

```bash
python scripts/demo_render.py
```

### Render con DQN entrenado

```bash
python scripts/demo_dqn_render.py
```

Este modo permite observar:

- marcador
- cuarto actual
- tiempo restante
- momentum
- lineup activo
- jugadores en banca
- faltas personales
- fatiga por stint
- recompensa obtenida
- trigger que activó la siguiente decisión

---

## 12. Descripción del MDP

El problema se modela como un Proceso de Decisión de Markov:

```text
M = <S, A, R, P, γ>
```

### Estado S

El estado tiene 29 variables:

- diferencia de marcador
- cuarto actual
- tiempo restante
- momentum
- bonus de faltas
- para cada uno de los 8 jugadores:
  - minutos del stint actual
  - minutos totales
  - faltas personales

### Acciones A

Cada acción corresponde a seleccionar un lineup real de 5 jugadores. El espacio de acciones contiene 44 acciones. Incluye lineups reales filtrados y una acción especial `MAINTAIN` que representa mantener el lineup actual.

### Recompensa R

La recompensa combina:

```text
R = r_net + r_fatiga + r_faltas + r_momentum
```

Donde:

- `r_net`: diferencial de puntos generado durante el stint
- `r_fatiga`: penalización por jugadores fatigados
- `r_faltas`: penalización por riesgo de foul-out
- `r_momentum`: bonus por mantener un lineup con momentum positivo

La recompensa terminal es `+1` si gana, `-1` si pierde.

---

## 13. Triggers de decisión

El agente no actúa en cada segundo, sino cuando ocurre un evento relevante.

| Trigger         | Descripción                             |
| --------------- | --------------------------------------- |
| `QUARTER_START` | Inicio de cuarto                        |
| `TIMEOUT`       | Timeout                                 |
| `FOUL_ALERT`    | Jugador alcanza 3 faltas                |
| `FOUL_CRITICAL` | Jugador alcanza 5 faltas                |
| `FATIGUE`       | Jugador supera umbral crítico de stint  |
| `MOMENTUM_NEG`  | Parcial negativo sostenido              |
| `FORCED`        | Sustitución forzada por expulsión       |
| `GAME_END`      | Final del partido                       |

Estos triggers hacen que el entorno sea más realista que un entorno con pasos de tiempo fijos.

---

## 14. Resultados principales

### Entrenamiento

| Algoritmo  | Reward promedio | Win rate |
| ---------- | --------------: | -------: |
| Q-Learning |          +9.281 |   97.00% |
| DQN        |         +43.670 |   98.00% |

### Evaluación sin exploración

| Algoritmo  | Reward promedio | Win rate |
| ---------- | --------------: | -------: |
| Q-Learning |          +12.57 |   96.00% |
| DQN        |          +61.80 |  100.00% |

---

## 15. Interpretación

Q-Learning aprende una política funcional, pero queda limitado por la discretización del estado. Al agrupar variables continuas en buckets, pierde información relevante.

DQN obtiene mejores resultados porque trabaja directamente con el estado continuo de 29 variables. Esto le permite generalizar mejor entre situaciones parecidas y aprender una política más estable.

Los resultados no deben interpretarse como predicciones reales de partidos NBA, sino como desempeño dentro del simulador construido.

---

## 16. Comandos principales

```bash
# activar entorno
venv\Scripts\activate

# instalar dependencias
pip install -r requirements.txt

# generar datos
python data_loader.py

# probar entorno Gymnasium
python test_gym_make.py

# entrenar Q-Learning
python training/train_q_learning.py

# entrenar DQN
python training/train_dqn.py

# generar gráficas
python plot_results.py

# evaluar agentes
python evaluate_agent.py

# render aleatorio
python scripts/demo_render.py

# render con DQN entrenado
python scripts/demo_dqn_render.py
```

---

## 17. Trabajo futuro

Posibles mejoras:

- usar datos play-by-play reales
- calcular varianza por lineup en lugar de usar aproximación heurística
- incluir rival y matchups defensivos
- probar Double DQN, Dueling DQN o PPO
- mejorar la función de recompensa
- desarrollar un render gráfico avanzado con cancha, jugadores y barras de fatiga
- construir una interfaz interactiva para visualizar decisiones del agente

---

## 18. Nota sobre reproducibilidad

Los entrenamientos usan semillas fijas para facilitar la reproducibilidad:

```text
seed = 42
```

La evaluación usa:

```text
seed = 123
```

Debido a la naturaleza estocástica del entorno, los resultados pueden variar ligeramente entre ejecuciones.

---

## Nota

Los modelos entrenados, métricas y figuras no se incluyen en el repositorio para mantener un tamaño ligero. Todos pueden regenerarse siguiendo las instrucciones de entrenamiento y evaluación descritas en este README.