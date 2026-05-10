import gymnasium as gym

from environment.basketball_env import register_env

register_env()

env = gym.make("BasketballLineup-v0")

obs, info = env.reset(seed=42)

print("Entorno creado con gym.make correctamente.")
print("Observation shape:", obs.shape)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

env.close()