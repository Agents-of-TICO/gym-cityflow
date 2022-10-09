import gym_cityflow
import gym
from gym.utils.env_checker import check_env

env = gym.make(
    "cityflow-v0",
    config_path="examples/config2.json",
    episode_steps=3600,
)

check_env(env.unwrapped)
