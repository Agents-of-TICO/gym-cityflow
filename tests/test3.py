import gym_cityflow
import gym
from gym.utils.env_checker import check_env

env = gym.make(
    "cityflow-v0",
    config_path="sample_data/sample_config.json",
    episode_steps=3600,
)

check_env(env.unwrapped)
