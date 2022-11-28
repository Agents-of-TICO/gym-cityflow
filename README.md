# gym-cityflow

`gym_cityflow` is a custom OpenAI gym environment designed to handle any cityflow config file.
This is a fork of the original `gym_cityflow` environment by [MaxVanDijck](https://github.com/MaxVanDijck/gym-cityflow)
and has been updated to work with current versions of OpenAI Gym (v0.21.0).

## Prerequisites

As an OpenAI Gym environment that implements the CityFlow simulation engine the following 
prerequisites are required to use this environment.

- [OpenAI Gym](https://www.gymlibrary.dev/)
- [CityFlow](https://cityflow.readthedocs.io/en/latest/install.html)

Unfortunately, as CityFlow is not registered with a python package index you need to clone the repository
and install the package using the 'pip install -e .' command from the root directory of the repository.

## Installation

`gym_cityflow` is currently not a part of any package must be manually installed by running the following 
commands in the root directory of the repository after downloading/cloning the repository:

`pip install -e .`

`gym_cityflow` can then be used as a python library as follows:

```python
import gym
import gym_cityflow

env = gym.make('cityflow-v0', 
               config_path = 'sample_path/sample_config.json',
               episode_steps = 3600)
```
NOTE: config_path must be a valid CityFlow `config.json` file, episode_steps is how many steps the environment will 
take before it is done

## Basic Functionality

The action and observation space can be checked like so:

```python
observationSpace = env.observation_space
actionSpace = env.action_space
```

`env.step()` can be called to step the environment, it returns an observation, reward, done and info as specified in
the [OpenAI Documentation](https://gym.openai.com/docs/)

`env.reset()` can be called to restart the environment

`env.reset(seed=42)` can also be called with a new seed to restart the environment with a new seed

Here is an example of how to train and run PPO on a CityFlow environment by using [stable-baselines3](https://github.com/DLR-RM/stable-baselines3):

```python
import gym
import gym_cityflow
from stable_baselines3 import PPO

env = gym.make('cityflow-v0', config_path="examples/default/config.json", episode_steps=1000)
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

obs = env.reset()

for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

env.close()
```

For convenience, we have added some example CityFlow config.json, roadnet.json, and flow.json files 
to 'examples/default' and 'examples/double_intersection'.
