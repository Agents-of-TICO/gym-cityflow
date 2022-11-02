# gym-cityflow

`gym_cityflow` is a custom OpenAI gym environment designed to handle any cityflow config file.
This is a fork of the original `gym_cityflow` environment by [MaxVanDijck](https://github.com/MaxVanDijck/gym-cityflow)
and has been updated to work with current versions of OpenAI Gym (v0.21.0).

## Prerequisites

As an OpenAI Gym environment that implements the CityFlow simulation engine the following 
prerequisites are required to use this environment:

- [OpenAI Gym](https://www.gymlibrary.dev/)
- [CityFlow](https://cityflow.readthedocs.io/en/latest/install.html)

OpenAI Gym can be installed via pip with the following command:

'pip install gym'

On the other hand, as CityFlow is not registered with a python package index you need to build the package
from the source code:

Install Dependencies:
`sudo apt update && sudo apt install -y build-essential cmake`

Clone CityFlow project from github:
`git clone https://github.com/cityflow-project/CityFlow.git`

Then go to CityFlow project’s root directory and run:
`pip install .`

## Installation

`gym_cityflow` is currently not a part of any package must be manually installed from the source:

Clone the gym-cityflow project from github:
`git clone https://github.com/Agents-of-TICO/gym-cityflow.git`

Then go to  the gym-cityflow project’s root directory and run:
`pip install .`

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

Here is an example of how to train and run PPO on a CityFlow environment (NOTE: this example uses the PPO model provided
by stable-baselines3 to facilitate training, if you want to run this example you should install stable-baselines3 with the
command `pip install stable-baselines3`)

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
