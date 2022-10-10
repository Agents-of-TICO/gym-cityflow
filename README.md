# gym-cityflow

`gym_cityflow` is a custom OpenAI gym environment designed to handle any cityflow config file.
This is a fork of the original `gym_cityflow` environment by [MaxVanDijck](https://github.com/MaxVanDijck/gym-cityflow)
and has been updated to work with current versions of OpenAI Gym (v0.21.0).

## Prerequisites

As an OpenAI Gym environment that implements the CityFlow simulation engine the following 
prerequisites are required to use this environment.

- [OpenAI Gym](https://www.gymlibrary.dev/)
- [CityFlow](https://cityflow.readthedocs.io/en/latest/install.html)

## Installation

`gym_cityflow` is currently not a part of any package can be installed by running the following at the root directory of the repository:

`pip install -e .`

`gym_cityflow` can then be used as a python library as follows:
NOTE: configPath must be a valid cityflow `config.json` file, episodeSteps is how many steps the environment will take before it is done

```python
import gym
import gym_cityflow

env = gym.make('cityflow-v0', 
               configPath = 'sample_path/sample_config.json',
               episodeSteps = 3600)
```

## Basic Functionality

The action and observation space can be check like so:

```python
observationSpace = env.observation_space
actionSpace = env.action_space
```

`env.step()` can be called to step the environment, it returns an observation, reward, done and debug as specified in the [OpenAI Documentation](https://gym.openai.com/docs/)

`env.reset()` can be called to restart the environment

```python
observation, reward, done, debug = env.step(action=sampleAction)

if done:
    env.reset()
```
