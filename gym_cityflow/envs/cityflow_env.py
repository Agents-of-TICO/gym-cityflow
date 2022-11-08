import sys
from random import random
from statistics import mean

import gym
from gym import spaces
import cityflow
import json


class CityFlowEnv(gym.Env):
    metadata = {"render_modes": ["human"], "max_waiting": 128,
                "reward_funcs": ["queueSum", "queueSquared", "phaseTime", "queue&Time", "queue&TimeF", "avgSpeed", "phaseTime"]
                }

    def __init__(self, config_path, episode_steps=10000, num_threads=1, reward_func="queueSum", seed=None,
                 render_mode=None):
        self.episode_steps = episode_steps  # The number of steps to simulate
        self.current_step = 0
        self.total_wait_time = 0
        self.phase_step_goal = 24
        self.transition_phase_time = 3
        self.max_phase_time = 64
        self.config_path = config_path
        self.num_threads = num_threads
        self.phase_times = []
        # self.reward_range = (-float("inf"), float(1))

        assert reward_func in self.metadata["reward_funcs"]
        self.reward_func = reward_func

        self.reward_func_dict = {"queueSum": self._get_reward_queue_sum,
                                 "queueSquared": self._get_reward_queue_squared,
                                 "avgSpeed": self._get_reward_avg_speed,
                                 "queue&Time": self._get_reward_sum_and_phase_time,
                                 "queue&TimeF": self._get_reward_sum_and_phase_time_flat,
                                 "phaseTime": self._get_reward_phase_time
                                 }

        print(f"Using reward function: {self.reward_func_dict[reward_func].__name__}")

        # open cityflow config file into dict
        self.configDict = json.load(open(config_path))
        self.interval = self.configDict['interval']
        # open cityflow roadnet file into dict
        self.roadnetDict = json.load(open(self.configDict['dir'] + self.configDict['roadnetFile']))
        self.flowDict = json.load(open(self.configDict['dir'] + self.configDict['flowFile']))

        # Get list of non-virtual intersections
        intersection = list(filter(lambda val: not val['virtual'], self.roadnetDict['intersections']))[0]

        # Initialize
        self.steps_in_current_phase = 0
        self.last_action = 0

        # Get number of available phases available in each intersection and use it to create the action
        # space since each intersection has a number of actions equal to the number of states/phases the
        # intersection has. Here we also generate a dictionary to get the id of an intersection given an index
        # Subtract one to remove yellow phase from action space
        self.num_phases = len(intersection['trafficLight']['lightphases']) - 1
        self.action_space = spaces.Discrete(self.num_phases)
        self._intersection_id = intersection['id']

        # create cityflow engine
        self.eng = cityflow.Engine(config_path, thread_num=num_threads)

        # set seed if given
        if seed is not None:
            self.eng.set_random_seed(seed)

        # Observations are dictionaries containing the number of waiting vehicles in each lane in the simulation.
        # Maximum number of waiting vehicles in each lane is defined by the "max_waiting" metadata parameter
        observation_space_dict = self.eng.get_lane_waiting_vehicle_count()
        for key in observation_space_dict:
            observation_space_dict[key] = spaces.Discrete(self.metadata["max_waiting"])
        observation_space_dict["steps_in_phase"] = spaces.Discrete(self.max_phase_time)
        self.observation_space = spaces.Dict(observation_space_dict)

        # Verify and set render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        # Get Dictionary where keys are lane id's and values are the # of waiting vehicles in the lane
        obs = self.eng.get_lane_waiting_vehicle_count()
        # Add steps in current phase to dictionary
        obs["steps_in_phase"] = self.steps_in_current_phase
        return obs

    def _get_info(self):
        return {"steps_in_current_phase": self.steps_in_current_phase,
                "last_action": self.last_action
                }

    def _get_reward(self):
        return self.reward_func_dict[self.reward_func]()

    # Sum of waiting vehicles
    def _get_reward_queue_sum(self):
        return -sum(self.eng.get_lane_waiting_vehicle_count().values())

    # Queue squared reward
    def _get_reward_queue_squared(self):
        return -1 * (sum(self.eng.get_lane_waiting_vehicle_count().values()))^2

    # Time in current phase relative to phase_step_goal
    def _get_reward_phase_time(self):
        reward = None
        if 2 <= self.steps_in_current_phase <= self.phase_step_goal:
            # If the same phase as last time is selected, give reward proportional to the number of steps we have been
            # in the current stage.
            reward = 128 * self.steps_in_current_phase
        elif 2 > self.steps_in_current_phase:
            # If the phase was just changed, give a positive reward if the env spent more steps in the phase than
            # defined in self.phase_step_goal, decreasing the reward if the env exceeds the value in
            # self.phase_step_goal. If the env spends less time in a phase that the time given in self.phase_step_goal,
            # provide a negative reward proportional to difference.
            if self.phase_times[-1] >= self.phase_step_goal:
                reward = 128 * self.phase_step_goal - 128 * (self.phase_times[-1] - self.phase_step_goal)
            else:
                reward = -128 * (self.phase_step_goal - self.phase_times[-1])
        else:
            # If the reward function gets here we have picked the same phase for more steps than is defined by
            # self.phase_step_goal, so we provide a negative reward proportional to the number of steps we exceed
            # self.phase_step_goal by
            reward = -128 * (self.steps_in_current_phase - self.phase_step_goal)
        return reward

    # One over Sum of waiting vehicles plus wait time
    def _get_reward_sum_and_phase_time(self):
        reward = 1 / (1 + sum(self.eng.get_lane_waiting_vehicle_count().values()))
        if 2 <= self.steps_in_current_phase <= self.phase_step_goal:
            reward += self.steps_in_current_phase / self.phase_step_goal
        return reward

    # One over Sum of waiting vehicles plus flat wait time reward
    def _get_reward_sum_and_phase_time_flat(self):
        reward = 1 / (1 + sum(self.eng.get_lane_waiting_vehicle_count().values()))
        if 2 <= self.steps_in_current_phase <= self.phase_step_goal:
            reward += 5
        elif 2 > self.steps_in_current_phase:
            reward -= 2
        return reward

    # Average Speed reward function
    def _get_reward_avg_speed(self):
        return (sum(self.eng.get_vehicle_speed().values()) / 16.67) / self.eng.get_vehicle_count()

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)

        print("Total wait time: " + str(self.total_wait_time))
        if len(self.phase_times) > 0:
            print(f"Average phase time: {mean(self.phase_times)} seconds")

        if seed is not None:
            self.eng.set_random_seed(seed)
        self.eng.reset(seed=False)
        self.current_step = 0
        self.total_wait_time = 0
        self.phase_times = []
        self.steps_in_current_phase = 0
        self.last_action = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        # The Newest version of gym has info returned w/ reset but this causes issues with stable baselines 3
        return observation # , info

    def step(self, action):
        # Set each traffic light phase to specified action. If a change in phase is requested, go to 'yellow' phase
        # (phase: 0) for self.transition_phase_time before changing to new phase
        action += 1  # increment selected phase by one since 0 is yellow phase

        # If we reach or exceed the maximum phase time allowed switch to a random phase
        if self.steps_in_current_phase >= self.max_phase_time and self.last_action == action:
            action = random.choice([range(1, self.num_phases + 1)].remove(self.last_action))

        # If we are in the same phase as last time increment the relevant value in steps_in_current_phase
        if self.last_action == action:
            self.steps_in_current_phase += 1
        else:
            self._transition_phase(action)

        # Step the CityFlow env
        self.eng.next_step()

        # increment the step counter
        self.current_step += 1

        # add current wait time to total
        self.total_wait_time += sum(self.eng.get_lane_waiting_vehicle_count().values())*self.interval

        # An episode is done once we have simulated the number of steps defined in episode_steps
        terminated = self.episode_steps == self.current_step
        reward = self._get_reward()
        observation = self._get_obs()
        info = self._get_info()
        truncated = False

        if self.render_mode == "human":
            self.render()

        # Update last action taken
        self.last_action = action

        return observation, reward, terminated, info

        # New return statement for updated gym, commented because stable baselines 3 hasn't updated
        # to account for these changes
        # return observation, reward, terminated, truncated, info

    def _transition_phase(self, next_phase):
        # Switch to the given phase after taking a predefined number of steps in a 'red'/transition phase

        # Set intersection to intermediate phase and take self.steps_in_current_phase steps
        self.eng.set_tl_phase(self._intersection_id, 0)
        for i in range(self.transition_phase_time):
            self.eng.next_step()
        # Switch to given phase
        self.eng.set_tl_phase(self._intersection_id, next_phase)

        # Record and reset self.steps_in_current_phase
        self.phase_times.append(self.steps_in_current_phase)
        self.steps_in_current_phase = 1

    def load_fresh_engine(self):
        # Recreate the engine with the same params to generate fresh replay file
        self.eng = cityflow.Engine(self.config_path, thread_num=self.num_threads)

    def get_phase_times(self):
        return self.phase_times

    def render(self):
        # Function called to render environment
        print("Current time: " + str(self.eng.get_current_time()))
        print("Running Total wait time: " + str(self.total_wait_time))

    def close(self):
        # if we need to do anything on env exit this is where we do it
        print("Total wait time: " + str(self.total_wait_time))
        if len(self.phase_times) > 0:
            print(f"Average phase time: {mean(self.phase_times)} seconds")
        print("Exiting...")
