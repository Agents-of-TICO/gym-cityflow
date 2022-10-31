from statistics import mean

import gym
from gym import spaces
import cityflow
import json


class CityFlowEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "max_waiting": 128}

    def __init__(self, config_path, episode_steps=10000, num_threads=1, reward_fun=1 render_mode=None):
        self.episode_steps = episode_steps  # The number of steps to simulate
        self.current_step = 0
        self.total_wait_time = 0
        self.min_phase_time = 24
        self.transition_phase_time = 3
        self.phase_times = []
        self.reward_fun = reward_fun
        # self.reward_range = (-float("inf"), float(1))

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
        self.intersection_phases = len(intersection['trafficLight']['lightphases']) - 1
        self.action_space = spaces.Discrete(self.intersection_phases)
        self._intersection_id = intersection['id']

        # create cityflow engine
        self.eng = cityflow.Engine(config_path, thread_num=num_threads)

        # Observations are dictionaries containing the number of waiting vehicles in each lane in the simulation.
        # Maximum number of waiting vehicles in each lane is defined by the "max_waiting" metadata parameter
        observation_space_dict = self.eng.get_lane_waiting_vehicle_count()
        for key in observation_space_dict:
            observation_space_dict[key] = spaces.Discrete(self.metadata["max_waiting"])
        self.observation_space = spaces.Dict(observation_space_dict)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        return self.eng.get_lane_waiting_vehicle_count()

    def _get_info(self):
        return {"steps_in_current_phase": self.steps_in_current_phase,
                "last_action": self.last_action
                }

    def _get_reward(self):
        if self.reward_fun == 1:
            num_waiting = sum(self.eng.get_lane_waiting_vehicle_count().values())
            reward = -num_waiting  # Negate the value since we want higher values to represent better performance

        # Queue squared reward
        elif self.reward_fun == 2:
            reward = -1 * (sum(self.eng.get_lane_waiting_vehicle_count().values()))^2

        # Average Speed reward function
        elif self.reward_fun == 3:
            reward = sum(self.eng.get_vehicle_speed().values()/ 16.67) / sum(self.eng.get_vehicle_count())

        return reward

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

        observation = self.eng.get_lane_waiting_vehicle_count()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        # The Newest version of gym has info returned w/ reset but this causes issues with stable baselines 3
        return observation # , info

    def step(self, action):
        # Set each traffic light phase to specified action. If a change in phase is requested, go to 'yellow' phase
        # (phase: 0) for self.transition_phase_time before changing to new phase
        action += 1  # increment selected phase by one since 0 is yellow phase

        # If we are in the same phase as last time increment the relevant value in steps_in_current_phase
        if self.last_action == 0 and self.steps_in_current_phase < self.transition_phase_time:
            action = 0    # Once we go to the 'all-red' state stay there for transition_phase_time steps
            self.steps_in_current_phase += 1
        elif self.last_action == 0:
            self.eng.set_tl_phase(self._intersection_id, action)
            self.steps_in_current_phase = 1
        elif self.last_action == action:
            self.steps_in_current_phase += 1
        else:
            action = 0
            self.eng.set_tl_phase(self._intersection_id, action)
            self.phase_times.append(self.steps_in_current_phase)
            self.steps_in_current_phase = 1

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

    def render(self):
        # Function called to render environment
        print("Current time: " + str(self.eng.get_current_time()))
        print("Running Total wait time: " + str(self.total_wait_time))

    def close(self):
        # if we need to do anything on env exit this is where we do it
        print("Exiting...")
        print("Total wait time: " + str(self.total_wait_time))
