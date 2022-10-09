import gym
from gym import spaces
import cityflow
import json


class CityFlowEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "max_waiting": 64}

    def __init__(self, config_path, episode_steps=10000, num_threads=1, render_mode=None):
        self.episode_steps = episode_steps  # The number of steps to simulate
        self.current_step = 0
        self.total_wait_time = 0

        # open cityflow config file into dict
        self.configDict = json.load(open(config_path))
        # open cityflow roadnet file into dict
        self.roadnetDict = json.load(open(self.configDict['dir'] + self.configDict['roadnetFile']))
        self.flowDict = json.load(open(self.configDict['dir'] + self.configDict['flowFile']))

        # Get list of non-virtual intersections
        intersections = list(filter(lambda val: not val['virtual'], self.roadnetDict['intersections']))

        # Get number of available phases available in each intersection and use it to create the action
        # space since each intersection has a number of actions equal to the number of states/phases the
        # intersection has. Here we also generate a dictionary to get the id of an intersection given an index
        intersection_phases = [None]*len(intersections)
        index_to_intersection_id = {}
        for i, intersection in enumerate(intersections):
            intersection_phases[i] = len(intersection['trafficLight']['lightphases'])
            index_to_intersection_id[i] = intersection['id']
        self.action_space = spaces.MultiDiscrete(intersection_phases)
        self._index_to_intersection_id = index_to_intersection_id

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
        return {}

    def _get_reward(self):
        num_waiting = sum(self.eng.get_lane_waiting_vehicle_count().values())
        return 1 / (num_waiting + 1)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)

        print("Total wait time: " + str(self.total_wait_time))

        if seed is not None:
            self.eng.set_random_seed(seed)
        self.eng.reset(seed=False)
        self.current_step = 0
        self.total_wait_time = 0

        observation = self.eng.get_lane_waiting_vehicle_count()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation

    def step(self, action):
        # Check that input action size is equal to number of intersections
        if len(action) != len(self._index_to_intersection_id):
            raise Warning('Action length not equal to number of intersections')

        # Set each traffic light phase to specified action
        for i, phase in enumerate(action):
            self.eng.set_tl_phase(self._index_to_intersection_id[i], phase)

        # Step the CityFlow env
        self.eng.next_step()

        # increment the step counter
        self.current_step += 1

        # add current wait time to total
        self.total_wait_time = sum(self.eng.get_lane_waiting_vehicle_count().values())

        # An episode is done once we have simulated the number of steps defined in episode_steps
        terminated = self.episode_steps == self.current_step
        reward = self._get_reward()
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, info

    def render(self):
        # Function called to render environment
        print("Current time: " + self.eng.get_current_time())
        print("Running Total wait time: " + str(self.total_wait_time))

    def close(self):
        # if we need to do anything on env exit this is where we do it
        print("Exiting...")
        print("Total wait time: " + str(self.total_wait_time))
