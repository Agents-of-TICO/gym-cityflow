import json
import cityflow
import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding


class Cityflow(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, config_path, episode_steps, render_mode=None):
        # steps per episode
        self.steps_per_episode = episode_steps
        self.is_done = False
        self.current_step = 0

        # open cityflow config file into dict
        self.configDict = json.load(open(config_path))
        # open cityflow roadnet file into dict
        self.roadnetDict = json.load(open(self.configDict['dir'] + self.configDict['roadnetFile']))
        self.flowDict = json.load(open(self.configDict['dir'] + self.configDict['flowFile']))

        # create dict of controllable intersections and number of light phases
        self.intersections = {}
        for i in range(len(self.roadnetDict['intersections'])):
            # check if intersection is controllable
            if not self.roadnetDict['intersections'][i]['virtual']:
                # for each roadLink in intersection store incoming lanes, outgoing lanes and direction in lists
                incoming_lanes = []
                outgoing_lanes = []
                directions = []
                for j in range(len(self.roadnetDict['intersections'][i]['roadLinks'])):
                    incoming_roads = []
                    outgoing_roads = []
                    directions.append(self.roadnetDict['intersections'][i]['roadLinks'][j]['direction'])
                    for k in range(len(self.roadnetDict['intersections'][i]['roadLinks'][j]['laneLinks'])):
                        incoming_roads.append(self.roadnetDict['intersections'][i]['roadLinks'][j]['startRoad'] +
                                              '_' +
                                              str(self.roadnetDict['intersections'][i]['roadLinks'][j]['laneLinks'][k][
                                                      'startLaneIndex']))
                        outgoing_roads.append(self.roadnetDict['intersections'][i]['roadLinks'][j]['endRoad'] +
                                              '_' +
                                              str(self.roadnetDict['intersections'][i]['roadLinks'][j]['laneLinks'][k][
                                                      'endLaneIndex']))
                    incoming_lanes.append(incoming_roads)
                    outgoing_lanes.append(outgoing_roads)

                # add intersection to dict where key = intersection_id
                # value = no of lightPhases, incoming lane names, outgoing lane names, directions for each lane group
                self.intersections[self.roadnetDict['intersections'][i]['id']] = [
                    [len(self.roadnetDict['intersections'][i]['trafficLight']['lightphases'])],
                    incoming_lanes,
                    outgoing_lanes,
                    directions
                ]

        # setup intersectionNames list for agent actions
        self.intersectionNames = []
        for key in self.intersections:
            self.intersectionNames.append(key)

        # define action space MultiDiscrete()
        action_space_array = []
        for key in self.intersections:
            action_space_array.append(self.intersections[key][0][0])
        self.action_space = spaces.MultiDiscrete(action_space_array)

        # define observation space
        observation_space_dict = {}
        for key in self.intersections:
            total_count = 0
            for i in range(len(self.intersections[key][1])):
                total_count += len(self.intersections[key][1][i])

            intersection_observation = []
            max_vehicles = len(self.flowDict)
            for i in range(total_count):
                intersection_observation.append([max_vehicles, max_vehicles])

            observation_space_dict[key] = spaces.MultiDiscrete(intersection_observation)
        self.observation_space = spaces.Dict(observation_space_dict)

        # create cityflow engine
        self.eng = cityflow.Engine(config_path, thread_num=1)

        # Waiting dict for reward function
        self.waiting_vehicles_reward = {}

    def step(self, action):
        # Check that input action size is equal to number of intersections
        if len(action) != len(self.intersectionNames):
            raise Warning('Action length not equal to number of intersections')

        # Set each traffic light phase to specified action
        for i in range(len(self.intersectionNames)):
            self.eng.set_tl_phase(self.intersectionNames[i], action[i])

        # env step
        self.eng.next_step()
        # observation
        self.observation = self._get_observation()

        # reward
        self.reward = self.getReward()
        # Detect if Simulation is finshed for done variable
        self.current_step += 1

        if self.current_step + 1 == self.steps_per_episode:
            self.is_done = True

        # return observation, reward, done, info
        return self.observation, self.reward, self.is_done, {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        if seed is not None:
            self.eng.set_random_seed(seed)
        self.eng.reset(seed=False)
        self.is_done = False
        self.current_step = 0

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def render(self, mode='human'):
        print("Current time: " + self.cityflow.get_current_time())

    def _get_observation(self):
        # observation
        # get arrays of waiting cars on input lane vs waiting cars on output lane for each intersection
        self.lane_waiting_vehicles_dict = self.eng.get_lane_waiting_vehicle_count()
        self.observation = {}
        for key in self.intersections:
            waiting_intersection = []
            for i in range(len(self.intersections[key][1])):
                for j in range(len(self.intersections[key][1][i])):
                    waiting_intersection.append([self.lane_waiting_vehicles_dict[self.intersections[key][1][i][j]],
                                                self.lane_waiting_vehicles_dict[self.intersections[key][2][i][j]]])
            self.observation[key] = waiting_intersection

        return self.observation

    def _get_obs(self):
        # Get information about environment state
    def _get_info(self):
        # Get information about environment state

    def getReward(self):
        reward = []
        self.vehicle_speeds = self.eng.get_vehicle_speed()
        self.lane_vehicles = self.eng.get_lane_vehicles()

        # list of waiting vehicles
        waiting_vehicles = []
        reward = []

        # for intersection in dict retrieve names of waiting vehicles
        for key in self.intersections:
            for i in range(len(self.intersections[key][1])):
                # reward val
                intersection_reward = 0
                for j in range(len(self.intersections[key][1][i])):
                    vehicle = self.lane_vehicles[self.intersections[key][1][i][j]]
                    # if lane is empty continue
                    if len(vehicle) == 0:
                        continue
                    for k in range(len(vehicle)):
                        # If vehicle is waiting check for it in dict
                        if self.vehicle_speeds[vehicle[k]] < 0.1:
                            waiting_vehicles.append(vehicle[k])
                            if vehicle[k] not in self.waiting_vehicles_reward:
                                self.waiting_vehicles_reward[vehicle[k]] = 1
                            else:
                                self.waiting_vehicles_reward[vehicle[k]] += 1
                            # calculate reward for intersection, cap value to -2e+200
                            if intersection_reward > -1e+200:
                                if self.waiting_vehicles_reward[vehicle[k]] < 460:
                                    intersection_reward += -np.exp(self.waiting_vehicles_reward[vehicle[k]])
                                else:
                                    intersection_reward += -1e-200
                            else:
                                intersection_reward = -1e+200
            reward.append([key, intersection_reward])

        waiting_vehicles_remove = []
        for key in self.waiting_vehicles_reward:
            if key in waiting_vehicles:
                continue
            else:
                waiting_vehicles_remove.append(key)

        for item in waiting_vehicles_remove:
            self.waiting_vehicles_reward.pop(item)

        return reward

    def getReward_nonExp(self):
        reward = []
        self.vehicle_speeds = self.eng.get_vehicle_speed()
        self.lane_vehicles = self.eng.get_lane_vehicles()

        # list of waiting vehicles
        waiting_vehicles = []
        reward = []

        # for intersection in dict retrieve names of waiting vehicles
        for key in self.intersections:
            for i in range(len(self.intersections[key][1])):
                # reward val
                intersection_reward = 0
                for j in range(len(self.intersections[key][1][i])):
                    vehicle = self.lane_vehicles[self.intersections[key][1][i][j]]
                    # if lane is empty continue
                    if len(vehicle) == 0:
                        continue
                    for k in range(len(vehicle)):
                        # If vehicle is waiting check for it in dict
                        if self.vehicle_speeds[vehicle[k]] < 0.1:
                            waiting_vehicles.append(vehicle[k])
                            if vehicle[k] not in self.waiting_vehicles_reward:
                                self.waiting_vehicles_reward[vehicle[k]] = 1
                            else:
                                self.waiting_vehicles_reward[vehicle[k]] += 1
                            # calculate reward for intersection, cap value to -2e+200
                            if intersection_reward > -1e+200:
                                intersection_reward += -(self.waiting_vehicles_reward[vehicle[k]])
                            else:
                                intersection_reward = -1e+200
            reward.append([key, intersection_reward])

        waiting_vehicles_remove = []
        for key in self.waiting_vehicles_reward:
            if key in waiting_vehicles:
                continue
            else:
                waiting_vehicles_remove.append(key)

        for item in waiting_vehicles_remove:
            self.waiting_vehicles_reward.pop(item)

        return reward
