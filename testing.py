import gym
from gym import spaces
import numpy as np
import cityflow
import json


# create cityflow engine
config_path = "examples/config2.json"
eng = cityflow.Engine(config_path, thread_num=1)

for i in range(1000):
    eng.next_step()
    print(eng.get_lane_waiting_vehicle_count())