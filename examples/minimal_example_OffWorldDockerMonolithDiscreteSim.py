# !/usr/bin/env python

# Copyright 2019 OffWorld Inc.
# Doing business as Off-World AI, Inc. in California.
# All rights reserved.
#
# Licensed under GNU General Public License v3.0 (the "License")
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law, any source code or other materials
# distributed under the License is distributed on an "AS IS" basis,
# without warranties or conditions of any kind, express or implied.

import gym
import logging

import offworld_gym
from offworld_gym.envs.common.channels import Channels
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)

# create the environment
env = gym.make("OffWorldMonolithDiscreteSim-v0", channel_type=Channels.RGB_ONLY)
env.seed(42)

# logging.info(f"action space: {env.action_space} observation_space: {env.observation_space}")
while True:
    env.reset()
    done = False
    get_proper = False
    while not done:
        while not get_proper:
            sampled_action = int(input("give an action"))
            if sampled_action in [0,1,2,3]:
                get_proper = True
            else:
                print("invalid action select between 0 1 2 3")
        get_proper = False 
        env.render()
        obs, rew, done, info = env.step(sampled_action)

        print("the observation", obs , obs.shape , "the current reward", rew , "sample action", sampled_action)
        plt.imshow(obs[0])
        plt.show()
        print("showing the plot")