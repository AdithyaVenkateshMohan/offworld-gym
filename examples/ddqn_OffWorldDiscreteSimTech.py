#!/usr/bin/env python

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

from offworld_gym import version

__version__     = version.__version__

import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import sys, pdb, time, glob
import numpy as np
from datetime import datetime
import pickle
import h5py

# configure tensorflow and keras
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import gym
import offworld_gym
from offworld_gym.envs.common.channels import Channels

import keras
from keras.models import Model, load_model
from keras.layers import Dense, Activation, Flatten, Conv2D, Input, MaxPooling2D, LeakyReLU, BatchNormalization, Permute
from keras.optimizers import Adam

from kerasrl.agents.dqn import DQNAgent
from kerasrl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from kerasrl.memory import SequentialMemory
from kerasrl.processors import Processor
from kerasrl.callbacks import ModelIntervalCheckpoint, TerminateTrainingOnFileExists, SaveDQNTrainingState, Visualizer

from utils import TB_RL, GetLogPath , convert_image2gray

import pickle
from tempfile import mkdtemp

import os
import numpy as np
import glob, shutil




# define paths
NAME              = 'sim_offworld_monolith-{}'.format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
OFFWORLD_GYM_ROOT = os.environ['OFFWORLD_GYM_ROOT']
LOG_PATH          = '%s/logs/sim' % OFFWORLD_GYM_ROOT
MODEL_PATH        = '%s/models/sim' % OFFWORLD_GYM_ROOT
STATE_PATH        = '/media/caplab/Seagate Backup Plus Drive/Adithya/Off_world/Off_world/sim_agent_state'

if not os.path.exists(LOG_PATH): os.makedirs(LOG_PATH)
if not os.path.exists(MODEL_PATH): os.makedirs(MODEL_PATH)


# create the envronment
env = gym.make('OffWorldMonolithDiscreteSim-v0', channel_type=Channels.RGBD)
env.seed(123)
nb_actions = env.action_space.n
""" currently not used """




class SequentialMemoryReloadable(SequentialMemory):
    def __init__(self, limit, **kwargs):
        super(SequentialMemoryReloadable, self).__init__(limit,**kwargs)
        # self.limit = limit
        # # Do not use deque to implement the memory. This data structure may seem convenient but
        # # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        # self.actions = RingBuffer(limit)
        # self.rewards = RingBuffer(limit)
        # self.terminals = RingBuffer(limit)
        # self.observations = RingBuffer(limit)
    def reload(self, actions, rewards, terminals , observations ,limit_given, grp):
        assert limit_given <= self.limit
        print(type(actions), "is the type I am operating with now")
        length = min (len(list(actions)), limit_given)
        def numpy_toring(ring,n_data):
            ring.length = length
            ring.data[:ring.length] = list(n_data)
            ring.start = 0
            return ring
        def observationcollection_toring(ring,o_data):
            ring.length = length
            ring.start = 0
            i =0
            for o in o_data.keys():
                obs = list(o_data.get(o))
                ring.data[i:i+len(obs)] = obs
                i += len(obs)
            print(i, length)
            assert(i == length)
            return ring
        self.actions = numpy_toring(self.actions,actions)
        self.rewards = numpy_toring(self.rewards,rewards)
        self.terminals = numpy_toring(self.terminals,terminals)
        if observations.shape != (1,):    
            self.observation = numpy_toring(self.observations,observations)
        else:
            print("Appraently the observations were broken to save space so passing observations as grp")
            print(grp.keys())
            observations = grp.get('observations\\observations-collections')
            self.observation = observationcollection_toring(self.observations,observations)
        

    def cvt_2numpyarray(self):
        try:
            obser_num = np.array(self.observations.data[:self.observations.length])
            break_nsave = False
        except MemoryError:
            print("the observation doesn't have enough space so it's gonna be broken and saved")
            obser_num = self.observations.data[:self.observations.length]
            break_nsave = True
        actions_num = np.array(self.actions.data[:self.observations.length])
        rewards_num = np.array(self.rewards.data[:self.observations.length])
        term_num = np.array(self.terminals.data[:self.observations.length])
        return obser_num , actions_num, rewards_num , term_num , break_nsave

class SaveDQNTrainingState_Large(SaveDQNTrainingState):
    """
    Save agent progress, memory and model weights
    """

    def __init__(self, interval, state_path, memory, dqn, snapshot_limit=None):
        super(SaveDQNTrainingState_Large, self).__init__(interval, state_path, memory, dqn, snapshot_limit)

    def save_training_state(self, episode_nr):
        print("\nSaving the state of the agent... please wait")
    
        if not os.path.exists("%s/episode-%d" % (self.state_path, episode_nr)): os.makedirs("%s/episode-%d" % (self.state_path, episode_nr))

        self.dqn.model.save("%s/episode-%d/model.h5" % (self.state_path, episode_nr))

        parametersdump = (episode_nr, self.dqn.step)
        parametersfile = open("%s/episode-%d/parameters.pkl" % (self.state_path, episode_nr), "wb")
        pickle.dump(parametersdump, parametersfile)
        parametersfile.close()

        try:

            self.save_memory_large(episode_nr, file_path="%s/episode-%d/memory.h5" % (self.state_path, episode_nr))

        except MemoryError:
            print("can't save memory failed....")
        # memdump = self.memory

        # memfile = open("%s/episode-%d/memory.pkl" % (self.state_path, episode_nr), "wb")
        # pickle.dump(memdump, memfile)
        # memfile.close()
    
        # remove oldest snapshot if there is a limit
        if self.snapshot_limit is not None and len(glob.glob("%s/episode-*" % (self.state_path))) > self.snapshot_limit:
            oldest_episode = np.min([int(os.path.basename(s).replace("episode-", "")) for s in glob.glob("%s/episode-*" % self.state_path)])
            shutil.rmtree("%s/episode-%d" % (self.state_path, oldest_episode))

        print("State of the agent has been saved.\n")

    def save_memory_large(self, episode_nr , file_path = ''):
        assert file_path != ''
        memdum = self.memory
        observations_n , actions_n, rewards_n , term_n , break_nsave = memdum.cvt_2numpyarray()
        mem_hfile = h5py.File(file_path , 'w')
        memory_g = mem_hfile.create_group('episode-%d' % (episode_nr))
        # print(type(observations_n))
        # print(observations_n.shape)
        memory_g.create_dataset('actions',data=  actions_n )
        if not break_nsave:
            memory_g.create_dataset('observations',data=  observations_n )
        else:
            no_of_break = 10
            print("the we are braking the memory")
            memory_g.create_dataset('observations',data=  np.array([no_of_break]) )
            obs_g = memory_g.create_group('observations\observations-collections')
            len_obs = len(observations_n)
            part_obs = int(len_obs/no_of_break)
            for i in range(no_of_break):
                if i == no_of_break -1:
                    end = len_obs
                else:
                    end = ((i+1)*part_obs)

                data_to_be = np.array(observations_n[i*part_obs:end])
                obs_g.create_dataset('observations'+str(i) , data = data_to_be )
        memory_g.create_dataset('rewards',data=  rewards_n )
        memory_g.create_dataset('terminals',data=  term_n )
        mem_hfile.close()
        del actions_n , observations_n , rewards_n , term_n
    

    

    


# define network architecture
def convBNRELU(input1, filters=8, kernel_size = 5, strides = 1, id1=0, use_batch_norm=False, use_leaky_relu=False, leaky_epsilon=0.1):
    cname = 'conv%dx%d_%d' % (kernel_size, kernel_size, id1)
    bname = 'batch%d' % (id1 + 1) # hard-coded + 1 because the first layer takes batch0
    elu_name = 'elu_%d' % (id1 + 1)
    leaky_relu_name = 'leaky_relu_%d' % (id1 + 1)
    out = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, name = cname)(input1)

    if use_batch_norm == True:
        out = BatchNormalization(name=bname)(out)

    if use_leaky_relu:
        out = LeakyReLU(leaky_epsilon, name=leaky_relu_name)(out)
    else:
        out = Activation('relu')(out)

    return out

def create_network():
    input_image_size = env.observation_space.shape[1:]
    img_input = Input(shape=(240, 320, 8), name='img_input')

    x = img_input
    for i in range(2):
        x = convBNRELU(x, filters=4, kernel_size=5, strides=2, id1=i, use_batch_norm=False, use_leaky_relu=True)
        x = MaxPooling2D((2, 2))(x)
    x = convBNRELU(x, filters=1, kernel_size=5, strides=1, id1=9, use_batch_norm=False, use_leaky_relu=True)
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation('relu')(x)
    x = Dense(16)(x)
    x = Activation('relu')(x)

    output = Dense(nb_actions)(x)
    model = Model(inputs=[img_input], outputs=output)
    print(model.summary())

    return model


# observation processor
class RosbotProcessor(Processor):

    def process_observation(self, observation):

        return observation

    def process_state_batch(self, batch):
        imgs_batch = []
        for exp in batch:
            imgs = []
            configs = []
            for state in exp:
                imgs.append(np.expand_dims(state[0], 0))
                configs.append(np.expand_dims(100, 0))
            imgs_batch.append(np.concatenate(imgs, -1))
        imgs_batch = np.concatenate(imgs_batch, 0)

        return imgs_batch

class RosbotProcessorRGBD_GRAYD(Processor):

    def process_observation(self, observation):
        rbg_img = observation[0][:,:,:3]
        img_gray = np.array([convert_image2gray(rbg_img)]).reshape(240,320,1)
        dep_img = np.array([observation[0][:,:,3]]).reshape(240,320,1)
        assert dep_img.shape == img_gray.shape
        print(dep_img.shape , img_gray.shape)
        observation = np.append(dep_img, img_gray, axis =2)
        observation = np.array([observation])
        return observation

    def process_state_batch(self, batch):
        imgs_batch = []
        for exp in batch:
            imgs = []
            configs = []
            for state in exp:
                imgs.append(np.expand_dims(state[0], 0))
                configs.append(np.expand_dims(100, 0))
            imgs_batch.append(np.concatenate(imgs, -1))
        imgs_batch = np.concatenate(imgs_batch, 0)

        return imgs_batch

def train():

    # waiting for the ROS messages to clear
    time.sleep(5)

    # check whether to resume training
    print("\n====================================================")
    print("the states path is ", STATE_PATH)
    if os.path.exists("%s" % STATE_PATH):
        print("State from the previous run detected. Do you wish to resume learning from the latest available snapshot? (y/n)")
        while True:
            choice = input().lower()
            if choice == 'y':
                last_episode = np.max([int(os.path.basename(s).replace("episode-", "")) for s in glob.glob("%s/episode-*" % STATE_PATH)])
                LAST_STATE_PATH = "%s/episode-%d" % (STATE_PATH, last_episode)
                print("Resuming training from %s" % LAST_STATE_PATH)
                resume_training = True
                break
            elif choice == 'n':
                print("Please remove or move %s and restart this script." % STATE_PATH)
                exit()
            else:
                print("Please answer 'y' or 'n'")

    else:
        print("Nothing to resume. Training a new agent.")
        os.makedirs(STATE_PATH)
        resume_training = False

    print("====================================================\n")

    # agent parameters
    memory_size = 15000
    window_length = 4
    total_nb_steps = 100000
    exploration_anneal_nb_steps = 20000
    max_eps = 0.95
    min_eps = 0.1
    learning_warmup_nb_steps = 50
    target_model_update = 1e-2
    learning_rate = 1e-3
    # callback parameters
    model_checkpoint_interval = 35000 # steps
    verbose_level = 2                # 1 for step interval, 2 for episode interval
    log_interval = 200               # steps
    save_state_interval = 300       # episodes

    processor = RosbotProcessorRGBD_GRAYD()
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), 'eps', max_eps, min_eps, 0.0, exploration_anneal_nb_steps)
    
    model = create_network()
    memory = SequentialMemoryReloadable(limit=memory_size, window_length=window_length)
    # create or load model and memory
    if resume_training:
        # model = load_model("%s/model.h5" % LAST_STATE_PATH)
        # model = create_network()
        try:
            hf = h5py.File("%s/memory.h5" % LAST_STATE_PATH, 'r')
            keys = list(hf.keys())
            grp = hf.get(keys[0])
            observations = grp.get('observations')
            rewards = grp.get('rewards')
            actions = grp.get('actions')
            terminals = grp.get('terminals')
            if (memory_size < len(actions)):
                memory_size = len(actions)
                print("the memory size is changed to", memory_size)
                memory = SequentialMemoryReloadable(limit=memory_size, window_length=window_length)

            memory.reload(actions, rewards , terminals , observations, len(actions), grp)
            hf.close()
        except FileNotFoundError:
            print("the Memory file is not found")
            
        
        

    # create the agent
    dqn = DQNAgent(processor=processor, model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=learning_warmup_nb_steps,
                   enable_dueling_network=True, dueling_type='avg', target_model_update=target_model_update, policy=policy , train_interval = 5)
    dqn.compile(Adam(lr=learning_rate), metrics=['mae'])

    # model snapshot and tensorboard callbacks
    if resume_training:
        dqn.load_weights("%s/model.h5" % LAST_STATE_PATH)
        callback_tb = pickle.load(open("%s/tb_callback.pkl" % STATE_PATH, "rb"))
        (episode_nr, step_nr) = pickle.load(open("%s/parameters.pkl" % LAST_STATE_PATH, "rb"))
    else:
        loggerpath, _ = GetLogPath(path=LOG_PATH, developerTestingFlag=False)
        callback_tb = TB_RL(None, loggerpath)
        tbfile = open("%s/tb_callback.pkl" % STATE_PATH, "wb")
        pickle.dump(callback_tb, tbfile)
        tbfile.close()
        episode_nr = 0
        step_nr = 0

    # other callbacks
    callback_poisonpill = TerminateTrainingOnFileExists(dqn, '/tmp/killrlsim')
    # print("hello making the save state call back", save_state_interval )
    callback_modelinterval = ModelIntervalCheckpoint('%s/dqn_%s_step_{step:02d}.h5f' % (MODEL_PATH, NAME), model_checkpoint_interval, verbose=1)
    callback_save_state = SaveDQNTrainingState_Large(save_state_interval, STATE_PATH, memory, dqn, snapshot_limit=3)
    state_visualizer = Visualizer()
    cbs = [callback_modelinterval, callback_tb, callback_save_state, callback_poisonpill]

    # train the agent
    model = dqn.fit(env, callbacks=cbs, action_repetition=1, nb_steps=total_nb_steps, visualize=False,
                verbose=verbose_level, log_interval=log_interval, resume_episode_nr=episode_nr, resume_step_nr=step_nr)

    save_state = True
    return model ,int(total_nb_steps / 100) , memory , dqn

def save_training_state(episode_nr , memory ,state_path , dqn , snapshot_limit):
        print("\nSaving the state of the agent... please wait")
    
        if not os.path.exists("%s/episode-%d" % (state_path, episode_nr)): os.makedirs("%s/episode-%d" % (state_path, episode_nr))

        memdump = (memory, memory.actions, memory.rewards, memory.terminals, memory.observations)
        memfile = open("%s/episode-%d/memory.pkl" % (state_path, episode_nr), "wb")
        pickle.dump(memdump, memfile)
        memfile.close()

        dqn.model.save("%s/episode-%d/model.h5" % (state_path, episode_nr))

        parametersdump = (episode_nr, dqn.step)
        parametersfile = open("%s/episode-%d/parameters.pkl" % (state_path, episode_nr), "wb")
        pickle.dump(parametersdump, parametersfile)
        parametersfile.close()
    
        # remove oldest snapshot if there is a limit
        if snapshot_limit is not None and len(glob.glob("%s/episode-*" % (state_path))) > snapshot_limit:
            oldest_episode = np.min([int(os.path.basename(s).replace("episode-", "")) for s in glob.glob("%s/episode-*" % state_path)])
            shutil.rmtree("%s/episode-%d" % (state_path, oldest_episode))

        print("State of the agent has been saved.\n")




if __name__ == "__main__":
    save_state = False
    model ,episode_nr, memory , dqn = train()
    # save_training_state(episode_nr , memory ,STATE_PATH , dqn , 3)
    
