from args import parse_args
from agent import *
from preprocessing import *
from CNN import *
from stable_baselines3 import DQN
import sys
import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros


from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from stable_baselines3.common.evaluation import evaluate_policy
import time

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C


args = parse_args()

MODEL_DIR = './model/' + args.agent + args.cnn


env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode='rgb_array', apply_api_compatibility=True)
# Override `reset` method to get rid of error
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
env = JoypadSpace(env, [["right"], ["right", "A"], ['right', 'A', 'B']])

# Preprocessing
if args.skip_frame_num > 0:
    env = SkipFrame(env, skip=args.skip_frame_num)
    
if args.gray == 'True':
    env = GrayScaleObservation(env)
    
if args.resize > 0:
    env = ResizeObservation(env, shape=args.resize)
    
if args.agent == "DQN":
    model = DQN.load(MODEL_DIR, env = env)
elif args.agent == 'PPO':
    model = PPO.load(MODEL_DIR, env = env)
elif args.agent == 'A2C':
    model = A2C.load(MODEL_DIR, env = env)

episode = 1

for episode in range(1, episode+1):
    states = env_wrap.reset()
    done = False
    score = 0
    
    while not done:
        env_wrap.render()
        action, _ = model.predict(states, deterministic=True)
        states, reward, done, info = env_wrap.step(action)
        score += reward
        time.sleep(0.01)
    print('Episode:{} Score:{}'.format(episode, score))
#env.close()