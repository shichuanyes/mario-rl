from args import parse_args

from preprocessing import *


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




# Get Arguments
args = parse_args()
seed = args.seed

# Preprocessing

if args.skip_frame_num > 0:
    env = SkipFrame(env, skip=args.skip_frame_num)
    
elif args.gray is True:
    env = GrayScaleObservation(env)
    
elif args.resize > 0:
    env = ResizeObservation(env, shape=args.resize)

elif args.stack_frame_num > 0:
    if gym.__version__ < '0.26':
        env = FrameStack(env, num_stack=args.stack_frame_num, new_step_api=True)
    else:
        env = FrameStack(env, num_stack=args.stack_frame_num)
        

