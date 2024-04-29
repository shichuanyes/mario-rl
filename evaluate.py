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

import cv2


args = parse_args()

MODEL_DIR = './model/' + args.agent + '_' + args.cnn + '.zip'
print(MODEL_DIR)


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

# evaluate_policy(model, env, n_eval_episodes=1, deterministic=True, render=False, return_episode_rewards=False)

# episode = 1

# for episode in range(1, episode+1):
#     states = env.reset()
#     done = False
#     score = 0
    
#     while not done:
#         env.render()
#         action, _ = model.predict(states, deterministic=True)
#         states, reward, done, info = env.step(action)
#         score += reward
#         time.sleep(0.01)
#     print('Episode:{} Score:{}'.format(episode, score))
# #env.close()

env = model.get_env()
state = env.reset()
frames = [state]

score = 0
for step in range(5000):
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    score += reward
    frames.append(state)
    if done:
        break
print('Score:{}'.format(score))
env.close()

frames = np.concatenate(frames)
frames = np.swapaxes(frames, 1, 3)
frames = np.swapaxes(frames, 1, 2)

def create_video(frames, output_video_path, fps=30):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()

create_video(frames, 'out.mp4')
