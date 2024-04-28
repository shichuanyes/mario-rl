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

if __name__ == '__main__':
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    
    # Get Arguments
    args = parse_args()

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

    if args.stack_frame_num > 0:
        if gym.__version__ < '0.26':
            env = FrameStack(env, num_stack=args.stack_frame_num, new_step_api=True)
        else:
            env = FrameStack(env, num_stack=args.stack_frame_num)
    cnn=getattr(sys.modules[__name__], args.cnn)
    
    model = targetmodel(args.agent, cnn, env)
    
    # Track time
    start_time = time.time()
    
    # Training
    
    model.learn(total_timesteps=args.total_timesteps)
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Save the trained model
    model.save(args.model_save_path)
    

    print("Training time:", datetime.timedelta(seconds=training_time))
    
    
    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
    print(f"Mean reward over 10 evaluation episodes: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    with open("result.txt", "a") as f:
        f.write(f"{args.agent}, {args.cnn}, {args.skip_frame_num}, {args.stack_frame_num}, {args.resize}, {args.gray}, {args.total_timesteps}, {training_time}, {mean_reward}, {std_reward}\n")
    
    