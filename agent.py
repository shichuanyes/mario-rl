# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 13:21:46 2024

@author: zhaoy
"""
from args import parse_args
from CNN import *
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C


args = parse_args()
def targetmodel(name,cnn,env):
    policy_kwargs = dict(
        features_extractor_class=cnn,
        features_extractor_kwargs=dict(features_dim=128),
    )
    if args.agent == "DQN":
        model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    elif args.agent == 'PPO':
        model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    elif args.agent == 'A2C':
        model = A2C("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    return model
    
