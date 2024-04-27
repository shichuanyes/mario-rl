# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 13:40:18 2024

@author: zhaoy
"""

import torch as th
import torch.nn as nn
from gymnasium import spaces
from args import parse_args
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision.models import resnet18, vgg16, inception_v3



class Baseline(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
    
    
    
class AlexNet(BaseFeaturesExtractor):
    """
    A custom CNN feature extractor using an AlexNet-like architecture.
    
    :param observation_space: (gym.Space) The space of the input observations.
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        # Define a smaller AlexNet-like architecture suitable for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        # Define the final linear layers
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Forward pass through the network.

        :param observations: (th.Tensor) The observations from the environment.
        :return: (th.Tensor) The extracted features.
        """
        features = self.cnn(observations)
        return self.linear(features)
    
    
class ResNet(BaseFeaturesExtractor):
    """
    A custom CNN feature extractor using a ResNet-like architecture.
    
    :param observation_space: (gym.Space) The space of the input observations.
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        # Load a pre-defined ResNet and modify it for our use case
        base_model = resnet18(pretrained=False)  # Use pretrained=True for pretrained weights
        base_model.conv1 = nn.Conv2d(n_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        base_model.fc = nn.Linear(base_model.fc.in_features, features_dim)

        self.resnet = base_model
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Forward pass through the network.

        :param observations: (th.Tensor) The observations from the environment.
        :return: (th.Tensor) The extracted features.
        """
        return self.resnet(observations)


class VGG(BaseFeaturesExtractor):
    """
    A custom CNN feature extractor based on VGG16 architecture.
    
    :param observation_space: (gym.Space) The space of the input observations.
    :param features_dim: (int) Number of features extracted.
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        # Load a pre-defined VGG16 model and modify it for our use case
        base_model = vgg16(pretrained=False)  # Use pretrained=True for pretrained weights
        base_model.features[0] = nn.Conv2d(n_input_channels, 64, kernel_size=3, padding=1)
        base_model.classifier[6] = nn.Linear(4096, features_dim)

        self.vgg = base_model
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Forward pass through the network.
        
        :param observations: (th.Tensor) The observations from the environment.
        :return: (th.Tensor) The extracted features.
        """
        return self.vgg(observations)
    

class Inception(BaseFeaturesExtractor):
    """
    A custom CNN feature extractor based on Inception v3 architecture.
    
    :param observation_space: (gym.Space) The space of the input observations.
    :param features_dim: (int) Number of features extracted.
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        # Load a pre-defined Inception v3 model and modify it for our use case
        base_model = inception_v3(pretrained=False, aux_logits=False)
        # Adjust the first convolution layer to match input channels
        base_model.Conv2d_1a_3x3.conv = nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=0, bias=False)
        # Modify the final fully connected layer
        base_model.fc = nn.Linear(2048, features_dim)

        self.inception = base_model
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Forward pass through the network.
        
        :param observations: (th.Tensor) The observations from the environment.
        :return: (th.Tensor) The extracted features.
        """
        return self.inception(observations)