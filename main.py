
import gym
import mentalgym
import gin
from mentalgym.envs import MentalEnv
from src.agent import MentalAgent, CustomCallback, TensorboardCallback
import numpy as np
import os
import datetime
from shutil import copyfile

from stable_baselines3.common import results_plotter


# Create 'results' folder
if not os.path.exists('results'):
    os.makedirs('results')

# Create timestamped sub-folder
timestamp = str(datetime.datetime.now())[:-7]
if not os.path.exists(os.path.join('results', timestamp)):
    os.makedirs(os.path.join('results', timestamp))

# Copy config file
copyfile('config.gin', os.path.join(os.path.join('results', timestamp),'config.gin'))

# Parse 'config.gin' for hyperparameters & env setup
gin.parse_config_file('config.gin')

# Instantiate agent
agent = MentalAgent()

# callback_ = CustomCallback(log_dir=os.path.join('results', timestamp), n_episodes=agent.num_episodes)
callback_ = TensorboardCallback()

# Train the agent
agent.train(log_dir=os.path.join('results', timestamp), callback=callback_)
# agent.train(log_dir=os.path.join('results', timestamp))

#########################
#   Plot/Save Results   #
#########################
