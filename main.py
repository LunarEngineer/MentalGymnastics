
# import gym
# import mentalgym
import argparse
import gin
from mentalgym.envs import MentalEnv
from src.agent import MentalAgent, TensorboardCallback
# import numpy as np
import os
# import datetime
from shutil import copyfile

# from stable_baselines3.common import results_plotter


# Create 'results' folder
if not os.path.exists('results'):
    os.makedirs('results')

# # Create timestamped sub-folder
# timestamp = str(datetime.datetime.now())[:-7]
# if not os.path.exists(os.path.join('results', timestamp)):
#     os.makedirs(os.path.join('results', timestamp))
# # Copy config file
# copyfile('config.gin', os.path.join(os.path.join('results', timestamp),'config.gin'))

# Parse 'config.gin' for hyperparameters & env setup
parser = argparse.ArgumentParser(description='Run an experiment given an experiment config file.')
parser.add_argument('--configfile', help='A config.gin file')
args = parser.parse_args()
gin.parse_config_file(args.configfile)

# Instantiate agent
agent = MentalAgent()

callback_ = TensorboardCallback()

# Train the agent
agent.train(log_dir='results', callback=callback_)

#########################
#   Plot/Save Results   #
#########################

# Results are handled by tensorboard
# To view, enter the following in command line
# tensorboard --log_dir = results
# Visit the results page at the specified site