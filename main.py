
# import gym
# import mentalgym
import argparse
import gin
from mentalgym.envs import MentalEnv
from src.agent import MentalAgent, TensorboardCallback, SaveOnBestTrainingRewardCallback
from stable_baselines3.common.callbacks import CallbackList
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
parser.add_argument('--experiment', help='A config.gin file experiment number')
args = parser.parse_args()
arg_file = os.path.join("experiment_configs", f"experiment_{args.experiment}.gin")
gin.parse_config_file(arg_file)

# Instantiate agent
agent = MentalAgent()


callback_tensorboard = TensorboardCallback(log_freq=agent.max_steps)
callback_save = SaveOnBestTrainingRewardCallback(check_freq=agent.max_steps*2, log_dir=agent.model_path)
call_back_list = CallbackList([callback_tensorboard, callback_save])

# Train the agent
agent.train(log_dir=f'results_{args.experiment}', callback=call_back_list)

###########################
#       Plot Results      #
###########################

# Results are handled by tensorboard
# To view, enter the following in command line
# tensorboard --log_dir = results
# Visit the results page at the specified site