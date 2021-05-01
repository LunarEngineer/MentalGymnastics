import mentalgym
import gym

from mentalgym.utils.data import testing_df
from src.agent import MentalAgent

###################
# Hyperparameters #
###################

hparams = {}
hparams["dataset"] = testing_df
hparams["verbose"] = 0
hparams["num_episodes"] = 1
hparams["number_functions"] = 8
hparams["max_steps"] = 4
hparams["seed"] = None
hparams["hidden_layers"] = (10,)
hparams["gamma"] = 0.99
hparams["alpha_start"] = 0.001
hparams["alpha_const"] = 2.0
hparams["alpha_maintain"] = 0.00001
hparams["epsilon_start"] = 1.0
hparams["epsilon_const"] = 20.0
hparams["epsilon_maintain"] = 0.01
hparams["buffer_len"] = 100
hparams["num_functions"] = 8
hparams["num_active_fns_init"] = 3
hparams["epochs"] = 5
hparams["net_lr"] = 0.0001
hparams["net_batch_size"] = 128


agent = MentalAgent(hparams)
agent.train()


#####################
# Plot/Save Results #
#####################
