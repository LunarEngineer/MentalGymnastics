import numpy as np
import torch
import mentalgym
import mentalgym.envs
import mentalgym.functionbank
import agentNets
import gym

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.env_checker import check_env



class MentalAgent:
    def __init__(self, hparams):
        # Initialize RL Hyperparameters
        self.epsilon = hparams["epsilon_start"]
        self.alpha = hparams["alpha_start"]

        # Instantiate environment
        self.env = mentalgym.envs.MentalEnv()

        #  It will check your custom environment and output additional warnings if needed
        check_env(self.env)

        # Create A2C Agent
#        policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[32, 32])
        self.model = A2C("MlpPolicy", self.env, verbose=1,
                         gamma=hparams["gamma"],
                         learning_rate=hparams["alpha_start"])
#                         policy_kwargs)

    def train(self, hparams):
        obs = self.env.reset()
        dones = False
        while not dones:
            action, _states = self.model.predict(obs)
            obs, rewards, dones, info = self.env.step(action)
            self.env.render()


if __name__ == "__main__":
    # Customize training run **HERE**
    hparams = {}
    hparams["num_episodes"] = 2
    hparams["max_steps"] = 10
    hparams["hidden_layers"] = (10,)
    hparams["gamma"] = 0.99
    hparams["alpha_start"] = 0.001
    hparams["alpha_const"] = 2.0
    hparams["alpha_maintain"] = 0.00001
    hparams["epsilon_start"] = 1.0
    hparams["epsilon_const"] = 20.0
    hparams["epsilon_maintain"] = 0.01
    hparams["buffer_len"] = 100
    hparams["minibatch_size"] = 8
    hparams["min_buffer_use_size"] = 5 * hparams["minibatch_size"]
    hparams["num_functions"] = 8
    hparams["num_active_fns_init"] = 3

    agent = MentalAgent(hparams)
    agent.train(hparams)
