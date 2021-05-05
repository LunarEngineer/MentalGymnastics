import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.datasets import make_classification
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from mentalgym.envs import MentalEnv
from mentalgym.utils.data import testing_df, make_sk2c, make_dataset
import gin
import gym
import os

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

@gin.configurable
class MentalAgent:
    def __init__(
        self, 
        env: gym.Env,
        num_episodes: int = 1,
        hidden_layers: tuple = (10,),
        gamma: float = 0.99,
        alpha_start: float = 0.001,
        alpha_const: float = 2.0,
        alpha_maintain: float = 0.00001,
        epsilon_start: float = 1.0,
        epsilon_const: float = 20.0,
        epsilon_maintain: float = 0.01,
        buffer_len: int = 100,
        num_active_fns_init: int = 3,
        verbose: bool = False
    ):
        """Initialize the RL agent, including setting up its environment"""

        # Instantiate environment
        self.env = env

        self.num_episodes = num_episodes
        self.max_steps = self.env.max_steps

        #  Check custom environment and output additional warnings if needed
        if verbose:
            check_env(self.env)

        # Create A2C Agent
        # policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[32, 32])
        self.model = A2C(
                "MlpPolicy",
                self.env,
                learning_rate=alpha_start,
                n_steps=1,
                gamma=gamma,
                verbose=1
            )
        #   policy_kwargs)

    def train(
            self,
            log_dir: str = None,
            callback = None
        ):
        """Train the RL agent.

        self.model.n_steps: number of env steps per update
        total_timesteps: number of times the agent will update, which should
            be roughly self.num_episodes * self.max_steps
        environment steps needed to achieve the total_timesteps:
            total_timesteps * self.model.n_steps"""
        
        if log_dir == None or callback == None:
            self.model.learn(total_timesteps=self.num_episodes * self.max_steps)
        else:
            self.model.tensorboard_log = log_dir
            self.model.learn(
                total_timesteps=self.num_episodes * self.max_steps,
                callback=callback, 
                log_interval=self.max_steps
                )

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_training_start(self):
        self._log_freq = 1000  # log every 1000 calls

        output_formats = self.logger.Logger.CURRENT.output_formats
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:

        max_steps = self.training_env.get_attr('max_steps')[0]
        if self.num_timesteps % max_steps == 0 and self.num_timesteps != 0:
            stats = self.training_env.env_method('return_statistics')[0]
            n_classes = self.training_env.get_attr('n_classes')[0]
            
            # TODO: un-hardcode this
            if n_classes == 2:
                functions = stats.iloc[101+3:]    # SK2C
            else:
                functions = stats.iloc[785+3:]    # MNIST

            # print('\nREWARD MEAN\n', functions.tail(1).score_reward_mean.item())
            # print('\nACC MEAN\n', functions.tail(1).score_accuracy_mean.item())
            # print('\nCOMPLEXITY MEAN\n', functions.tail(1).score_complexity_mean.item())

            self.tb_formatter.writer.add_scalars(f'complexity & acc', {
                                                  'mean_complexity': functions.tail(1).score_complexity_mean.item(),
                                                  'mean_acc': functions.tail(1).score_accuracy_mean.item(),
                                                }, self.num_timesteps)
            self.tb_formatter.writer.add_scalars(f'mean reward', {
                                                  'mean_reward': functions.tail(1).score_reward_mean.item()
                                                }, self.num_timesteps)
            self.tb_formatter.writer.flush()

        # print('NUM Timesteps\n', self.num_timesteps, max_steps)
        
        return True


if __name__ == "__main__":
    # Customize training run **HERE**
    hparams = {}
    hparams["dataset"] = "MNIST"
    hparams["verbose"] = 0
    hparams["experiment_folder"] = 'experiment_one'
    hparams["num_episodes"] = 3
    hparams["number_functions"] = 8
    hparams["max_steps"] = 5
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
    hparams["num_active_fns_init"] = 3
    hparams["epochs"] = 1
    hparams["net_lr"] = 1e-2
    hparams["net_batch_size"] = 512

    if hparams["dataset"] == "MNIST":
        set_list = make_dataset('MNIST')
    else:                                # hparams["dataset"] == "SK2C"
        set_list = make_sk2c()

    kwargs = {"force_refresh": False}

    env = MentalEnv(
            set_list=set_list,
            number_functions=hparams["number_functions"],
            max_steps=hparams["max_steps"],
            verbose=hparams["verbose"],
            epochs=hparams["epochs"],
            net_lr=hparams["net_lr"],
            net_batch_size=hparams["net_batch_size"],
            function_bank_directory=hparams["experiment_folder"],
            **kwargs,
        )

    agent = MentalAgent(env, num_episodes = hparams["num_episodes"])
    agent.train()


