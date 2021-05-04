import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.datasets import make_classification
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from mentalgym.envs import MentalEnv
from mentalgym.utils.data import testing_df, make_sk2c
import gin
import gym

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
            verbose=verbose,
        )
        #   policy_kwargs)

    def train(self):
        """Train the RL agent.

        self.model.n_steps: number of env steps per update
        total_timesteps: number of times the agent will update, which should
            be roughly self.num_episodes * self.max_steps
        environment steps needed to achieve the total_timesteps:
            total_timesteps * self.model.n_steps"""

        self.model.learn(total_timesteps=self.num_episodes * self.max_steps)


if __name__ == "__main__":
    # Customize training run **HERE**
    hparams = {}
    hparams["dataset"] = "SK2C"
    hparams["verbose"] = 0
    hparams["num_episodes"] = 500
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
        (Xtrain, ytrain), (Xtest, ytest) = tf.keras.datasets.mnist.load_data()
        Xtrain = (Xtrain - np.mean(Xtrain, axis=0)) / (np.std(Xtrain) + 1e-7)
        Xtest = (Xtest - np.mean(Xtest, axis=0)) / (np.std(Xtest) + 1e-7)
        Xtrain, Xval = Xtrain[0:5000], Xtrain[50000:55000]
        ytrain, yval = ytrain[0:5000], ytrain[50000:55000]
        Xtrain = Xtrain.reshape((Xtrain.shape[0], -1))
        Xval = Xval.reshape((Xval.shape[0], -1))
        Xtest = Xtest.reshape((Xtest.shape[0], -1))
        dataset = pd.DataFrame(Xtrain).assign(output=ytrain)
        valset = pd.DataFrame(Xval).assign(output=yval)
        testset = pd.DataFrame(Xtest)
        hparams["n_classes"] = 10
    elif hparams["dataset"] == "SK2C":
        dataset, valset, testset = make_sk2c()
        hparams["n_classes"] = 2
    else:
        dataset = testing_df
        hparams["n_classes"] = 2 #TODO: Need to bring in from data.py

    env = MentalEnv(
            dataset=dataset,
            valset=valset,
            testset=testset,
            number_functions=hparams["number_functions"],
            max_steps=hparams["max_steps"],
            verbose=hparams["verbose"],
            epochs=hparams["epochs"],
            net_lr=hparams["net_lr"],
            net_batch_size=hparams["net_batch_size"],
            n_classes=hparams["n_classes"]
        )

    agent = MentalAgent(env, num_episodes = hparams["num_episodes"])
    agent.train()


