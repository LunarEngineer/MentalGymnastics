import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from mentalgym.envs import MentalEnv
from mentalgym.utils.data import testing_df


class MentalAgent:
    def __init__(self, hparams):
        """Initialize the RL agent, including setting up its environment"""

        # Instantiate environment
        self.env = MentalEnv(
            testing_df,
            number_functions=hparams["number_functions"],
            max_steps=hparams["max_steps"],
            verbose=hparams["verbose"],
            epochs=hparams["epochs"],
            net_lr=hparams["net_lr"],
            net_batch_size=hparams["net_batch_size"],
        )

        self.num_episodes = hparams["num_episodes"]
        self.max_steps = hparams["max_steps"]

        #  Check custom environment and output additional warnings if needed
        if hparams["verbose"]:
            check_env(self.env)

        # Create A2C Agent
        #        policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[32, 32])
        self.model = A2C(
            "MlpPolicy",
            self.env,
            learning_rate=hparams["alpha_start"],
            n_steps=1,
            gamma=hparams["gamma"],
            verbose=hparams["verbose"],
        )

    #                         policy_kwargs)

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
