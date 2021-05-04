import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from mentalgym.envs import MentalEnv
from mentalgym.utils.data import testing_df
import gin
import gym
import os

from stable_baselines3.common.callbacks import BaseCallback

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
        
        self.model.tensorboard_log = log_dir
        self.model.learn(
            total_timesteps=self.num_episodes * self.max_steps,
            callback=callback
            )




class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, log_dir, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

        self.function_stats = {}
        self.ep_reward = []
        self.log_dir = log_dir

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        print('\n --------- Hi ----------- \n')
        # stats = self.training_env.env_method('return_statistics')[0]
        # self.function_stats = stats

        # print(stats)

        # self._function_bank

        # Linear, ReLU, Dropout

        # mean by function id
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        self.ep_reward.append(self.training_env.env_method('return_last_reward')[0])
        print(self.ep_reward)

        done = self.training_env.env_method('return_done')[0]
        if done:
            print('D O N E')
            f = open(os.path.join(self.log_dir, 'ep_reward.csv'), "a")  
  
            # writing newline character
            f.write(str(np.sum(self.ep_reward)))
            f.write(',')
            f.close()

            fb_stats = self.training_env.env_method('return_function_bank')[0]
            print(fb_stats)

            self.ep_reward = []
            # Gather all complexities vs score
            # TODO: add the score & complexity to this part of the code

        # else:
            
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

if __name__ == "__main__":
    # Customize training run **HERE**
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
    hparams["num_active_fns_init"] = 3
    hparams["epochs"] = 5
    hparams["net_lr"] = 0.0001
    hparams["net_batch_size"] = 128

    env = MentalEnv(
            dataset=hparams["dataset"],
            number_functions=hparams["number_functions"],
            max_steps=hparams["max_steps"],
            verbose=hparams["verbose"],
            epochs=hparams["epochs"],
            net_lr=hparams["net_lr"],
            net_batch_size=hparams["net_batch_size"],
        )

    agent = MentalAgent(env)
    agent.train()


