import gym
import numpy as np
import pandas as pd

from gym import (
    # error,
    spaces,
    # utils
    Env
)
from mentalgym.functionbank import FunctionBank
from mentalgym.utils.data import function_bank
from mentalgym.utils.spaces import refresh_experiment_container
from mentalgym.utils.reward import (
    connection_reward,
    linear_completion_reward
)
from numpy.typing import ArrayLike
from typing import Optional
# from gym.utils import seeding
# from typing import Tuple, Any


__FUNCTION_BANK_KWARGS__ = {
    'function_bank_directory',
    'dataset_scraper_function',
    'sampling_function',
    'pruning_function'
}


class MentalEnv(Env):
    """A Mental Gymnasium Environment.

    This class allows a reinforcement learning agent to learn to
    'paint by numbers' with composed functions. It can select
    functions from the palette to drop onto the experiment.

    Parameters
    ----------
    dataset: pd.DataFrame
        This is a modeling dataset.
    experiment_space_min: ArrayLike = numpy.array([0., 0.])
        This is an array of numbers which represents the minimum
        coordinates for functions.
    experiment_space_max: ArrayLike = numpy.array([100., 100.])
        This is an array of numbers which represents the maximum
        coordinates for functions.
    number_functions: int = 8
        This is the number of 'living' functions that the function
        bank will maintain. This does not cap the number of functions
        in the bank, it merely places a constraint on the number
        of functions which can be sampled.
    max_steps: int = 5
        The maximum number of steps the agent can take in an
        episode.
    seed: Optional[int] = None
        This is used to seed randomness.
    **kwargs: Any
        Optional keyword arguments. Current values allowed:
        * function_bank_directory: The directory where the function
            bank will instantiate and look for a function manifest.
        * dataset_scraper_function: The function which creates
            experiment nodes from a modeling dataset.
        * sampling_function: A function which samples the function
            bank to return Functions. Please see mentalgym.utils.sampling
            for more information.
        * pruning_function: A function which prunes the function
            bank to return Functions. Please see mentalgym.utils.spaces
            for more information.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        dataset: pd.DataFrame,
        experiment_space_min: ArrayLike = np.array([0., 0.]),
        experiment_space_max: ArrayLike = np.array([100., 100.]),
        number_functions: int = 8,
        max_steps: int = 5,
        seed: Optional[int] = None,
        **kwargs
    ):
        """Sets up the gym environment.

        This instantiates a function bank and an experiment space.
        """
        super(MentalEnv, self).__init__()
        ############################################################
        #                 Store Hyperparameters                    #
        ############################################################
        # These are used in building the experiment space and when
        #   validating agent actions.
        self.experiment_space_min = experiment_space_min
        self.experiment_space_max = experiment_space_max
        # This is used in the function bank.
        self.number_functions = number_functions
        # TODO: I don't think this is necessary; the function bank should extend arbitrarily for all scoring metrics.
        self.number_metrics = 2
        # The maximum number of actions the agent 
        self.max_steps = max_steps
        # The current step is 0
        self._step = 0
        # TODO: I don't know if this is necessary.
        # Do we need to enforce this, or should we simply cast any
        #   radius to positive?
        self.radius_min = 1
        self.radius_max = self.experiment_space_max - self.experiment_space_min
        # This is used to seed randomness.
        self._seed = seed
        # This is used to grab any parameters for the function bank.
        scrape_kwargs = [
            _ for _ in kwargs
            if _ in __FUNCTION_BANK_KWARGS__
        ]
        self._function_bank_kwargs = {
            _: kwargs.pop(_) for _
            in scrape_kwargs
        }
        ############################################################
        #             Instantiate the Function Space               #
        #                                                          #
        # This is the data structure used by the gym which holds   #
        #   composed functions that have been created over time.   #
        ############################################################
        self._function_bank = FunctionBank(
            modeling_data = dataset,
            population_size = number_functions,
            **self._function_bank_kwargs
        )
        ############################################################
        #            Instantiate the Experiment Space              #
        #                                                          #
        # This is the data structure used by the gym which holds   #
        #   composed functions that the agent has placed onto the  #
        #   canvas. This is built in reset.                        #
        ############################################################
        self.state = self.reset()
        print(self._experiment_space)
        ############################################################
        #           Instantiate the Observation Space              #
        ############################################################
        # The observation space, representing the state, is a
        #   (max_steps + I) x m sized snapshot of the experiment
        #   space. It is filled sequentially as the agent adds nodes
        #   to the experiment. The first I instances of data are
        #   representative of the input nodes (the features of the
        #   modeling data), while the remaining max_steps instances
        #   are composed or intermediate functions added to the
        #   space. The columns represent:
        #
        #   * The integer 'function id' representing the integer
        #       index in the function bank.
        #   * The location to emplace the function (minimum 2d)
        #   * Whether or not the function is 'connected'
        #   TODO: Since we're auto-connecting at completion and we are not adding functions which do not connect, this is unnecessary.
        #   TODO: We should remove the connection state element.
        self.observation_space = gym.spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape = (4, self.max_steps)
        )
        ############################################################
        #              Instantiate the Action Space                #
        ############################################################
        # The action space is a 1 x m array representing:
        #   * The integer 'function id' representing the integer
        #       index in the function bank.
        #   * The location to emplace the function (minimum 2d)
        #   * The radius to use to connect to functions in the
        #       experiment space.
        # A two-dimensional experiment space will look like:
        # [id, x, y, r]
        # Action validation should ensure that the location is in the
        #   bounds of the experiment space and that the radius is
        #   non-negative.
        # The shape of the action space is id + num_dim + r
        self._action_size = len(experiment_space_min) + 2
        self.action_space = gym.spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape = (self._action_size,)
        )

    def step(
        self,
        action: Optional[ArrayLike] = None
    ):
        self._step += 1
        connected_to_sink = False
        done = connected_to_sink or self._step >= self.max_steps
        # This is temporary
        if action is None:
            action = np.random.random(self._action_size)

        # get all source and sink nodes from the function bank
        # get the function representation from the function bank

        # determine if that is a atomic or composed function
        # if atomic
        #   determine all input notes using radius and other nodes locations
        # else
        #   just recreate as it was placed

        # When building the net, don't hook on output layer until done
        # net is build at this point

        # if reached maximum steps (when done) or if connected to the sink
        #   run and measure

        # calculate reward based on current observation space and


        # TODO: Check if new function connects to ANY node
        # check if the new function, connects to other actions that have been dropped
        # 
        # switch(action["type"]):
        #     case "composed": 
        #         # radius doesn't matter, it is put into place
        #         # add to experiment space
        #     case "atomic":
        #         # radius does matter
        #         # make tree from experiment space location query(action.r)
        #         """
        #         >>> from scipy.spatial import cKDTree

        #         >>> points_ref = np.array([(1, 1), (3, 3), (4, 4), (5, 4), (6, 6)])
        #         >>> tree = cKDTree(points_ref)

        #         >>> idx = tree.query_ball_point((4, 4), action.r)
        #         >>> points_ref[idx]
        #         # array([[3, 3], [4, 4], [5, 4]])
        #         """
        #         # if it is a good action (that connects)
        #         #   go get the actual atomic function and instantiate it
        #         #   add to experiment space, not the function bank YET.
        
        # # reward = connection_reward()

        # print("Action:", action)
        # self._step += 1

        # if self._step == self.max_steps:
        #     # run _build_net()
        #     # reward = linear_completion_reward()
        #     done = True

        info = {}
        
        reward = 0

        # self.state: all the function ids, all the locations, connection

        return self.state, reward, done, info

    def _build_net(self):
        pass

    def reset(self):
        self._step = 0
        self._experiment_space = refresh_experiment_container(
            function_bank = function_bank,
            min_loc = self.experiment_space_min,
            max_loc = self.experiment_space_max
        )
        self.function_connection = False
        # TODO: Initialize state to Input (get ID for this input to query in .step())
        state = np.zeros((4, self.max_steps))
        return state

    def render(self, mode="human"):
        pass

    def close(self):
        pass
