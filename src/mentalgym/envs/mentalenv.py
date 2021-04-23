import gym
import logging
import numpy as np
import pandas as pd

from gym import (
    Env,
)

from mentalgym.functionbank import (
    make_function,
    FunctionBank
)
from mentalgym.types import Function, FunctionSet
from mentalgym.utils.data import function_bank
from mentalgym.utils.reward import connection_reward, linear_completion_reward
from mentalgym.utils.spaces import (
    refresh_experiment_container,
    append_to_experiment,
)
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree
from typing import Optional


__FUNCTION_BANK_KWARGS__ = {
    "function_bank_directory",
    "dataset_scraper_function",
    "sampling_function",
    "pruning_function",
}

# Standard logging.
# TODO: Replace print statements.
logger = logging.getLogger(__name__)


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
    max_steps: int = 4
        The maximum number of steps the agent can take in an
        episode.
    seed: Optional[int] = None
        This is used to seed randomness.
    verbose: bool = False
        This can be used to produce verbose logging.
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
        experiment_space_min: ArrayLike = np.array([0.0, 0.0]),
        experiment_space_max: ArrayLike = np.array([100.0, 100.0]),
        number_functions: int = 8,
        max_steps: int = 4,
        seed: Optional[int] = None,
        verbose: bool = False,
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
        # The maximum number of actions the agent
        self.max_steps = max_steps
        # This is used to seed randomness.
        self._seed = seed
        # This is used to grab any parameters for the function bank.
        scrape_kwargs = [
            _ for _ in kwargs
            if _ in __FUNCTION_BANK_KWARGS__
        ]
        self._function_bank_kwargs = {
            _: kwargs.pop(_) for _ in scrape_kwargs
        }
        # This is storing the dimensionality of the experiment
        #   space for convenience
        self.ndim = len(experiment_space_min)
        # These are convenience properties that make subsetting
        #   a little more readable.
        self._loc_fields = [
            f'exp_loc_{_}' for _ in range(self.ndim)
        ]
        self._state_fields = ['i'] + self._loc_fields
        self._verbose = verbose
        ############################################################
        #             Instantiate the Function Space               #
        #                                                          #
        # This is the data structure used by the gym which holds   #
        #   composed functions that have been created over time.   #
        ############################################################
        # self._function_bank = FunctionBank(
        #     modeling_data=dataset,
        #     population_size=number_functions,
        #     **self._function_bank_kwargs
        # )
        self._function_bank = function_bank
        ############################################################
        #            Instantiate the Experiment Space              #
        #                                                          #
        # This is the data structure used by the gym which holds   #
        #   composed functions that the agent has placed onto the  #
        #   canvas. This is built in reset.                        #
        ############################################################
        self.state = self.reset()
        ############################################################
        #           Instantiate the Observation Space              #
        ############################################################
        # The observation space, representing the state, is a
        #   (max_steps + I) x m sized snapshot of the experiment
        #   space. It is filled sequentially as the agent adds nodes
        #   to the experiment. The first I instances of data are
        #   representative of the i / o nodes (the features of the
        #   modeling data and the output node), while the remaining
        #   max_steps instances are composed or intermediate
        #   functions added to the space. The columns represent:
        #
        #   * The integer 'function id' representing the integer
        #       index in the function bank.
        #   * The location to emplace the function (minimum 2d)
        # TODO: Stable baselines recommends this to beflattened, symmetric, and normalized.
        # TODO: If we do that we just need to change the parse.
        self.observation_space = gym.spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape = (1 + self.ndim, self._state_length)
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
        # The shape of the action space is id + num_dim + r
        self._action_size = self.ndim + 2
        self.action_space = gym.spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape=(self._action_size,)
        )
        if self._verbose:
            status_message = f"""Environment Init:
            Episodic Information
            --------------------
            Maximum number of steps in episode: {max_steps}

            Space Information
            -----------------
            Experiment Space:
                Dimensionality: {self.ndim}
                Minimum node locations: {experiment_space_min}
                Maximum node locations: {experiment_space_max}
                Location fields: {self._loc_fields}

            Function Bank:
                Maximum number of living functions: {number_functions}
                Key Word Arguments: {self._function_bank_kwargs}

            State Space:
                Features from experiment space: {self._state_fields}
                State Space Length: {self._state_length}

            Miscellaneous
            -------------
            Random seed: {seed}
            """
            print(status_message)

    def step(
        self,
        action: Optional[ArrayLike] = None
    ) -> ArrayLike:
        """Interprets action and puts a node into experiment space.

        Parameters
        ----------
        action: ArrayLike
            This is a 1 x m array representing an id, location, and
            radius for a new action.

        Returns
        -------
        state: ArrayLike
            This is a (max_steps + I) x m sized array representing
            the experiment space.
        """
        ############################################################
        #                       Bookkeeping                        #
        ############################################################
        # This iterator is used to let the gym know when to terminate.
        self._step += 1
        # This is defaulted to False and updated later on if we snap
        #   the output.
        connected_to_sink = False
        ############################################################
        #                     Function Parsing                     #
        #                                                          #
        # This section is used to parse the output from the agent. #
        ############################################################
        # Parse the function index. This ensures the function index
        #   is in the appropriate range of values.
        # TODO: Uncomment this after function bank implementation.
        action_index = 2 #round(action[0])
        # action_index = np.round(
        #     np.clip(
        #         action[0],
        #         0,
        #         function_bank.idxmax()
        #     )
        # )
        # This extracts the function location from the action.
        # This 'clips' the action location to the interior of the
        #   experiment space. It is already a float array, so nothing
        #   further is required.
        action_location = np.clip(
            action[1:-1],
            self.experiment_space_min,
            self.experiment_space_max
        )
        # This extracts the function radius from the action.
        # This is already a float array, no further parsing required.
        action_radius = action[-1]
        # Verbose logging here for development and troubleshooting.
        if self._verbose:
            debug_message = f"""Action Parse:
            Passed action: {action}
            Parsed action:
            \tIndex: {action_index}
            \tLocation: {action_location}
            \tRadius: {action_radius}
            """
            print(debug_message)
        ############################################################
        #                    Function Building                     #
        #                                                          #
        # This section uses the extracted index, location, and     #
        #   radius to build a representation and add it to the     #
        #   experiment space, dependent on the Function type.      #
        ############################################################
        # Use the action index to query the function bank and get
        #   the Function representation.
        functions: pd.DataFrame = self._function_bank.query(
            "i == @action_index"
        )
        function_set: FunctionSet = functions.to_dict(orient='index')
        # This should never return more than one Function.
        err_msg = f"""Function Error:
        When querying functions with index i == {action_index}
        the result were not a one-element iterable.

        Actual Results
        --------------
        {function_set}
        """
        assert len(function_set) == 1, err_msg
        fun: Function = function_set[0]
        # Pull out the important bits of the Function here.
        # These are the elements that are added to the experiment
        #   space.
        err_msg = f"""Function Error:
        When querying functions to place only composed and atomic
        functions should be returned.

        Actual Type
        -----------
        {fun['type']}
        """
        f_type: str = fun['type']
        assert f_type in ['composed', 'atomic'], err_msg
        # Dependent on what the Function type is, it will be handled
        #   differently.
        if f_type == "composed":
            # If it's a composed Function it is just appended to the
            #   experiment container. TODO: Test
            self._experiment_space = append_to_experiment(
                experiment_space_container = self._experiment_space,
                function_bank = self._function_bank,
                composed_functions = [fun]
            )
        else:
            # If it's an atomic Function it is added as an
            #   'intermediate' and then constructed appropriately
            #   in the build_net.
            action_radius = 200 # TODO: Testing data, remove later.
            # Build a KD tree from the locations of the nodes in the
            #   experiment space.
            tree = cKDTree(
                self._experiment_space[
                    self._loc_fields
                ].values
            )
            # idx = tree.query_ball_point((action[1], action[2]), action[3])
            # Query the KD Tree for all points within the radius.
            idx = tree.query_ball_point(
                action_location,
                action_radius
            )
            # If any indices are returned it's a valid action
            if len(idx):
                # This uses the returned indices to subset the
                #   experiment space. This can contain input,
                #   intermediate, and composed nodes, in addition
                #   to output. This checks for output, removes it,
                input_df = self._experiment_space.iloc[idx]
                output_df = input_df.query('type == "sink"')
                # If the output was connected, then we will trigger
                #   completion and build and run the net.
                if output_df.shape[0]:
                    connected_to_sink = True
                input_df = input_df.query('type != "sink"')
                # TODO:
                # This might need to be reworked.
                # What does this functionality need to do?
                # This entire remaining chunk simply needs to add an
                #   intermediate function to experiment space while
                #   retaining the *signature* of the atomic function.
                # The signature (Callable) can be used to construct
                #   the function when called in the baking function
                #   _build_net.
                # Should intermediate functions *care* about the index?
                # No. I do not believe they should. The only use of the
                #   index is for subsetting the function bank;
                #   intermediate functions will never exist in the
                #   function bank. Input / Output functions should get
                #   indices of -1, while atomic get -2? Open for discussion,
                #   we simply need a method to distinguish them. We can
                #   keep the callables in the experiment space, or abstract
                #   them to a separate data structure. Advantages and disadvantages either way.
                built_function = make_function(
                    lambda x: x,
                    "intermediate",
                    input_df.id.to_list(),
                    action_location
                )
                new_function = {
                    k: v for k, v in built_function.items()
                    if k in ["i", "id", "type", "input"]
                }
                new_new_function["exp_loc_0"] = action[1]
                new_new_function["exp_loc_1"] = action[2]
                new_new_function["i"] = 8 # make this increment, not hard coded
                self._function_bank = self._function_bank.append(
                    new_new_function,
                    ignore_index=True
                )
        if self._verbose:
            debug_message = f"""Function Build:
            Queried Function:\n{fun}
            Queried Function Type: {fun['type']}
            
            New Experiment Space
            --------------------\n{self._experiment_space}
            """
            print(debug_message)
        ############################################################
        #                    Reward and Rollup                     #
        #                                                          #
        # This uses the experiment space to determine appropriate  #
        #   rewards and whether or not to build and run the graph. #
        ############################################################
        # Return a minor reward if there are *any* nodes added.
        reward = connection_reward(
            self._experiment_space,
            self._function_bank
        )
        # Check to see if it's time to call it a day.
        done = connected_to_sink or (self._step >= self.max_steps)
        if done:
            # TODO: Bake the net.
            self._build_net()
            # Add the completion reward.
            reward += float(linear_completion_reward(
                self._experiment_space, None, 0.5
            ))

        # Default values here, or pass some info?
        info = {}
        # Extract the state space.
        state = self.build_state()
        if self._verbose:
            debug_message = f"""End of Step:
            Connected to the sink node: {connected_to_sink}
            Hit maximum steps: {self._step >= self.max_steps}
            Total Reward: {reward}
            Done: {done}

            State Observation
            -----------------{state}
            """
            print(debug_message)
        return state, reward, done, info

    def build_state(self) -> ArrayLike:
        """Builds an observation from experiment space."""
        _exp_state = self._experiment_space[
            self._state_fields
        ].values
        _pad_state = np.zeros(
            (
                self._state_length - _exp_state.shape[0]
                , 1 + self.ndim
            )
        )
        return np.concatenate([_exp_state, _pad_state]).T

    def _build_net(self):
        pass

    def reset(self):
        """Initialize state space.

        This creates an empty canvas for the experiment space
        consisting of nothing but input and output nodes.
        """
        # Reset the step counter
        self._step = 0
        # Fill the experiment space.
        self._experiment_space = refresh_experiment_container(
            function_bank=function_bank,
            min_loc=self.experiment_space_min,
            max_loc=self.experiment_space_max,
        )
        # Get the number of input and output.
        n_io = self._experiment_space.query(
            'type in ["source", "sink"]'
        ).shape[0]
        # Set the state length.
        self._state_length = n_io + self.max_steps
        # Then build the state.
        state = self.build_state()
        if self._verbose:
            debug_message = f"""Environment Reset:
            Current Step: {self._step}

            Current Experiment Space
            ------------------------\n{self._experiment_space}

            Current Observation Space
            -------------------------\n{state}
            """
            print(debug_message)
        return state

    def render(self, mode="human"):
        pass

    def close(self):
        pass
