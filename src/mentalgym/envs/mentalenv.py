import logging
from typing import Optional
import numpy as np
from copy import copy, deepcopy
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree
import gym
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
import pandas as pd
from mentalgym.functionbank import FunctionBank
from mentalgym.types import Function, FunctionSet
from mentalgym.utils.function import make_function
from mentalgym.functions.composed import ComposedFunction
from mentalgym.utils.reward import connection_reward, linear_completion_reward
from mentalgym.utils.spaces import (
    refresh_experiment_container,
    append_to_experiment,
)
from mentalgym.functions.atomic import Linear, ReLU, Dropout
from mentalgym.constants import (
    experiment_space_fields,
    linear_i,
    relu_i,
    dropout_i,
    linear_output_size,
    dropout_p,
)


# Customize pandas DataFrame display width
pd.set_option("display.expand_frame_repr", False)

__FUNCTION_BANK_KWARGS__ = {
    "function_bank_directory",
    "dataset_scraper_function",
    "sampling_function",
    "pruning_function",
}

# Standard logging.
# TODO: Replace print statements.
logger = logging.getLogger(__name__)


class MentalEnv(gym.Env):
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
    epochs: int = 5
    net_lr: float = 0.0001
    net_batch_size: int = 128
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
        epochs: int = 5,
        net_lr: float = 0.0001,
        net_batch_size: int = 128,
        seed: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Sets up the gym environment.

        This instantiates a function bank and an experiment space.
        """
        super(MentalEnv, self).__init__()

        dataset.columns = [str(_) for _ in dataset.columns]
        self.dataset = dataset
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
        scrape_kwargs = [_ for _ in kwargs if _ in __FUNCTION_BANK_KWARGS__]
        self._function_bank_kwargs = {_: kwargs.pop(_) for _ in scrape_kwargs}
        # This is storing the dimensionality of the experiment
        #   space for convenience
        self.ndim = len(experiment_space_min)
        # These are convenience properties that make subsetting
        #   a little more readable.
        self._loc_fields = [f"exp_loc_{_}" for _ in range(self.ndim)]
        self._state_fields = ["i"] + self._loc_fields
        self._verbose = verbose
        ############################################################
        #             Instantiate the Function Space               #
        #                                                          #
        # This is the data structure used by the gym which holds   #
        #   composed functions that have been created over time.   #
        ############################################################
        self._function_bank = FunctionBank(
            modeling_data=self.dataset,
            population_size=number_functions,
            **self._function_bank_kwargs,
        )
        ############################################################
        #            Instantiate the Experiment Space              #
        #                                                          #
        # This is the data structure used by the gym which holds   #
        #   composed functions that the agent has placed onto the  #
        #   canvas. This is built in reset.                        #
        ############################################################
        self._episode = -1
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
        # TODO: Stable baselines recommends this to be flattened, symmetric,
        # and normalized.
        # TODO: If we do that we just need to change the parse.
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1 + self.ndim, self._state_length),
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
            low=-np.inf, high=np.inf, shape=(self._action_size,)
        )

        ############################################################
        # Hyperparameters for Training and Testing composed Net    #
        ############################################################
        self.epochs = epochs
        self.net_lr = net_lr
        self.net_batch_size = net_batch_size

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

    def step(self, action: Optional[ArrayLike] = None) -> ArrayLike:
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
        # This is to prevent errors below.
        done = False
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
        action_index = int(
            np.round(
                np.clip(
                    2.5 * action[0], 0, self._function_bank.idxmax()
                )  # TODO: remove multiplier
            )
        )

        # This extracts the function location from the action.
        # This 'clips' the action location to the interior of the
        #   experiment space. It is already a float array, so nothing
        #   further is required.
        action_location = np.clip(
            action[1:-1], self.experiment_space_min, self.experiment_space_max
        )
        # This extracts the function radius from the action.
        # This 'clips' the radius to be non-negative
        action_radius = np.clip(
            50 * action[-1], 0, None
        )  # TODO: remove multiplier

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
        function_row: pd.DataFrame = self._function_bank.query(
            "i == {}".format(action_index)
        )
        function_set: FunctionSet = function_row.to_dict(orient="records")

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
        f_type: str = fun["type"]
        assert f_type in ["composed", "atomic"], err_msg
        # Dependent on what the Function type is, it will be handled
        #   differently.

        # Add composed functions directly to experiment space
        if f_type == "composed":
            self._experiment_space = append_to_experiment(
                experiment_space_container=self._experiment_space,
                function_bank=self._function_bank,
                composed_functions=[fun],
            )
        # Build properties of atomic functions then add to experiment space
        else:
            # Build KD tree with all function locations currently in exp space
            tree = cKDTree(self._experiment_space[self._loc_fields].values)

            # Query the KD Tree for all points within the action radius.
            idx = tree.query_ball_point(action_location, action_radius)

            # If any indices are returned it's a valid action
            if len(idx):
                # This uses the returned indices to subset the
                #   experiment space. This can contain input,
                #   intermediate, composed, and output nodes.
                connected_df = self._experiment_space.iloc[idx]

                # Add current function to experiment space
                self._build_atomic_function(
                    action_index, action_location, connected_df
                )

                # If current function connects to output, then trigger
                #   episode completion and connect function to sink.
                output_df = connected_df.query('type == "sink"')
                if output_df.shape[0]:
                    connected_to_sink = True
                    self._experiment_space.at[
                        self._experiment_space.query(
                            'type=="sink"'
                        ).index.item(),
                        "input",
                    ] = [self._experiment_space.tail(1).id.item()]

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
            self._experiment_space, self._function_bank
        )

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
            -----------------\n{state}
            """
            print(debug_message)

        # Check to see if it's time to call it a day.
        done = connected_to_sink or (self._step >= self.max_steps)
        if done:
            # Create an experiment space without sources or sinks.  This is
            # useful for several functions below
            intermediate_es = self._experiment_space.query(
                '(type != "source") and (type != "sink")'
            )

            # Check if net is empty, and if so, return 0 reward
            if not len(intermediate_es.index):
                return state, 0, done, info

            # Check if net has exactly 1 composed function coneected to the
            # sink, and if so return 0 reward
            if (
                len(intermediate_es.index) == 1
                and connected_to_sink
                and intermediate_es.iloc[0]["type"] == "composed"
            ):
                return state, 0, done, info

            # Else we have a legitimate net: connect to a sink if needed.
            if not connected_to_sink:
                # Build tree with intermediate function locations
                tree = cKDTree(intermediate_es[self._loc_fields].values)

                # Location of sink
                sink_loc = self._experiment_space.query('type == "sink"')[
                    self._loc_fields
                ]

                # Find closest intermediate function to sink node and attach
                last_index = tree.query(sink_loc, k=1)[1][0]

                self._experiment_space.at[
                    self._experiment_space.query('type=="sink"').index.item(),
                    "input",
                ] = [intermediate_es.iloc[last_index].id]

            # Get the id of the function that connects to the sink
            last_id = self._experiment_space.loc[
                self._experiment_space["type"] == "sink", "input"
            ].item()

            print("\n\nEPISODE:", self._episode)
            print("\nFinal Experiment Space:\n", self._experiment_space)

            # Build and save net
            id = 100 # TODO:How to generate ID?
            new_composed_fn = ComposedFunction(id, self._experiment_space,
                                                self._function_bank)

            # TODO: Train the net.
            # self._train_net()

            # TODO: Validate the net.
            # self._validate_net()

            # Add the completion reward.
            # reward += float(
            #     linear_completion_reward(self._experiment_space, None, 0.5)
            # )

        return state, reward, done, info

    def _build_atomic_function(
        self, action_index, action_location, connected_df
    ):
        """Takes an action index and action location for an atomic function
        and builds the corresponding row to be added to the experiment space.

        Parameters
        ----------
        action_index: float
            The function the agent chose to drop. Should be an atomic action.
        action_location: tuple(float, float)
            Where in the experiment space the function is dropped
        input_df: pd.DataFrame

        """
        # Check if valid atomic function
        assert action_index in [linear_i, relu_i, dropout_i]

        # Extract class of function
        function_class = self._function_bank.query(
            "i=={}".format(action_index)
        ).object.item()

        # Find all inputs to function
        inputs_df = connected_df.query('type != "sink"')
        inputs_hparams = inputs_df.hyperparameters.to_list()

        # Add all input sizes together
        sum_of_inputs = 0
        for inp_dict in inputs_hparams:
            if not len(inp_dict):
                sum_of_inputs += 1
            else:
                sum_of_inputs += inp_dict["output_size"]

        # Set function-specific hyperparameters
        if function_class == ReLU:
            intermediate_i = relu_i
            function_parameters = {
                "output_size": sum_of_inputs,
                "input_size": sum_of_inputs,
            }
        elif function_class == Dropout:
            intermediate_i = dropout_i
            function_parameters = {
                "p": dropout_p,
                "output_size": sum_of_inputs,
                "input_size": sum_of_inputs,
            }
        elif function_class == Linear:
            intermediate_i = linear_i
            function_parameters = {
                "output_size": linear_output_size,
                "input_size": sum_of_inputs,
            }

        # Create dictionary representation of function attributes
        built_function = make_function(
            function_index=intermediate_i,
            function_object=function_class,
            function_type="intermediate",
            function_inputs=inputs_df.id.to_list(),
            function_location=action_location,
            function_hyperparameters=function_parameters,
        )

        # Extract only fields needed for experiment space
        locs = [x for x in built_function.keys() if x.startswith("exp_loc")]
        new_function = {
            k: v
            for k, v in built_function.items()
            if k in experiment_space_fields + locs
        }

        # Add new function to experiment space
        self._experiment_space = append_to_experiment(
            experiment_space_container=self._experiment_space,
            function_bank=self._function_bank,
            composed_functions=[new_function],
        )

    def _build_net(self):
        """Builds the experiment space's envisioned net.

        Inputs
        ----------
        Takes in the last step's experiment space & layer right before the
        output layer.

        Returns
        ----------
        Adds the newly tested composed function to the function bank.
        Updates the metrics for the atomic/composed functions used in the
        newly composed function.

        """
        # comment out if function bank only has 'None' in inputs
        self._experiment_space = self._experiment_space.fillna(
            np.nan
        ).replace([np.nan], [None])

        # Create new experiment space with only functions in the net.  This
        # new data frame will have only intermediate and composite functions,
        # in reverse order from output to input.
        net_df = pd.DataFrame().reindex(
            columns=self._experiment_space.columns
        )
        net_df.loc[0] = self._experiment_space.query('type == "sink"').iloc[0]
        cur_inputs = deepcopy(net_df.tail(1).input.item())

        while len(cur_inputs):
            cur_input = cur_inputs[0]
            if cur_input not in net_df.id.values:
                net_df.loc[len(net_df.index)] = self._experiment_space.query(
                    "id == @cur_input"
                ).iloc[0]
                inps = net_df.tail(1).input.item()
                if inps != None:
                    for inp in inps:
                        if (
                            inp not in net_df.id.values
                            and self._experiment_space.query(
                                "id == @inp"
                            ).type.values
                            != "source"
                        ):
                            cur_inputs.append(inp)
            cur_inputs.pop(0)

        print("\n\nFinal Net (df):\n", net_df)

        # for ind in range(len(net_df)):
        # fn_type = net_df.iloc[ind]["i"]
        # fn_params = net_df.iloc[ind]["hyperparameters"]
        # if fn_type == relu_i:
        # self.net_init.append(nn.ReLU())
        # elif fn_type == linear_i:
        # self.net_init.append(
        # nn.Linear(
        # fn_params["input_size"], fn_params["output_size"]
        # )
        # )
        # elif fn_type == dropout_i:
        # self.net_init.append(nn.Dropout(fn_params["p"]))

        # print("\n\nPyTorch Init:\n", self.net_init)

        # TODO: BUILD COMPUTATION GRAPH

        # Creating np arrays
        target = self.dataset["output"].values
        features = self.dataset.drop("output", axis=1).values

        # Passing to DataLoader
        train = data_utils.TensorDataset(
            torch.tensor(features), torch.tensor(target)
        )
        train_loader = data_utils.DataLoader(
            train, batch_size=self.net_batch_size, shuffle=True
        )
#        for idx, (data, target) in enumerate(train_loader):
#            print("train_loader:\n", (data, target))
#            print("column 0 !!\n", (data[:, 0:2], target))
        #        optimizer = torch.optim.Adam(, lr=self.net_lr)
        criterion = nn.CrossEntropyLoss()

    #        _train_net()

    def _train_net(self, data_loader, optimizer, criterion):
        iter_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        for idx, (data, target) in enumerate(data_loader):
            start = time.time()

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            ###########################################################
            # TODO: Complete the body of training loop                #
            #       1. forward data batch to the model                #
            #       2. Compute batch loss                             #
            #       3. Compute gradients and update model parameters  #
            ###########################################################

            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            ###########################################################
            #                              END OF YOUR CODE           #
            ###########################################################

            batch_acc = accuracy(out, target)

            losses.update(loss, out.shape[0])
            acc.update(batch_acc, out.shape[0])

            iter_time.update(time.time() - start)
            if idx % 10 == 0:
                print(
                    (
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t"
                    ).format(
                        epoch,
                        idx,
                        len(data_loader),
                        iter_time=iter_time,
                        loss=losses,
                        top1=acc,
                    )
                )

    def build_state(self) -> ArrayLike:
        """Builds an observation from experiment space."""
        _exp_state = self._experiment_space[self._state_fields].values
        _pad_state = np.zeros(
            (self._state_length - _exp_state.shape[0], 1 + self.ndim)
        )
        return np.concatenate([_exp_state, _pad_state]).T

    def reset(self):
        """Initialize state space.

        This creates an empty canvas for the experiment space
        consisting of nothing but input and output nodes.
        """
        self.net_init = nn.ModuleList([])
        # Reset the step counter
        self._step = 0
        # Increment the episode counter
        self._episode += 1
        # Fill the experiment space.
        self._experiment_space = refresh_experiment_container(
            function_bank=self._function_bank,
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
        # And save the bank.
        # TODO: Uncomment this when ready to test it.
        self._function_bank._save_bank()
        if self._verbose:
            debug_message = f"""Environment Reset:
            Current Step: {self._step}

            Current Experiment Space
            ------------------------\n{self._experiment_space}

            Current Observation Space
            -------------------------\n{state}

            Current Function Bank
            ---------------------\n{self._function_bank.to_df()}
            """
            print(debug_message)
        return state

    def render(self, mode="human"):
        pass

    def close(self):
        pass


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
