import logging
import time
from typing import Optional
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree
import gym
import gin
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
import pandas as pd
from mentalgym.functionbank import FunctionBank
from mentalgym.types import Function, FunctionSet
from mentalgym.utils.function import make_function, make_id
from mentalgym.functions.composed import ComposedFunction
import mentalgym.functions.composed
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


@gin.configurable
class MentalEnv(gym.Env):
    """A Mental Gymnasium Environment.

    This class allows a reinforcement learning agent to learn to
    'paint by numbers' with composed functions. It can select
    functions from the palette to drop onto the experiment.

    Parameters
    ----------
    dataset: pd.DataFrame
        This is a modeling dataset.
    valset: pd.DataFrame
        This is the validation set.
    testset: pd.DataFrame
        This is the test set.
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
    n_classes: int = 2
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
        valset: pd.DataFrame,
        testset: pd.DataFrame,
        experiment_space_min: ArrayLike = np.array([0.0, 0.0]),
        experiment_space_max: ArrayLike = np.array([100.0, 100.0]),
        number_functions: int = 8,
        max_steps: int = 4,
        epochs: int = 5,
        net_lr: float = 0.0001,
        net_batch_size: int = 128,
        n_classes: int = 2,
        seed: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Sets up the gym environment.

        This instantiates a function bank and an experiment space.
        """
        super(MentalEnv, self).__init__()

        dataset.columns = [str(_) for _ in dataset.columns]
        valset.columns = [str(_) for _ in valset.columns]
        testset.columns = [str(_) for _ in testset.columns]
        self.dataset = dataset
        self.valset = valset
        self.testset = testset

        ############################################################
        #                 Store Hyperparameters                    #
        ############################################################
        # These are used in building the experiment space and when
        #   validating agent actions.
        self.experiment_space_min = np.array(experiment_space_min)
        self.experiment_space_max = np.array(experiment_space_max)
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
        self.n_classes = n_classes

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
        # Default values here, or pass some info?
        info = {}
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
        # TODO: If flipping to palette, edit here to palette size.
        action_index = int(
            np.round(
                np.clip(
                    action[0], 0, self._function_bank.idxmax()
                )
            )
        )
        # This extracts the function location from the action.
        # This 'clips' the action location to the interior of the
        #   experiment space. It is already a float array, so nothing
        #   further is required.
        action_location = np.clip(
            15 * action[1:-1], self.experiment_space_min, self.experiment_space_max
        )  # TODO: remove multiplier
        # This extracts the function radius from the action.
        # This 'clips' the radius to be non-negative
        action_radius = np.clip(
            20 * action[-1], 0, None
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

        # If function is composed, add directly to experiment space after
        #   providing its location in the experiment space and the function
        #   bank
        if f_type == "composed":
            fun["exp_loc_0"] = action_location[0]
            fun["exp_loc_1"] = action_location[1]
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

                # Don't allow ReLUs to connect to source nodes
                if action_index == relu_i:
                    connected_df = connected_df[connected_df.type != "source"]
                    if connected_df.empty:
                        return self.state, 0, done, info

                # Add current function to experiment space
                self._build_atomic_function(
                    action_index, action_location, connected_df
                )

                # If current function connects to output, then trigger
                #   episode completion and connect function to sink.
                output_df = connected_df.query('type == "sink"')
                if output_df.shape[0]:
                    last_index = self._experiment_space.tail(1).index.item()
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

        # Return a minor reward if there are *any* nodes added.
        reward = connection_reward(
            self._experiment_space, self._function_bank
        )


        # Extract the state space.
        self.state = self.build_state()
        if self._verbose:
            debug_message = f"""End of Step:
            Connected to the sink node: {connected_to_sink}
            Hit maximum steps: {self._step >= self.max_steps}
            Total Reward: {reward}
            Done: {done}

            State Observation
            -----------------\n{self.state}
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
                return self.state, 0, done, info

            # Check if net has exactly 1 composed function coneected to the
            # sink, and if so return 0 reward
            if (
                len(intermediate_es.index) == 1
                and connected_to_sink
                and intermediate_es.iloc[0]["type"] == "composed"
            ):
                return self.state, 0, done, info

            # Else we have a legitimate net: connect to a sink if needed.
            if not connected_to_sink:
                # Build tree with intermediate function locations
                tree = cKDTree(intermediate_es[self._loc_fields].values)

                # Location of sink
                sink_loc = self._experiment_space.query('type == "sink"')[
                    self._loc_fields
                ]

                # Find closest intermediate function to sink node and attach
                last_es_index = tree.query(sink_loc, k=1)[1][0]
                self._experiment_space.at[
                    self._experiment_space.query('type=="sink"').index.item(),
                    "input",
                ] = [intermediate_es.iloc[last_es_index].id]

                # We need to count the number of sources and sinks here
                #   in order to find the last layer's index, since the
                #   intermediate experiment space lost that information
                num_sources_and_sinks = len(
                    self._experiment_space.query(
                        '(type == "source") or (type == "sink")'
                    )
                )
                last_index = last_es_index + num_sources_and_sinks

            print("\nFinal Experiment Space:\n", self._experiment_space)

            # TODO: Use the make_function to generate the ID, then create
            #   a new composed function like you're doing. The ID will
            #   get updated in the 'id' field and added to the hyperparameter
            #   field {'id': inp}
            # This composed function *is* your net. Assign it to a model,
            #   and call it on the input.

            # Generate ID and get output size for newly composed function
            # TODO allow for concatenation of arbitrary number of output
            #   layers
            id = make_id()
            composed_output_size = self._experiment_space.iloc[
                last_index
            ].hyperparameters["output_size"]

            # Get final id
            final_id = self._experiment_space.iloc[last_index].id

            # Instantiate new composed function
            new_composed_fn = ComposedFunction(
                id,
                self._experiment_space,
                self._function_bank,
                output_size=composed_output_size,
            )

            # Make new composed function
            made_function = make_function(
                function_id=id,
                function_object=ComposedFunction,
                function_hyperparameters={
                    "input_size": new_composed_fn.input_size,
                    "output_size": new_composed_fn.output_size,
                    "function_dir": new_composed_fn._function_dir
                },
                function_inputs=list(new_composed_fn.inputs.keys()),
                function_type="composed",
            )

            # Append new composed function to function bank
            self._function_bank.append(made_function)
            print("\nFUNCTION BANK:\n", self._function_bank.to_df())

            model = new_composed_fn
            model._module_dict['output'] = nn.Linear(new_composed_fn.output_size, self.n_classes)

            # Set training data
            new_fn_space = new_composed_fn._net_subspace
            cols = new_fn_space.query('type == "source"').id.values

            Xtrain = self.dataset[cols].values
            ytrain = self.dataset.loc[:, self.dataset.columns == 'output'].values.squeeze()

            # Set validation data
            Xval = self.valset[cols].values
            yval = self.valset.loc[:, self.valset.columns == 'output'].values.squeeze()

            print("\nMODEL:", model.parameters)
            complexity = new_composed_fn.complexity
            optimizer = torch.optim.Adam(model.parameters(), lr=self.net_lr)
            criterion = nn.CrossEntropyLoss()

            train_loss_history = []
            train_acc_history = []
            valid_loss_history = []
            valid_acc_history = []
            best_acc = 0.0
            for epoch in range(self.epochs):
                # Train
                batched_train_data, batched_train_label = self._generate_batched_data(Xtrain, ytrain)
                self._adjust_learning_rate(optimizer, epoch)
                epoch_loss, epoch_acc = self._train_net(epoch, batched_train_data, batched_train_label, model, optimizer, criterion, final_id)
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)

                # Validate
                batched_val_data, batched_val_label = self._generate_batched_data(Xval, yval)
                valid_loss, valid_acc = self._validate_net(batched_val_data, batched_val_label, model, criterion, final_id)
                print("* Validation Accuracy: {accuracy:.4f}".format(accuracy=valid_acc))

                valid_loss_history.append(valid_loss)
                valid_acc_history.append(valid_acc)

                if valid_acc > best_acc:
                    best_acc = valid_acc

            # Add the completion reward.
            reward += float(
                linear_completion_reward(self._experiment_space, None, best_acc)
            )

            # self._function_bank.score(** experiment_space.IDs)

        return self.state, reward, done, info

    def _generate_batched_data(self, data, label):
        batched_data = []
        batched_label = []
        indices = list(range(len(label)))

        for i in range(0, len(label), self.net_batch_size):
            batched_data.append(np.array([data[j] for j in indices[i:i+self.net_batch_size]]))
            batched_label.append([label[j] for j in indices[i:i+self.net_batch_size]])
        return batched_data, batched_label

    def _adjust_learning_rate(self, optimizer, epoch):
        lr_start = self.net_lr
        lr_max = 1e-1
        lr = min(lr_max, lr_start / (epoch + 1))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


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
        sum_of_inputs = self._sum_inputs(inputs_hparams)

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

    def _sum_inputs(self, inputs_hparams):
        """ Checks container of hyperparameter dictionaries in order to
            compute the input width of a given layer.  Each dictionary
            in the container should represent a layer that is itself an
            input to the layer whose input width we're computing.
            Therefore, we're going to be checking each dict's "output_size".

        Inputs
        ----------
        Container of dictionaries

        Returns
        ----------
        Sum of the outputs of all layers represented in the container
        """
        sum_of_inputs = 0
        for parameter_dict in inputs_hparams:
            # If the dictionary is empty, assume output of this layer is a
            #   single column
            if not len(parameter_dict):
                sum_of_inputs += 1
            # Otherwise it's the sum of the output size from the
            #   layer above tacked on to the accumulating value.
            else:
                sum_of_inputs += parameter_dict["output_size"]
        return sum_of_inputs

    def _accuracy(self, output, target):
        """Computes the precision@k for the specified values of k"""
        batch_size = target.shape[0]
        _, pred = torch.max(output, dim=-1)
        correct = pred.eq(target).sum() * 1.0
        acc = correct / batch_size
        return acc

    def _train_net(self, epoch, batched_train_data, batched_train_label, model, optimizer, criterion, final_id):

        epoch_loss = 0.0
        hits = 0
        count_samples = 0.0
        for idx, (input, target) in enumerate(zip(batched_train_data, batched_train_label)):
            start_time = time.time()
            input = torch.Tensor(input)
            target = torch.LongTensor(target)
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            out = model._recursive_forward(final_id, input)

            try:
                assert target[target<0].numel() == 0 and target[target>=self.n_classes]. numel() == 0
            except:
                print("Bad targets:", target[target<0], target[target>=self.n_classes])
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                batch_acc = self._accuracy(out, target)
                epoch_loss += loss
                hits += batch_acc * input.shape[0]
                count_samples += input.shape[0]
                forward_time = time.time() - start_time
                if idx % 1 == 0:
                    print(epoch)
                    print(idx)
                    print(len(batched_train_data))
                    print(forward_time)
                    print(loss)
                    print(batch_acc)
                    print(('Epoch: [{0}][{1}/{2}]\t'
                      'Batch Time {batch_time:.3f} \t'
                      'Batch Loss {loss:.4f}\t'
                      'Train Accuracy ' + "{accuracy:.4f}" '\t').format(
                    epoch, idx+1, len(batched_train_data), batch_time=forward_time,
                    loss=loss, accuracy=batch_acc))
        epoch_loss /= len(batched_train_data)
        epoch_acc = hits / count_samples

        print("* Average Training Accuracy of Epoch {} is: {:.4f}".format(epoch, epoch_acc))

        return epoch_loss, epoch_acc

    def _validate_net(self, batched_test_data, batched_test_label, model, criterion, final_id):
        epoch_loss = 0.0
        hits = 0
        count_samples = 0.0
        for idx, (input, target) in enumerate(zip(batched_test_data, batched_test_label)):
            start_time = time.time()
            input = torch.Tensor(input)
            target = torch.LongTensor(target)
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            with torch.no_grad():
                out = model._recursive_forward(final_id, input) # TODO: Need ID
                loss = criterion(out, target)
                batch_acc = self._accuracy(out, target)
                epoch_loss += loss
                hits += batch_acc * input.shape[0]
                count_samples += input.shape[0]
                forward_time = time.time() - start_time
                if idx % 1 == 0:
                    print(('Validate: [{0}/{1}]\t'
                      'Batch Time {batch_time:.3f} \t'
                      'Batch Loss {loss:.4f}\t'
                      'Batch Accuracy ' + "{accuracy:.4f}" '\t').format(
                        idx+1, len(batched_test_data), batch_time=forward_time,
                        loss=loss, accuracy=batch_acc))
        epoch_loss /= len(batched_test_data)
        epoch_acc = hits / count_samples

        return epoch_loss, epoch_acc

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
        # Reset the step counter
        self._step = 0
        # Increment the episode counter
        self._episode += 1
        if self._episode > 0:
            print("\n\nEPISODE:", self._episode)
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
        self.state = self.build_state()
        # And save the bank.
        # TODO: Uncomment this when ready to test it.
        self._function_bank._save_bank()
        if self._verbose:
            debug_message = f"""Environment Reset:
            Current Step: {self._step}

            Current Experiment Space
            ------------------------\n{self._experiment_space}

            Current Observation Space
            -------------------------\n{self.state}

            Current Function Bank
            ---------------------\n{self._function_bank.to_df()}
            """
            print(debug_message)
        return self.state

    def render(self, mode="human"):
        pass

    def close(self):
        pass
