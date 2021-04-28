import logging
from typing import Optional
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree
import gym
import torch.nn as nn
import pandas as pd
from mentalgym.constants import experiment_space_fields
from mentalgym.functionbank import FunctionBank
from mentalgym.types import Function, FunctionSet
from mentalgym.utils.function import make_function
from mentalgym.utils.reward import connection_reward, linear_completion_reward
from mentalgym.utils.spaces import (
    refresh_experiment_container,
    append_to_experiment,
)
from mentalgym.functions.atomic import Linear, ReLU, Dropout
from mentalgym.constants import intermediate_i, linear_output_size, dropout_p
from mentalgym.utils.data import testing_df


# Customize pandas DataFrame display width
# pd.set_option("display.expand_frame_repr", False)

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
        **kwargs,
    ):
        """Sets up the gym environment.

        This instantiates a function bank and an experiment space.
        """
        super(MentalEnv, self).__init__()

        dataset.columns = [str(_) for _ in dataset.columns]
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
            modeling_data=dataset,
            population_size=number_functions,
            **self._function_bank_kwargs,
        )
        # self._function_bank = function_bank
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
            np.round(np.clip(action[0], 0, self._function_bank.idxmax()))
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
        action_radius = 50
#        action_radius = np.clip(action[-1], 0, None)

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

        if f_type == "composed":
            # If it's a composed Function it is just appended to the
            #   experiment container. TODO: Test
            self._experiment_space = append_to_experiment(
                experiment_space_container=self._experiment_space,
                function_bank=self._function_bank,
                composed_functions=[fun],
            )
        else:
            # If it's an atomic Function it is added as an
            #   'intermediate' and then constructed appropriately
            #   in the build_net.

            # Build a KD tree from the locations of the nodes in the
            # experiment space.
            tree = cKDTree(self._experiment_space[self._loc_fields].values)

            # Query the KD Tree for all points within the radius.
            idx = tree.query_ball_point(action_location, action_radius)

            # If any indices are returned it's a valid action
            if len(idx):
                # This uses the returned indices to subset the
                #   experiment space. This can contain input,
                #   intermediate, and composed nodes, in addition
                #   to output. This checks for output, removes it,
                input_df = self._experiment_space.iloc[idx]
                output_df = input_df.query('type == "sink"')

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
                #   them to a separate data structure. Advantages and
                #   disadvantages either way.
                function_class = self._function_bank.query(
                    "i=={}".format(action_index)
                ).object.item()
                input_df = input_df.query('type != "sink"')
                input_hparams = input_df.hyperparameters.to_list()
                sum_of_inputs = 0

                for inp_dict in input_hparams:
                    if not len(inp_dict):
                        sum_of_inputs += 1
                    else:
                        sum_of_inputs += inp_dict["output_size"]

                if function_class == ReLU:
                    self.function_parameters = {
                        "output_size": sum_of_inputs,
                        "input_size": sum_of_inputs,
                    }
                elif function_class == Dropout:
                    self.function_parameters = {
                        "p": dropout_p,
                        "output_size": sum_of_inputs,
                        "input_size": sum_of_inputs,
                    }
                elif function_class == Linear:
                    self.function_parameters = {
                        "output_size": linear_output_size,
                        "input_size": sum_of_inputs,
                    }

                #################
                # Note: See if we can build computation graph HERE.
                # Need to import dataset and assign the inputs
                # (and possibly batch them)
                #################

                built_function = make_function(
                    function_index=intermediate_i,
                    function_object=function_class,
                    function_type="intermediate",
                    function_inputs=input_df.id.to_list(),
                    function_location=action_location,
                    function_hyperparameters=self.function_parameters,
                )

                locs = [
                    x
                    for x in built_function.keys()
                    if x.startswith("exp_loc")
                ]
                new_function = {
                    k: v
                    for k, v in built_function.items()
                    if k in experiment_space_fields + locs
                }

                self._experiment_space = append_to_experiment(
                    experiment_space_container=self._experiment_space,
                    function_bank=self._function_bank,
                    composed_functions=[new_function],
                )

                # If the output was connected, then we will trigger
                #   completion and build and run the net.
                if output_df.shape[0]:
                    connected_to_sink = True

                    self._experiment_space.loc[
                        self._experiment_space["type"] == "sink", "input"
                    ] = self._experiment_space.tail(1).id.item()

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
                self._experiment_space.loc[
                    self._experiment_space["type"] == "sink", "input"
                ] = intermediate_es.iloc[last_index].id

            # Get the id of the function that connects to the sink
            last_id = self._experiment_space.loc[
                          self._experiment_space["type"] == "sink", "input"
                      ].item()
 
            print("\n\nEPISODE:", self._episode)
            print("\nFinal Experiment Space:\n", self._experiment_space)

            # TODO: Bake the net.
            self._build_net()

            # Add the completion reward.
            reward += float(
                linear_completion_reward(self._experiment_space, None, 0.5)
            )

        return state, reward, done, info

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
        net_df = pd.DataFrame().reindex(columns=self._experiment_space.columns)
        net_df.loc[0] = self._experiment_space.query('type == "sink"').iloc[0]
        cur_inputs = [net_df.tail(1).input.item()]

        while len(cur_inputs):
            if cur_inputs[0] not in net_df.id.values:
                net_df.loc[len(net_df.index)] = self._experiment_space.query('id == @cur_inputs[0]').iloc[0]
                inps = net_df.tail(1).input.item()
                if inps != None:
                    for inp in inps:
                        if inp not in net_df.id.values and self._experiment_space.query('id == @inp').type.values != 'source':
                            cur_inputs.append(inp)
                cur_inputs.pop(0)
        net_df = net_df.sort_index(ascending=False)
                
        print("\n\nFinal Net (df):\n", net_df)

#        net_list = net_df.sort_index(ascending=False).id.to_list()
#        print("\nNet Dict:\n", net_dict)

        for ind in range(len(net_df)):
            function_class = self._function_bank.query(
                "i=={}".format(-net_df.iloc[ind]['i']-1)
            ).object.item()
            if function_class == ReLU:
                self.net_init.append(nn.ReLU())
            elif function_class == Linear:
                self.net_init.append(
                    nn.Linear(
                        self.function_parameters["input_size"],
                        self.function_parameters["output_size"],
                    )
                )
            elif function_class == Dropout:
                self.net_init.append(
                    nn.Dropout(self.function_parameters["p"])
                )

        print("\n\nPyTorch Init:\n", self.net_init)

#        cur_inputs = [self._experiment_space.query('type == "sink"')["input"].item()]
#        while True:
#            net_df[len(net_df)+1] = cur_inputs
#
#            net_df = self._build_net_df(self._experiment_space, cur_inputs)
#
#        print("net_df:\n", net_df)
#
#    def _build_net_df(self, exp_space, ):
        

#        net_d = {}
#
#        # comment out if function bank only has 'None' in inputs
#        self._experiment_space = self._experiment_space.fillna(
#            np.nan
#        ).replace([np.nan], [None])
#        net_d[last_id] = self._recurser(self._experiment_space, last_id)


#            )
#        print("\nFinal Net:\n", self.net_init)

        # self._experiment_space['id']['object'].init()

        # make sure to instantiate the output layer

        #        print("NET_D ----------------------", net_d)

        # # init the layer
        # self.layer1 = nn.Linear(n_in, n_out)
        # # setting the weights
        # self.layer1.weights = load('weights.pt')

        # self.model = nn.ModuleList([])
        # self.model.append(steve(1, max(1/2, 16)))  # arbitrary output
        # self.model.append(bob(2, 1))    # input == prev_output
        # self.model.append(carl(?, 1))

        # self.layer1 = nn.Linear(n_in, n_out)
        # self.layer1a = nn.Linear(n_in, n_out)
        # self.layer2 = nn.Linear(n_out*2, final_out)

        # z = self.layer1(x)
        # y = self.layer1a(x)
        # out = self.layer2(y+z)

        # for i in range(len(self.model)):
        #     self.model[i](x)

        #     s_out = steve(col0)

        #     b_out = bob(col0,col1)

        #     carl(s_out, b_out, col1)

        # self.model.append(output(1, 1))

        # self._experiment_space.id['object'].item() #

        # recurse init(d['input']) till {}

        # Rearrange the order of the functions from output to input

        #   - parse exp space for functions and their inputs
        #   - recurse for composed actions

        # Option: WAN

        # Instantiate the net according to arrangement
        # Intermediate functions: use Vahe's torch functions
        # Composed functions:
        #   - pull Vahe's torch layer
        #   - initiate all function weights to shared,
        #     random value [-1, 0.5, 1] -- I forgot the real WAN values...

        # Option: Not WAN

        # Instantiate the net according to arrangement
        # Intermediate functions: use Vahe's torch functions
        # Composed functions:
        #   - pull Vahe's torch layer
        #   - get saved weights for composed function

        # Train the net
        # Make sure only inputs that are connected go into each layer

        # Save the weights of the composed function to disk

        # Get and update the metrics of the new function

        # Save the new function to the function bank

        # TODO: represent intermediate functions in a simple to use manner
        # whilestill persisting the info to build them from atomic functions

    # Alternate Recurser
    # def _recurser(self, exp_space, id):
    # data = exp_space.query("id==@id")
    # inputs = data.input.iloc[0]  # list of all inputs to that particular id
    # flag = False
    # if inputs == None:
    # return True

    # Linear(column0, column1)
    # x = n
    # x = self.nn.init[0](column0, column1)

    # if flag:
    # return {_: (self._recurser(exp_space, _)) for _ in inputs}

    # def _recurser(self, exp_space, id):
        # data = exp_space.query("id==@id")
        # # print("\n")
        # # print("exp_space:\n", exp_space)
        # # print("id:", id)
        # # print("data:\n", data)
        # inputs = data.input.iloc[
            # 0
        # ]  # list of all inputs to that particular id
        # # print("inputs:", inputs)
        # # print("\n")

        # if inputs == None:
            # return 1

        # return {
            # _: (self._recurser(exp_space, _), len(inputs)) for _ in inputs
        # }

    # TODO: finish this
    # def _recurser_init(self, order_d):
    #     # input ex: {'steve': {'column_0': {}}}

    #     # init the pytorch layers
    #     #   - need: # of inputs to the layer
    #     #   - need: # of outputs to the layer
    #     # ex: nn.Linear(n_in, n_out)
    #     # ex: function.Linear(n_in, n_out)
    #     # steve's input len == # of keys

    #     if inputs == {}:
    #         return 1

    #     return
    #     # return recurse(d['inputs'])
    #     return { _ : self._recurser(exp_space, _) for _ in inputs}

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
        self.function_parameters = dict()
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
