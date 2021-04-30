"""Contains testing for Composed Functions.

Composed functions can, when given a set of input nodes from a
FunctionBank, produce a coherent PyTorch structure.

The test cases below represent instances of Directed Acyclic Graphs
and the expected output.
"""
import numpy as np
import pandas as pd
import pytest
from mentalgym.functions import Linear, ReLU, Dropout
from mentalgym.functionbank import FunctionBank
from mentalgym.functions import (
    atomic_constants,

)
from mentalgym.functions.composed import ComposedFunction
from mentalgym.types import ExperimentSpace
from mentalgym.utils.function import make_function
from mentalgym.utils.spaces import (
    append_to_experiment,
    get_experiment_neighbors,
    get_output_inputs,
    refresh_experiment_container
)
from numpy.typing import ArrayLike
from sklearn.datasets import make_classification
from tempfile import TemporaryDirectory
from typing import Dict, Iterable

# TODO: More test cases here would be extremely useful. Ctrl+F for
#   Simple Test Case and create a new testing case. Ensure the
#   output is reasonable and insert the output.
####################################################################
#                        Prerequisite Data                         #
####################################################################
# This series of test cases requires both an experiment space and  #
#   a function bank. This test routine below simulates some        #
#   composed functions being created and added to the function     #
#   bank, one after another. The prerequisite data includes an     #
#   ExperimentSpace object and a FunctionBank object.              #
####################################################################

# Make a testing dataset.
X, y = make_classification(
    n_features = 100,
    n_informative = 40,
    n_redundant = 40,
    random_state = 0
)
test_data = pd.DataFrame(
    X
).assign(y=y)
# Give it string names so the gym doesn't get cranky.
test_data.columns = [str(_) for _ in test_data.columns]

# This is used in the testing functions to retrieve function objects.
def drop_layer(
    experiment_space: ExperimentSpace,
    function_bank: FunctionBank,
    index: int,
    location: ArrayLike,
    radius: float
):
    """Adds a layer to an Experiment Space.

    Parameters
    ----------
    experiment_space: ExperimentSpace
        An exeriment space object.
    function_bank: FunctionBank
        A function bank object.
    index: int
        An integer representing the index in the function bank.
    location: ArrayLike
        A location to drop a node.
    radius: float
        A radius for the dropped node.
    """
    # 1) Get the input ids.
    input_ids = get_experiment_neighbors(
        experiment_space = experiment_space,
        location = location,
        radius = radius
    )
    # 2) Make the layer with make_function
    layer_function = make_function(
        function_index = index,
        function_inputs = input_ids,
        function_location = location,
        function_type = 'intermediate',
        function_object = atomic_constants[index],
        function_hyperparameters = {}
    )
    # 3) This just does some name matching.
    locs = [
        x
        for x in layer_function.keys()
        if x.startswith("exp_loc")
    ]
    new_function = {
        k: v
        for k, v in layer_function.items()
        if k in list(experiment_space.columns) + locs
    }
    # 4) Then this sets the hyperparameters appropriately.
    sum_of_inputs = 0
    inputs_hparams = experiment_space.query(
        f'id=={input_ids}'
    ).hyperparameters.to_list()
    # This is walking down a list of dictionaries.
    for parameter_dict in inputs_hparams:
        # If the dictionary is empty
        if not len(parameter_dict):
            # Then it's a single column coming in. Increment the
            #   count of input by one.
            sum_of_inputs += 1
        else:
            # Otherwise it's the sum of the output size from the
            #   layer above tacked on to the accumulating value.
            sum_of_inputs += parameter_dict["output_size"]

    # Set function-specific hyperparameters
    if atomic_constants[index] == ReLU:
        new_function['hyperparameters'] = {
            "output_size": sum_of_inputs, #TODO: Think this needs to change to 
            "input_size": sum_of_inputs,
        }
    elif atomic_constants[index] == Dropout:
        new_function['hyperparameters'] = {
            "p": .5,
            "output_size": sum_of_inputs,
            "input_size": sum_of_inputs,
        }
    elif atomic_constants[index] == Linear:
        print("SHOULD BE HERE")
        new_function['hyperparameters'] = {
            "out_features": 12,
            "in_features": sum_of_inputs,
        }
    experiment_space = append_to_experiment(
        experiment_space_container = experiment_space,
        function_bank = function_bank,
        composed_functions = [new_function],
    )
    return experiment_space

####################################################################
#                        Simple Test Case 1                        #
####################################################################
# In this test case a very simple Linear Layer is produced which   #
#   takes as input two features ([1, 0]).                          #
####################################################################
test_set_1 = {
    'actions': [
        {'id': 0, 'location': (0,0), 'radius': 2},
    ],
    'expected_inputs': {},
    'minimal_space': pd.DataFrame()
}
####################################################################
#                        Simple Test Case 2                        #
####################################################################
# In this test case a very simple Linear Layer is produced which   #
#   takes as input two features ([1, 0]). Then, a ReLU layer is    #
#   placed on top of that. In this instance the ReLU layer is      #
#   closest to the sink.
####################################################################
test_set_2 = {
    'actions': [
        {'id': 0, 'location': (0,0), 'radius': 2},
        {'id': 1, 'location': (0,1), 'radius': 2},
    ],
    'expected_inputs': {},
    'minimal_space': pd.DataFrame()
}

####################################################################
#                            Unit Testing                          #
####################################################################
# These routines will test unit functionality of the
#   ComposedFunction.
def inputs_tester(
    expected_inputs: Dict[str, int],
    actual_inputs: Dict[str, int]
):
    f"""Tests the inputs of a composed function.

    This needs a .inputs dictionary which is used to map input
    names to integer locations. This is a 'key mapping'.

    {'input1': 0, 'input2': 2}

    Parameters
    ----------
    expected_inputs: Dict[str, int]
        The expected key mapping
    actual_inputs: Dict[str, int]
        The actual key mapping
    """
    err_msg = """ComposedFunction Input Mapping Error:

    The ComposedFunction stores a mapping dictionary allowing it
    to extract the correct positional indices from an input dataset.

    Expected Value
    --------------\n{expected_inputs}

    Actual Value
    ------------\n{actual_inputs}
    """
    # TODO: Implement this test.
    # This needs to test equivalency of the items in the data, but
    #   order is irrelevant.
    raise Exception(err_msg)

def minimal_subspace_tester(
    expected_space: pd.DataFrame,
    actual_space: pd.DataFrame
):
    """Tests the inputs of a composed function.

    This needs a .inputs dictionary which is used to map input
    names to integer locations. This is a 'key mapping'.

    {'input1': 0, 'input2': 2}

    Parameters
    ----------
    expected_space: pd.DataFrame
        The expected subspace
    actual_space: pd.DataFrame
        The actual subspace
    """
    err_msg = f"""ComposedFunction Input Subspace Error:

    The ComposedFunction stores a minimal subspace representation
    of the functions within it. This subspace is showing a difference
    between the actual and expected values.

    Expected Value
    --------------\n{expected_space}

    Actual Value
    ------------\n{actual_space}
    """
    assert expected_space.equals(actual_space), err_msg

####################################################################
#                       Integration Testing                        #
####################################################################
# This test routine tests much of the internal functionality of the#
#   composed function while it asserts that the output is a        #
#   runnable PyTorch layer.                                        #
####################################################################
test_sets = [
    test_set_1,
    test_set_2
]
verbose = True

@pytest.mark.parametrize('test_set', test_sets)
def test_composed_function(test_set):
    with TemporaryDirectory() as d:
        # Make a Function Bank.
        function_bank = FunctionBank(
            modeling_data = test_data,
            target = 'y',
            function_bank_directory = d,
            force_refresh = True
        )
        # Make an Experiment Space from the Function Bank
        experiment_space = refresh_experiment_container(
            function_bank = function_bank,
            min_loc = np.array([0, 0]),
            max_loc = np.array([100, 100])
        )
        status_message = f"""Composed Function Test:

        Function Bank
        -------------\n{function_bank.to_df()}

        Action Set
        ----------\n{test_set['actions']}

        Experiment Space Prior to Actions
        ---------------------------------\n{experiment_space}
        """
        if verbose: print(status_message)
        # Take the actions, one at a time.
        for action in test_set['actions']:
            experiment_space = drop_layer(
                experiment_space = experiment_space,
                function_bank = function_bank,
                index = action['id'],
                location = action['location'],
                radius = action['radius']
            )
        # Then snap the nearest node to the output.
        nearest_id: str = get_output_inputs(
            experiment_space = experiment_space
        )
        # Add that information to the experiment space.
        experiment_space.at[
            experiment_space.query('type == "sink"').index.item(),
            "input"
        ] = [nearest_id]
        status_message = f"""

        Experiment Space Posterior to Actions
        -------------------------------------\n{experiment_space}
        """
        if verbose:
            print(status_message)
        # Now we are going to create a Composed Function from these
        #   actions.
        composed_function = make_function(
            function_index = function_bank.idxmax() + 1,
            function_type = 'composed',
            function_object = ComposedFunction,
            function_hyperparameters = {}
        )
        # This, when it's called for the first time, builds
        #   a net and assigns it to forward.
        composed_instance = ComposedFunction(
            id = composed_function['id'],
            experiment_space = experiment_space,
            function_bank = function_bank,
            verbose = verbose
        )
        # At this point we have the first test. Does the minimal
        #   space created by the composed function match the expectation?
        inputs_tester(composed_function['inputs'], composed_instance.inputs)
        # Set inputs.
        composed_function['inputs'] = composed_instance.inputs
        composed_function['hyperparameters']['id'] = composed_function['id']
        status_message = f"""

        Composed Function Representation
        --------------------------------\n{composed_function}
        """
        if verbose:
            print(status_message)
####################################################################
#  The example PyTorch equivalent is shown below and the resultant #
#   weight structures compared.                                    #
####################################################################



test_composed_function(test_set_1)