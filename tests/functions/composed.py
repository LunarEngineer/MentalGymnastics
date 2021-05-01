"""Contains testing for Composed Functions.

Composed functions can, when given a set of input nodes from a
FunctionBank, produce a coherent PyTorch structure.

The test cases below represent instances of Directed Acyclic Graphs
and the expected output.
"""
import numpy as np
import pandas as pd
pd.options.display.max_columns = 200
import pytest
import torch
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
from torch import nn
from typing import Dict, Iterable, Type

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

def frame_tester(
    df_expected: pd.DataFrame,
    df_actual: pd.DataFrame
):
    if not df_expected.equals(df_actual):
        truth_dict = {}
        for col in df_expected.columns:
            truth_dict[col] = df_expected[col].equals(df_actual[col])
        err_msg = f"""Frame matching disparity:

        Expected Dataset
        ----------------\n{df_expected}

        Actual Dataset
        --------------\n{df_actual}

        Expected Types
        --------------\n{df_expected.dtypes}

        Actual Types
        ------------\n{df_actual.dtypes}

        Col for Col Match
        -----------------\n{truth_dict}
        """
        raise Exception(err_msg)

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
    # 1.5 Sort this stupid thing to enforce consistency.
    input_ids.sort()
    i = experiment_space.shape[0]
    # 2) Make the layer with make_function
    layer_function = make_function(
        function_index = index,
        function_inputs = input_ids,
        function_id = f'FAKE_ACTION_{abs(i)}',
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
            sum_of_inputs += parameter_dict["out_features"]

    # Set function-specific hyperparameters
    # TODO: Come back to this tomorrow morning.
    if atomic_constants[index] == ReLU:
        new_function['hyperparameters'] = {
            "out_features": sum_of_inputs, #TODO: Think this needs to change to 
            "in_features": sum_of_inputs,
        }
    elif atomic_constants[index] == Dropout:
        new_function['hyperparameters'] = {
            "p": .5,
            "out_features": sum_of_inputs,
            "in_features": sum_of_inputs,
        }
    elif atomic_constants[index] == Linear:
        new_function['hyperparameters'] = {
            "out_features": sum_of_inputs,
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
        {'id': 0, 'location': (0,0), 'radius': 2}
    ],
    'expected_inputs': {'1': 1, '0': 0},
    'expected_minimal_space': pd.DataFrame([
        {'id': '0', 'type': 'source', 'input': None, 'hyperparameters': {}, 'object': None},
        {'id': '1', 'type': 'source', 'input': None, 'hyperparameters': {}, 'object': None},
        {'id': 'FAKE_ACTION_101', 'type': 'intermediate', 'input': ['0', '1'], 'hyperparameters': {'out_features': 2, 'in_features': 2}, 'object': Linear},
        {'id': 'y', 'type': 'sink', 'input': ['FAKE_ACTION_101'], 'hyperparameters': {}, 'object': None}
    ]),
    'expected_graph': nn.ModuleDict({'FAKE_ACTION_101': Linear(in_features=2, out_features=12, bias=True)})
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
    'expected_inputs': {'0': 2, '1': 3},
    'expected_minimal_space': pd.DataFrame([
        {'id': '0', 'type': 'source', 'input': None, 'hyperparameters': {}, 'object': None},        
        {'id': '1', 'type': 'source', 'input': None, 'hyperparameters': {}, 'object': None},
        {'id': 'FAKE_ACTION_101', 'type': 'intermediate', 'input': ['0', '1'], 'hyperparameters': {'out_features': 12, 'in_features': 2}, 'object': Linear},
        {'id': 'FAKE_ACTION_102', 'type': 'intermediate', 'input': ['0', '1', 'FAKE_ACTION_101'], 'hyperparameters': {'out_features': 14, 'in_features': 14}, 'object': ReLU},
        {'id': 'y', 'type': 'sink', 'input': ['FAKE_ACTION_102'], 'hyperparameters': {}, 'object': None}
    ]),
    'expected_graph': nn.ModuleDict({
        'FAKE_ACTION_102': ReLU(in_features=14, out_features=14),
        'FAKE_ACTION_101': Linear(in_features=2, out_features=12, bias=True)
    })
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
    """Tests the inputs of a composed function.

    This needs a .inputs dictionary which is used to map input
    names to integer locations. This is a 'key mapping'.

    dict(input1 = 0, input2 = 2)

    Parameters
    ----------
    expected_inputs: Dict[str, int]
        The expected key mapping
    actual_inputs: Dict[str, int]
        The actual key mapping
    """
    err_msg = f"""ComposedFunction Input Mapping Error:

    The ComposedFunction stores a mapping dictionary allowing it
    to extract the correct positional indices from an input dataset.

    Expected Value
    --------------\n{expected_inputs}

    Actual Value
    ------------\n{actual_inputs}
    """
    assert expected_inputs == actual_inputs, err_msg

def minimal_subspace_tester(
    expected_space: pd.DataFrame,
    actual_space: pd.DataFrame
):
    """Tests the minimal subspace retained by a Function.

    The function maintains a small Frame that it uses to build
    the graph. It's likely not necessary to persist, but it's
    there.

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
    --------------\n{expected_space.to_dict(orient='records')}

    Actual Value
    ------------\n{actual_space}
    """
    try:
        frame_tester(expected_space, actual_space)
    except Exception as e:
        print(err_msg)
        raise e

def graph_tester(
    expected_graph: nn.ModuleDict,
    actual_graph: nn.ModuleDict
):
    """Tests the PyTorch module created.

    The Composed Function creates a ModuleDict; this checks to
    see if the *structure* for the DaG is valid, not so much the
    weight values. This also does not check class properties

    Parameters
    ----------
    expected_graph: nn.ModuleDict
        The expected graph
    actual_space: nn.ModuleDict
        The actual graph
    """
    all_modules = {}
    # 1) Check all the keys.
    err_msg = f"""ComposedFunction Graph Error:

    The ComposedFunction uses ModuleDict to store a computation
    graph. The keys for the expected and actual ModuleDict items do
    not match.

    Expected Keys
    -------------\n{set(expected_graph.keys())}

    Actual Keys
    -----------\n{set(actual_graph.keys())}
    """
    assert set(expected_graph.keys()) == set(actual_graph.keys()), err_msg
    for module_name in expected_graph.keys():
        # 2) For all the modules just check the type
        all_modules[module_name] = {
            'TypeMatch': type(
                expected_graph[module_name]
                ) == type(
                    actual_graph[module_name]
                )
        }
    all_modules = pd.DataFrame.from_dict(
        all_modules,
        orient='index'
    )
    err_msg = f"""ComposedFunction Graph Error:

    The ComposedFunction uses ModuleDict to store a computation
    graph. That computation graph did not meet the expected values.

    Expected Value
    --------------\n{expected_graph}

    Actual Value
    ------------\n{actual_graph}

    Module-by-Module Inspection
    ---------------------------\n{all_modules}

    Type Equality
    -------------\n{np.all(all_modules.TypeMatch)}
    """
    assert np.all(all_modules.TypeMatch), err_msg
    # TODO: 3. Consider doing more here.

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
        print("COMPOSED FUNCTION:\n", composed_function)
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
        inputs_tester(
            test_set['expected_inputs'],
            composed_instance.inputs
        )
        # Does the minimal subset match?
        minimal_subspace_tester(
            test_set['expected_minimal_space'],
            composed_instance._net_subspace
        )
        # Does the Torch graph match?
        graph_tester(
            test_set['expected_graph'],
            composed_instance._module_dict
        )
        # What about the forward method?
        # Let's turn the data into a torch Tensor.
        def torchify(x):
            """Turns a dataset into two torch tensors."""
            y = torch.tensor(x['y'].values)
            return torch.tensor(x.drop('y', axis=1).values), y

        X, y = torchify(test_data)
        pred = composed_instance(X)
        print(pred)




test_composed_function(test_set_1)