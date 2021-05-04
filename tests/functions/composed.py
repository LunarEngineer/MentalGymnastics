"""Contains testing for Composed Functions.

Composed functions can, when given a set of input nodes from a
FunctionBank, produce a coherent PyTorch structure.

The test cases below represent instances of Directed Acyclic Graphs
and the expected output.
"""
# These are two flags that can be set to *create* testing data
#  and to output verbose.
__VERBOSE__ = False
__GENERATE_DATA__ = False

import numpy as np
import os
import pandas as pd
pd.options.display.max_columns = 200
import pytest
import torch
from mentalgym.functions import Linear, ReLU, Dropout
from mentalgym.functionbank import FunctionBank
from mentalgym.functions import (
    atomic_constants,

)
from mentalgym.constants import linear_output_size, dropout_p
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
from torch import tensor
from torch import nn
from torch.nn.parameter import Parameter
from torchviz import make_dot
from typing import Dict, Optional, Type

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
    n_features = 101,
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
    df_actual: pd.DataFrame,
    extra_header_info: Optional[str] = None
):
    if not df_expected.equals(df_actual):
        truth_dict = {}
        for col in df_expected.columns:
            truth_dict[col] = df_expected[col].equals(df_actual[col])
        extra_info = pd.DataFrame(
            data = {
                'ExpectedType': df_expected.dtypes,
                'ActualType': df_actual.dtypes,
                'Column Equality Test': truth_dict
            }
        )
        if extra_header_info is None:
            extra_header_info = ""
        err_msg = f"""{extra_header_info}
        =======================================
        =      DataFrame Equality Error       =
        =======================================

        Expected Dataset
        ----------------\n{df_expected}

        Actual Dataset
        --------------\n{df_actual}

        Extra Information
        -----------------\n{extra_info}
        """
        print(err_msg)
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
    i = experiment_space.shape[0]

    # Do not add to experiment space if function is invalid (unconnected)
    if not input_ids:
        return experiment_space

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
            sum_of_inputs += parameter_dict["output_size"]

    # Set function-specific hyperparameters
    if atomic_constants[index] == ReLU:
        new_function['hyperparameters'] = {
            "output_size": sum_of_inputs,
            "input_size": sum_of_inputs,
        }
    elif atomic_constants[index] == Dropout:
        new_function['hyperparameters'] = {
            "p": 0.5,
            "output_size": sum_of_inputs,
            "input_size": sum_of_inputs,
        }
    elif atomic_constants[index] == Linear:
        new_function['hyperparameters'] = {
            "output_size": linear_output_size,
            "input_size": sum_of_inputs,
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
#   takes as input three features                                  #
####################################################################
test_set_1 = {
    'name': 'test_set_1_simple_linear',
    "actions": [{"id": 0, "location": (50, 0), "radius": 1}],
    "expected_inputs": {"49": 0, "50": 1, "51": 2},
    "expected_minimal_space": pd.DataFrame(
        [
            {
                "id": "49",
                "type": "source",
                "input": None,
                "hyperparameters": {},
                "object": None,
            },
            {
                "id": "50",
                "type": "source",
                "input": None,
                "hyperparameters": {},
                "object": None,
            },
            {
                "id": "51",
                "type": "source",
                "input": None,
                "hyperparameters": {},
                "object": None,
            },
            {
                "id": "FAKE_ACTION_102",
                "type": "intermediate",
                "input": ["49", "50", "51"],
                "hyperparameters": {
                    "output_size": linear_output_size,
                    "input_size": 3,
                },
                "object": Linear,
            },
            {
                "id": "y",
                "type": "sink",
                "input": ["FAKE_ACTION_102"],
                "hyperparameters": {},
                "object": None,
            },
        ]
    ),
    "expected_graph": nn.ModuleDict(
        {
            "FAKE_ACTION_102": Linear(
                input_size=3, output_size=linear_output_size, bias=True
            )
        }
    ),
    'expected_scores': {'score_default_count': 2.0, 'score_default_mean': 0.35, 'score_default_std': 0.49497474683058323, 'score_default_min': 0.0, 'score_default_25%': 0.175, 'score_default_50%': 0.35, 'score_default_75%': 0.5249999999999999, 'score_default_max': 0.7}
}

####################################################################
#                        Simple Test Case 2                        #
####################################################################
# Input -> Linear -> ReLU -> Sink.  There is a skip connection     #
# from the inputs to the ReLU and the ReLU                         #
# layer is closest to the sink.                                    #
####################################################################
test_set_2 = {
    "name": "test_set_2_two_layer",
    "actions": [
        {"id": 0, "location": (0, 0), "radius": 2},
        {"id": 1, "location": (0, 1), "radius": 2},
    ],
    "expected_inputs": {"0": 0, "1": 1, "2": 2},
    "expected_minimal_space": pd.DataFrame(
        [
            {
                "id": "0",
                "type": "source",
                "input": None,
                "hyperparameters": {},
                "object": None,
            },
            {
                "id": "1",
                "type": "source",
                "input": None,
                "hyperparameters": {},
                "object": None,
            },
            {
                "id": "2",
                "type": "source",
                "input": None,
                "hyperparameters": {},
                "object": None,
            },
            {
                "id": "FAKE_ACTION_102",
                "type": "intermediate",
                "input": ["0", "1", "2"],
                "hyperparameters": {
                    "output_size": linear_output_size,
                    "input_size": 3,
                },
                "object": Linear,
            },
            {
                "id": "FAKE_ACTION_103",
                "type": "intermediate",
                "input": ["0", "1", "FAKE_ACTION_102"],
                "hyperparameters": {
                    "output_size": linear_output_size + 2,
                    "input_size": linear_output_size + 2,
                },
                "object": ReLU,
            },
            {
                "id": "y",
                "type": "sink",
                "input": ["FAKE_ACTION_103"],
                "hyperparameters": {},
                "object": None,
            },
        ]
    ),
    "expected_graph": nn.ModuleDict(
        {
            "FAKE_ACTION_103": ReLU(
                input_size=linear_output_size + 2,
                output_size=linear_output_size + 2,
            ),
            "FAKE_ACTION_102": Linear(
                input_size=3, output_size=linear_output_size, bias=True
            ),
        }
    ),
    'expected_scores': {'score_default_count': 2.0, 'score_default_mean': 0.35, 'score_default_std': 0.49497474683058323, 'score_default_min': 0.0, 'score_default_25%': 0.175, 'score_default_50%': 0.35, 'score_default_75%': 0.5249999999999999, 'score_default_max': 0.7}
}

####################################################################
#                        Simple Test Case 3                        #
####################################################################
# In this test case we have Input -> Linear -> ReLU -> Dropout     #
# with a skip connection from the inputs to the RELU.              #
# The dropout layer is closest to the sink.                        #
####################################################################
test_set_3 = {
    "name": "test_set_3_three_layer",
    "actions": [
        {"id": 0, "location": (5, 0), "radius": 1},
        {"id": 1, "location": (6, 2), "radius": 3},
        {"id": 2, "location": (4, 4), "radius": 3},
    ],
    "expected_inputs": {"4": 0, "5": 1, "6": 2, "7": 3, "8": 4},
    "expected_minimal_space": pd.DataFrame(
        [
            {
                "id": "4",
                "type": "source",
                "input": None,
                "hyperparameters": {},
                "object": None,
            },
            {
                "id": "5",
                "type": "source",
                "input": None,
                "hyperparameters": {},
                "object": None,
            },
            {
                "id": "6",
                "type": "source",
                "input": None,
                "hyperparameters": {},
                "object": None,
            },
            {
                "id": "7",
                "type": "source",
                "input": None,
                "hyperparameters": {},
                "object": None,
            },
            {
                "id": "8",
                "type": "source",
                "input": None,
                "hyperparameters": {},
                "object": None,
            },
            {
                "id": "FAKE_ACTION_102",
                "type": "intermediate",
                "input": ["4", "5", "6"],
                "hyperparameters": {
                    "output_size": linear_output_size,
                    "input_size": 3,
                },
                "object": Linear,
            },
            {
                "id": "FAKE_ACTION_103",
                "type": "intermediate",
                "input": ["4", "5", "6", "7", "8", "FAKE_ACTION_102"],
                "hyperparameters": {
                    "output_size": linear_output_size + 5,
                    "input_size": linear_output_size + 5,
                },
                "object": ReLU,
            },
            {
                "id": "FAKE_ACTION_104",
                "type": "intermediate",
                "input": ["FAKE_ACTION_103"],
                "hyperparameters": {
                    "output_size": linear_output_size + 5,
                    "input_size": linear_output_size + 5,
                    "p": dropout_p,
                },
                "object": Dropout,
            },
            {
                "id": "y",
                "type": "sink",
                "input": ["FAKE_ACTION_104"],
                "hyperparameters": {},
                "object": None,
            },
        ]
    ),
    "expected_graph": nn.ModuleDict(
        {
            "FAKE_ACTION_104": Dropout(
                input_size=linear_output_size + 3,
                output_size=linear_output_size + 3,
                p = dropout_p,
            ),
            "FAKE_ACTION_103": ReLU(
                input_size=linear_output_size + 3,
                output_size=linear_output_size + 3,
            ),
            "FAKE_ACTION_102": Linear(
                input_size=3, output_size=linear_output_size, bias=True
            ),
        }
    ),
    'expected_scores': {'score_default_count': 2.0, 'score_default_mean': 0.35, 'score_default_std': 0.49497474683058323, 'score_default_min': 0.0, 'score_default_25%': 0.175, 'score_default_50%': 0.35, 'score_default_75%': 0.5249999999999999, 'score_default_max': 0.7}
}

####################################################################
#                        Simple Test Case 4                        #
####################################################################
# In this test case we have Input -> Linear -> [ReLU] -> Dropout   #
# where the ReLU has a small enough radius that it doesn't connect #
# to the Linear.  The dropout layer is closest to                  #
# the sink.                                                        #
####################################################################
test_set_4 = {
    "name": "test_set_4_ignored_node",
    "actions": [
        {"id": 0, "location": (5, 0), "radius": 1},
        {"id": 1, "location": (6, 2), "radius": 1},
        {"id": 2, "location": (5, 1), "radius": 1},
    ],
    "expected_inputs": {"4": 1, "5": 0, "6": 2},
    "expected_minimal_space": pd.DataFrame(
        [
            {
                "id": "4",
                "type": "source",
                "input": None,
                "hyperparameters": {},
                "object": None,
            },
            {
                "id": "5",
                "type": "source",
                "input": None,
                "hyperparameters": {},
                "object": None,
            },
            {
                "id": "6",
                "type": "source",
                "input": None,
                "hyperparameters": {},
                "object": None,
            },
            {
                "id": "FAKE_ACTION_102",
                "type": "intermediate",
                "input": ["4", "5", "6"],
                "hyperparameters": {
                    "output_size": linear_output_size,
                    "input_size": 3,
                },
                "object": Linear,
            },
            {
                "id": "FAKE_ACTION_103",
                "type": "intermediate",
                "input": ["5", "FAKE_ACTION_102"],
                "hyperparameters": {
                    "output_size": linear_output_size + 1,
                    "input_size": linear_output_size + 1,
                    "p": dropout_p,
                },
                "object": Dropout,
            },
            {
                "id": "y",
                "type": "sink",
                "input": ["FAKE_ACTION_103"],
                "hyperparameters": {},
                "object": None,
            },
        ]
    ),
    "expected_graph": nn.ModuleDict(
        {
            "FAKE_ACTION_103": Dropout(
                input_size=linear_output_size + 1,
                output_size=linear_output_size + 1,
                p = dropout_p,
            ),
            "FAKE_ACTION_102": Linear(
                input_size=3, output_size=linear_output_size, bias=True
            ),
        }
    ),
    'expected_scores': {'score_default_count': 2.0, 'score_default_mean': 0.35, 'score_default_std': 0.49497474683058323, 'score_default_min': 0.0, 'score_default_25%': 0.175, 'score_default_50%': 0.35, 'score_default_75%': 0.5249999999999999, 'score_default_max': 0.7}
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

    """
    frame_tester(expected_space, actual_space, err_msg)

def graph_tester(
    expected_graph: nn.ModuleDict,
    actual_graph: nn.ModuleDict
):
    """Tests the PyTorch module created.

    The Composed Function creates a ModuleDict; this checks to
    see if the *structure* for the DaG is as expected.

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

def weight_tester(
    composed,
    test_name
):
    """Tests the weights for a Composed Function.

    Parameters
    ----------
    composed: ComposedFunction
        A composed function instance
    test_name:
        The name of the test set.
    """
    tensor_dir = os.path.join(
        os.path.dirname(__file__),
        'expected_params'
    )
    # Create and persist the testing data.
    # GENERATE_DATA: Uncomment this section to make the tensor.
    tensor_matching = {}
    for (param_name, param) in composed.named_parameters():
        tensor_path = os.path.join(
            tensor_dir,
            f'{test_name}_{param_name}.pt'
        )
        if __GENERATE_DATA__:
            torch.save(param, tensor_path)
        t = torch.load(tensor_path)
        tensor_matching[param_name] = torch.all(t.eq(param))
    tensor_match_status = pd.DataFrame(data={'Matches Expected': pd.Series(tensor_matching)})
    err_msg = f"""Weight Testing Error: {test_name}

    When attempting to match expected weights there was a disparity:

    Matching Status
    ---------------\n{tensor_match_status}
    """
    assert np.all(tensor_match_status.iloc[:,0]), err_msg

def forward_tester(
    composed_function: ComposedFunction,
    input: torch.Tensor,
    test_name: str
):
    """Tests the composed function's forward.

    This calls the recursive forward for the composed function.

    Parameters
    ----------
    composed_function: ComposedFunction
        The composed function object created in the tests.
    input: torch.Tensor
        The input dataset.
    test_name: str
        The name of the test set, also used as a unique filename.
    """
    # Call forward
    actual_forward = composed_function(input)
    # Specify filepaths
    graph_path = os.path.join(
        os.path.dirname(__file__),
        'expected_graphs',
        f'{test_name}.graph'
    )
    tensor_path = os.path.join(
        os.path.dirname(__file__),
        'expected_graphs',
        f'{test_name}.pt'
    )
    # Write out the graph viz image.
    graph_dot = make_dot(actual_forward)
    with open(graph_path, 'w') as f:
        f.write(graph_dot.source)
    # Create and persist the testing data.
    # GENERATE_DATA: Uncomment this section to make the tensor.
    if __GENERATE_DATA__:
        torch.save(actual_forward, tensor_path)
    # Load the testing data.
    expected_tensor = torch.load(tensor_path)
    # Calculate the difference.
    diff = expected_tensor - actual_forward
    err_msg = f"""ComposedFunction Forward Error:

    When calling the Composed Function's forward on the input
    a disparity exists between the forward and actual values..

    Difference Between Expected And Actual
    --------------------------------------\n{diff}
    """
    assert diff.sum() < 1e-6, err_msg

def composed_append_tester(
    function_bank,
    composed_function
) -> ComposedFunction:
    """Tests FunctionBank to append/return Composed function.

    This sticks the composed function into the FunctionBank and
    tries to query for that composed, returning it.

    Parameters
    ----------
    composed_function: ComposedFunction
        The composed function object created in the tests.
    input: torch.Tensor
        The input dataset.
    test_name: str
        The name of the test set, also used as a unique filename.

    Returns
    -------
    newly_queried_composed: ComposedFunction
        This is the function returned by the FunctionBank
    """
    composed_id = composed_function['id']
    function_bank.append(composed_function)
    new_composed = function_bank.query(f'id=="{composed_id}"').iloc[0].to_dict()
    # Here we test for equality for the composed and new composed.
    new_instance = new_composed['object'](**new_composed['hyperparameters'])
    # If we've gotten this far then we've pulled the item from
    #   the function bank and we can check to see if it's ok.
    err_msg = """Composed Function Append Error:

    When checking the {} field there was a discrepancy between the
    original value and the value returned by the Function Bank.

    Original Value
    --------------\n{}

    Value Returned by Function Bank
    -------------------------------\n{}
    """
    # 1. Check the IDs
    assert composed_function['id'] == new_composed['id'], err_msg.format(
        'id',
        composed_function['id'],
        new_composed['id']
    )
    # 2. Check the i
    assert composed_function['i'] == new_composed['i'], err_msg.format(
        'id',
        composed_function['i'],
        new_composed['i']
    )
    # 3. Check the type
    assert composed_function['type'] == new_composed['type'], err_msg.format(
        'type',
        composed_function['type'],
        new_composed['type']
    )
    # 4. Check the input (nan/None)
    if composed_function['input'] is None or np.isnan(composed_function['input']):
        boolstatement = new_composed['input'] is None or np.isnan(new_composed['input'])
        assert boolstatement, err_msg.format(
            'type',
            composed_function['type'],
            new_composed['type']
        )
    else:
        assert composed_function['input'] == new_composed['input'], err_msg.format(
            'input',
            composed_function['input'],
            new_composed['input']
        )
    # 5. Check living
    assert composed_function['living'] == new_composed['living'], err_msg.format(
        'living',
        composed_function['living'],
        new_composed['living']
    )
    # 6. Check object and hyperparameters
    assert Type[composed_function['object']] == Type[new_composed['object']], err_msg.format(
        'object',
        composed_function['object'],
        new_composed['object']
    )
    assert composed_function['hyperparameters'] == new_composed['hyperparameters'], err_msg.format(
        'hyperparameters',
        composed_function['hyperparameters'],
        new_composed['hyperparameters']
    )
    return new_instance


def score_tester(
    function_bank,
    ids,
    score,
    expected_scores
):
    function_bank.score(
        function_set = ids,
        score = score
    )
    actual_scores = function_bank.function_statistics(
        ids = ids
    ).drop('id', axis=1).iloc[0].to_dict()
    err_msg = f"""Composed Function Scoring Error:

    When scoring the composed function the values did not match
    expectations.

    Expected Values
    ---------------\n{expected_scores}

    Actual Values
    -------------\n{actual_scores}
    """
    assert actual_scores == expected_scores, err_msg

def place_composed_tester(
    function_bank,
    experiment_space,
    actions,
    composed_id,
    test_name
):
    """Drop a composed layer and use in a net.
    
    Parameters
    ----------
    function_bank: FunctionBank
        This has the composed function.
    experiment_space: ExperimentSpace
        A *fresh* experiment space.
    actions: ActionSet
        A set of actions to take after placing the composed node.
    composed_id: str
        The ID of the composed function to place.
    test_name: str
        The name of the test routine
    """
    # Go fetch, and place, the composed function into the space.
    composed_func = function_bank.query(
        f'id=="{composed_id}"'
    ).iloc[0].to_dict()
    # Take the location of the first action and *nudge* it just a wee bit.
    action_loc = actions[0]['location']
    locs = {
        f'exp_loc_{i}': action_loc[i] + 0.001 for i in range(len(action_loc))
    }
    composed_func.update(locs)
    # Now drop that into the space.
    experiment_space = append_to_experiment(
        experiment_space_container = experiment_space,
        function_bank = function_bank,
        composed_functions = [composed_func],
    )
    # Replay all the actions taken previously, but accounting
    #   for a composed function placed prior.
    for action in actions:
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
    # Then make that a new function.
    composed_function = make_function(
        function_index = function_bank.idxmax() + 1,
        function_type = 'composed',
        function_object = ComposedFunction,
        function_hyperparameters = {}
    )
    composed_instance = ComposedFunction(
        id = composed_function['id'],
        experiment_space = experiment_space,
        function_bank = function_bank
    )
    weight_tester(composed_instance,test_name)


####################################################################
#                       Integration Testing                        #
####################################################################
# This test routine tests much of the internal functionality of the#
#   composed function while it asserts that the output is a        #
#   runnable PyTorch layer.                                        #
####################################################################
test_sets = [
    test_set_1,
    test_set_2,
    test_set_3,
    test_set_4
]


@pytest.mark.parametrize('test_set', test_sets)
def test_composed_function(test_set):
    # Ensure reproducible results
    torch.manual_seed(0)
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
        if __VERBOSE__: print(status_message)
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
        # Now we are going to create a Composed Function from these
        #   actions.
        composed_function = make_function(
            function_index = function_bank.idxmax() + 1,
            function_type = 'composed',
            function_id = test_set['name'] + '_01',
            function_object = ComposedFunction,
            function_hyperparameters = {}
        )
        status_message = f"""

        Experiment Space Posterior to Actions
        -------------------------------------\n{experiment_space}

        Composed Function Created
        -------------------------\n{composed_function}
        """
        if __VERBOSE__:
            print(status_message)
        # This, when it's called for the first time, builds
        #   a net and assigns it to forward.
        composed_instance = ComposedFunction(
            id = composed_function['id'],
            experiment_space = experiment_space,
            function_bank = function_bank,
            verbose = __VERBOSE__
        )
        extra_keys = {
            'input_size': composed_instance.input_size,
            'output_size': composed_instance.output_size,
            'function_dir': composed_instance._function_dir,
        }
        composed_function['hyperparameters'].update(extra_keys)
        # At this point we have the first test. Do the initialized
        #   properties match?
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
        # What about the weights?
        weight_tester(
            composed_instance,
            test_set['name'] + '_prior'
        )
        # What about the forward method?
        # Let's turn the data into a torch Tensor.
        def torchify(x):
            """Turns a dataset into two torch tensors."""
            y = torch.tensor(x['y'].values)
            return torch.tensor(x.drop('y', axis=1).values), y

        X, y = torchify(test_data)
        forward_tester(
            composed_instance,
            X,
            test_set['name']
        )

        # Awesome, now that it's been used for forward, let's
        #   append it to the function bank.
        composed_instance_new = composed_append_tester(
            function_bank,
            composed_function
        )
        # Now, we're going to score this function.
        # This actually tests the statistics reporting at the same time.
        # This *also* returns the copied object.
        score_tester(
            function_bank = function_bank,
            ids = list(composed_instance.input.keys()) + [composed_function["id"]],
            score = .7,
            expected_scores = test_set['expected_scores']
        )
        
        # This test calls forward on the composed new
        forward_tester(
            composed_instance_new,
            X,
            test_set['name'] + '_reload'
        )

        # Now, we simulate a second episode. In this episode we refresh
        #   the container
        experiment_space = refresh_experiment_container(
            function_bank = function_bank,
            min_loc = np.array([0, 0]),
            max_loc = np.array([100, 100])
        )
        # Finally, what happens when this Composed Function is used
        #   in a new episode?
        place_composed_tester(
            function_bank= function_bank,
            experiment_space=experiment_space,
            actions = test_set['actions'],
            composed_id = composed_function["id"],
            test_name = test_set['name'] + 'posterior'
        )








test_composed_function(test_set_1)
test_composed_function(test_set_2)
test_composed_function(test_set_3)
test_composed_function(test_set_4)