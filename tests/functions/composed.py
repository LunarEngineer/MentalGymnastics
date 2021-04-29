"""Contains testing for Composed Functions.

Composed functions can, when given a set of input nodes from a
FunctionBank, produce a coherent PyTorch structure.

The test cases below represent instances of Directed Acyclic Graphs
and the expected output.
"""
import numpy as np
import pandas as pd
from mentalgym.constants import linear_i
from mentalgym.functions import Linear
from mentalgym.functionbank import FunctionBank
from mentalgym.utils.function import make_function
from mentalgym.utils.spaces import (
    append_to_experiment,
    get_experiment_neighbors,
    refresh_experiment_container
)
from sklearn.datasets import make_classification
from tempfile import TemporaryDirectory

# TODO: More test cases here would be extremely useful. Simply look
#   at the structure of the Experiment Space below, mimic it, then
#   add a test case with expected output.
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

####################################################################
#                        Simple Test Case 1                        #
####################################################################
# In this test case a very simple Linear Layer is produced which   #
#   takes as input two features.                                   #
####################################################################
test_set_1 = {
    'action_kwargs': {
        'location': (0,0),
        'radius': 2
    },
    'make_function_kwargs':{
        'function_index': linear_i,
        'function_object': Linear,
        'function_type': 'intermediate',
    }
}
def drop_layer(experiment_space, action_kwargs, make_function_kwargs):
    # 1) Get the input ids.
    input_ids = get_experiment_neighbors(
        experiment_space,
        **action_kwargs
    )
    layer_function = make_function(
        function_inputs = input_ids,
        function_location = action_kwargs['location'],
        **make_function_kwargs
    )
    locs = [
        x
        for x in layer_function.keys()
        if x.startswith("exp_loc")
    ]
    new_function = {
        k: v
        for k, v in layer_function.items()
        if k in experiment_space.columns + locs
    }
    experiment_space = append_to_experiment(
        experiment_space_container = experiment_space,
        function_bank = function_bank,
        composed_functions = [new_function],
    )
    return experiment_space


with TemporaryDirectory() as d:
    function_bank = FunctionBank(
        modeling_data = test_data,
        target = 'y',
        function_bank_directory = d,
        force_refresh = True
    )
    experiment_space = refresh_experiment_container(
        function_bank = function_bank,
        min_loc = np.array([0, 0]),
        max_loc = np.array([100, 100])
    )
    status_message = f"""Composed Function Test:

    Function Bank
    -------------\n{function_bank.to_df()}

    Experiment Space
    ----------------\n{experiment_space}
    """
    experiment_space = drop_layer(experiment_space, **test_set_1)
    print(experiment_space)
####################################################################
#  The example PyTorch equivalent is shown below and the resultant #
#   weight structures compared.                                    #
####################################################################