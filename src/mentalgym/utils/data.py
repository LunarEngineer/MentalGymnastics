"""Contains data for testing utilities in the environment."""
import pandas as pd
from sklearn.datasets import make_classification

####################################################################
#                   Create simple testing data                     #
####################################################################
# This file creates simple testing data.

####################################################################
#                Create simple Experiment Space                    #
####################################################################
function_input_one = {
    'id': 'column_0',
    'type': 'source',
    'input': None
}
function_input_two = {
    'id': 'column_1',
    'type': 'source',
    'input': None
}
function_input_three = {
    'id': 'column_1',
    'type': 'source',
    'input': None
}
function_output = {
    'id': 'output',
    'type': 'sink',
    'input': None
}
# The second set of actions are *composed* actions
# These are created by agents during episodes.
function_composed_one = {
    'id': 'steve',
    'type': 'composed',
    'input': ['column_0']
}
function_composed_two = {
    'id': 'bob',
    'type': 'composed',
    'input': ['column_0', 'column_1']
}

function_set = [
    function_input_one,
    function_input_two,
    function_input_three,
    function_output,
    function_composed_one,
    function_composed_two
]

function_bank = pd.DataFrame(function_set)

simple_experiment_space = {
    "function_id": ["column_0", "column_1", "column_3", "output"],
    "function_location": np.array([[0, 0], [0, 1], [0, 2], [1, 1]]),
    ""
}

min_loc = [0, 0]
max_loc = [100, 100]
buffer = 0.1

# This is a testing function bank.
test_function_bank = pd.DataFrame(
    data = {
        'type': ['composed', 'input', 'input', 'input', 'output'],
        'random': ['1', '2', '3', '4', '5']
    }
)

# This is what the Experiment Space should look like on the far side
test_experiment_space = pd.DataFrame(
    data = {
        'type': ['composed', 'input', 'input', 'input', 'output'],
        'random': ['1', '2', '3', '4', '5']
    }
)