"""Contains data and data utilities for the environment."""
import pandas as pd
from sklearn.datasets import make_classification
from typing import Any, Dict, Iterable, Optional
####################################################################
#                   Create simple testing data                     #
####################################################################
# This dataset will be used as a testing dataset for the Gym.
X, y = make_classification(
    n_samples = 100000,
    n_features = 4,
    n_informative = 2,
    n_redundant = 2,
    n_repeated = 0,
    n_classes = 2,
    n_clusters_per_class = 2,
    flip_y=0.01,
    class_sep=1.0,
    shift=0.0,
    scale=1.0,
    shuffle=True,
    random_state=42
)
testing_df = pd.DataFrame(
    X,
    columns = [
        "A",
        "B",
        "C",
        "D"
    ]
).assign(Y=y)
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
    'id': 'column_2',
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

###
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


def dataset_to_functions(
    dataset: pd.DataFrame,
    target: Optional[str] = None
) -> FunctionSet:
    """Convert Pandas data to a function set.

    Parameters
    ----------
    df: pandas.DataFrame
        A modeling dataset.
    target: Optional[str] = None
        If left blank this will assume the final column is the target.

    Returns
    -------
    input_actions: FunctionSet
        An iterable of dictionaries which represent input and output
        nodes.

    Examples
    --------
    >>> import pandas as pd
    >>> static_df = pd.DataFrame(
    >>>     data = {
    >>>         'A': [1, 2, 3],
    >>>         'B': [4, 5, 6],
    >>>         'C': [7, 8, 9]
    >>>     }
    >>> )
    >>> dataset_to_actions(static_df, target = 'A')
    [{'id': 'A',
      'type': 'sink',
      'input': None},
     {'id': 'B',
      'type': 'source',
      'input': None},
     {'id': 'C',
      'type': 'source',
      'input': None}]
    """
    # Set a default target if none available
    if target is None:
        target = static_df.columns[-1]
    # Create an empty list
    output = []
    # Now, walk through the columns
    for col, vals in static_df.iteritems():
        # Create an action
        col_dict = {
            'id': col,
            'type': 'sink' if col == target else 'source',
            'input': None
        }
        # Add it to the output list
        output.append(col_dict)
    return output
