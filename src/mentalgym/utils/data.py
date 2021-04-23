"""Contains data and data utilities for the environment."""
import pandas as pd
from mentalgym.types import FunctionSet
from mentalgym.utils.spaces import (
    append_to_experiment,
    refresh_experiment_container
)
from sklearn.datasets import make_classification
from typing import Optional
import numpy as np

####################################################################
#                  Tools for working with data                     #
####################################################################
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
    >>> dataset_to_functions(static_df, target = 'A')
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
        target = dataset.columns[-1]
    # Create an empty list
    output = []
    # Now, walk through the columns
    for col, vals in dataset.iteritems():
        # Create an action
        # TODO: Dependent on how we're serving the data
        #   the values might need to be part of the dict.
        col_dict = {
            'id': col,
            'type': 'sink' if col == target else 'source',
            'input': None
        }
        # Add it to the output list
        output.append(col_dict)
    return output

####################################################################
#                   Create simple testing data                     #
####################################################################
# This dataset will be used as a testing dataset for the Gym.
X, y = make_classification(
    n_samples=100000,
    n_features=3,
    n_informative=2,
    n_redundant=1,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=2,
    flip_y=0.01,
    class_sep=1.0,
    shift=0.0,
    scale=1.0,
    shuffle=True,
    random_state=42
)
testing_df = pd.DataFrame(
    X,
    columns=[
        "column_0",
        "column_1",
        "column_2"
    ]
).assign(output=y)

####################################################################
#       Default and Meaningless Testing Atomic Functions           #
####################################################################
function_atomic_one = {
    'i': 0,
    'id': 'ReLU',
    'type': 'atomic',
    'input': None,
    'exp_loc_0': None,
    'exp_loc_1': None,
}
function_atomic_two = {
    'i': 1,
    'id': 'Dropout',
    'type': 'atomic',
    'input': None,
    'exp_loc_0': None,
    'exp_loc_1': None,
}
atomic_functions = [
    function_atomic_one,
    function_atomic_two
]

####################################################################
#                Create simple Experiment Space                    #
####################################################################
# This base container has the input and output elements of the
# testing dataset within it.

base_container = refresh_experiment_container(
    pd.DataFrame(dataset_to_functions(testing_df))
)

####################################################################
#                Simple Composed Function Example                  #
####################################################################
# This base container has the input and output elements of the
#   testing dataset within it. These functions are added to extend
#   the space. These represent actions which would be made by agents
#   during the course of an episode.

function_composed_one = {
    'i': 2,
    'id': 'steve',
    'type': 'composed',
    'input': ['column_0'],
    'exp_loc_0': 25.,
    'exp_loc_1': 50.,
}
function_composed_two = {
    'i': 3,
    'id': 'bob',
    'type': 'composed',
    'input': ['column_0', 'column_1'],
    'exp_loc_0': 50.,
    'exp_loc_1': 75.,
}

####################################################################
#                  Create simple Function Bank                     #
####################################################################
# This default function bank has information for the input, output,
#   atomic, and composed functions. It reads the atomic from the
#   mentalgym.atomic module, reads the composed from the directory
#   within which it's instantiated, and reads the input and output
#   from the refreshed experiment container.

function_set = [
    function_composed_one,
    function_composed_two
]

function_set += atomic_functions

function_bank = append_to_experiment(
    base_container,
    pd.DataFrame(function_set),
    function_set
)
