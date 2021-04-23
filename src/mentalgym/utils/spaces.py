"""Contains functions for working with spaces.

This has functions for working with the experiment space, function
space, and the built in Gym spaces.
"""
import numpy as np
import pandas as pd
from mentalgym.constants import experiment_space_fields
from mentalgym.types import FunctionSet, ExperimentSpace
from mentalgym.utils.validation import is_function
from numpy.typing import ArrayLike
from typing import Callable, Optional


def refresh_experiment_container(
    function_bank: pd.DataFrame,
    min_loc: ArrayLike = np.array([0, 0]),
    max_loc: ArrayLike = np.array([100, 100])
) -> pd.DataFrame:
    """Create default experiment space from a function set.

    This creates an experiment space container from a Function Bank,
    which can be queried for Function objects. It couples that with
    the given extents of the Experiment Space to create the base set
    of functions.

    This function can be called in .reset to initialize the
    experiment space elements of the observation space. The output
    has elements for the input and output nodes.

    Parameters
    ----------
    function_bank: pd.DataFrame
        This is an object which can be queried for function information
    min_loc: ArrayLike[float] = np.array([0, 0])
        This is an array of 'minimum' locations for every axis in the
        experiment space. This (in a 2d space) would be [x_min, y_min]
    max_loc: ArrayLike[float] = np.array([100, 100])
        This is an array of 'maximum' locations for every axis in the
        experiment space. This (in a 2d space) would be [x_max, y_max]

    Returns
    -------
    experiment_space_container: pd.DataFrame
        This is a Pandas representation of the experiment space. This
        has solely the input and output functions within it.

    Examples
    --------
    >>> from mentalgym.data import function_bank
    >>> min_loc = [0, 0]
    >>> max_loc = [100, 100]
    >>> refresh_experiment_container(function_bank, min_loc, max_loc)
       i        id    type input  exp_loc_0  exp_loc_1
    0 -1  column_0  source  None        0.0        0.0
    1 -1  column_1  source  None       50.0        0.0
    2 -1  column_2  source  None      100.0        0.0
    3 -1    output    sink  None        0.0      100.0
    >>> min_loc = [0, 50, 100]
    >>> max_loc = [100, 200, 300]
       i        id    type input  exp_loc_0  exp_loc_1  exp_loc_2
    0 -1  column_0  source  None        0.0        0.0      100.0
    1 -1  column_1  source  None       50.0        0.0      100.0
    2 -1  column_2  source  None      100.0        0.0      100.0
    3 -1    output    sink  None        0.0      100.0      300.0
    """
    # This is to allow using Numpy syntax with the min and max loc
    _min_loc = np.array(min_loc)
    _max_loc = np.array(max_loc)
    # This has id, type, and input among the base keys.
    # This subsets to just those fields.
    input_functions: pd.DataFrame = function_bank.query(
        "type=='source'"
    )[['id', 'type', 'input']]
    assert input_functions.shape[0] > 0, "No input functions available."
    output_functions: pd.DataFrame = function_bank.query(
        "type=='sink'"
    )[['id', 'type', 'input']]
    assert output_functions.shape[0] > 0, "No output functions available."
    # This tests to make sure nothing silly came up.
    assert _min_loc.shape == _max_loc.shape, "Min and max location length mismatch."
    ndim = _min_loc.shape[0]
    assert np.all(_max_loc > _min_loc), "Maximum values should be greater than minimum."
    assert ndim >= 2, "Experiment dimensionality should be at least 2."
    # Define the locational fields
    loc_fields = [f'exp_loc_{_}' for _ in range(ndim)]
    # We're going to create locations for input.
    # Input nodes will be uniformly distributed over the first axis.
    # Or at least... close enough.
    num_inputs = input_functions.shape[0]
    input_0 = np.linspace(
        min_loc[0],
        max_loc[0],
        num_inputs
    )
    input_locations = np.concatenate([
        input_0.reshape(-1, 1),
        # All dimensions but the first are set to ones for the inputs
        np.ones((num_inputs, np.amax(ndim - 1, 0)))
    ], axis=1)
    # Then we set the values to be the experiment minimum along the other
    #   dimensions.
    input_locations[:, 1:] = _min_loc[1:]
    # Output nodes will also be uniformly distributed along the first axis.
    num_outputs = output_functions.shape[0]
    output_0 = np.linspace(
        min_loc[0],
        max_loc[0],
        num_outputs
    )
    # But with all *other* dimensions set to max value.
    output_locations = np.concatenate([
        output_0.reshape(-1, 1),
        # All dimensions but the first are set to ones for the inputs
        np.ones((num_outputs, np.amax(ndim - 1, 0)))
    ], axis=1)
    output_locations[:, 1:] = _max_loc[1:]

    # Now we take the DataFrame representation of the inputs and
    #   outputs, stack them vertically, and add the locations on
    #   as additional columns named exp_loc_0, ..., exp_loc_(n-1)
    location_df = pd.DataFrame(
        np.concatenate(
            [
                input_locations,
                output_locations
            ]
        ),
        columns=loc_fields
    )
    function_df = pd.concat([input_functions, output_functions])
    # This assigns a null Object column to the input.
    function_df = function_df.assign(object=None)

    # The final output has a row for every input and out node,
    #   and every node is in a proper location.
    output_df = pd.concat(
        [
            function_df, 
            location_df
        ], 
        axis=1
    )

    # This assigns a 'Function index' to the input
    output_df.loc[output_df.type.isin(["source", "sink"]), 'i'] = -1
    output_df['i'] = output_df['i'].astype('int')

    # This does a final 'reordering' simply for prettiness
    i_column = output_df.pop('i')
    output_df.insert(0, 'i', i_column)

    # Then finally do a check to ensure that the expected elements
    #   of the space are there. This is to ensure that if you are
    #   adding elements to the state space willynilly that you have
    #   to account for them.
    expected_columns = experiment_space_fields + loc_fields
    err_msg = f"""Refresh Experiment Space Container:
    Expected Fields: {expected_columns}
    Actual Fields: {output_df.columns}
    """
    assert set(expected_columns) == set(output_df.columns), err_msg

    return output_df

# TODO: This needs to enforce scraping only the existent fields in the composed function
def append_to_experiment(
    experiment_space_container: pd.DataFrame,
    function_bank: pd.DataFrame,
    composed_functions: FunctionSet
) -> pd.DataFrame:
    """Extend an experiment space container with a composed action.

    Parameters
    ----------
    experiment_space_container: pd.DataFrame
        The original experiment space container to insert new
        functions into.
    function_bank: FunctionBank
        This is the function bank, which can be queried for function
        information, and is used for validation here.
    composed_functions: FunctionSet
        This is an iterable of functions, each of which have keys
        expected to be seen in the experiment_space. These can be
        intermediate functions.

    Returns
    -------
    experiment_space_container: pd.DataFrame
        This is the original experiment space container with new and
        shiny composed functions added in.

    Examples
    --------
    >>> # Load in a testing function bank
    >>> from mentalgym.utils.data import function_bank
    >>> # Grab the utilities to create some experiment data.
    >>> from mentalgym.utils.spaces import refresh_experiment_container
    >>> container = refresh_experiment_container(function_bank)
    >>> # Now, take the composed actions and add them to the experiment
    >>> composed_funcs = function_bank.query('type=="composed"')
    >>> composed_iter = [
    ...    row.to_dict() for
    ...    ind, row in composed_funcs.iterrows()
    ... ]
    >>> append_to_experiment(
    ...     container,
    ...     function_bank,
    ...     composed_iter
    ... )
       i        id      type                 input                                 object  exp_loc_0  exp_loc_1
    0 -1  column_0    source                  None                                   None        0.0        0.0
    1 -1  column_1    source                  None                                   None       50.0        0.0
    2 -1  column_2    source                  None                                   None      100.0        0.0
    3 -1    output      sink                  None                                   None        0.0      100.0
    0  2     steve  composed            [column_0]  <function <lambda> at 0x7fb13d8b5820>       25.0       50.0
    1  3       bob  composed  [column_0, column_1]  <function <lambda> at 0x7fb13d8b5a60>       50.0       75.0
    """
    # 1) Ensure the function has the basic requirements
    for composed_function in composed_functions:
        is_function(composed_function, raise_early=True)
    # 2) Ensure the function inputs all exist in the bank
    f_inputs = pd.DataFrame(composed_functions)
    f_queried = function_bank.query('id in @f_inputs.id')
    test_mask = f_inputs.id.isin(f_queried.id)
    err_msg = f"""Composed Function Error:
    The following id's were not in the Function Bank.
    {f_inputs[~test_mask]}
    """
    assert np.all(test_mask), err_msg
    # 3) Append the composed functions onto the space.
    return experiment_space_container.append(
        f_inputs
    )


def experiment_space_eq(
    experiment_space_container_a: pd.DataFrame,
    experiment_space_container_b: pd.DataFrame
) -> bool:
    """Tests two containers for equivalency.

    Parameters
    ----------
    experiment_space_container_a: pd.DataFrame
        An experiment space container
    experiment_space_container_b: pd.DataFrame
        Another experiment space container

    Returns
    -------
    eq: bool
        Whether or not two experiment spaces are equivalent

    Examples
    --------
    >>> import pandas as pd
    >>> from mentalgym.utils.spaces import refresh_experiment_container
    >>> from mentalgym.utils.spaces import experiment_space_eq
    >>> from mentalgym.utils.data import function_bank
    >>> metadata_df = pd.DataFrame(
    ...     data = {
    ...         'id': ['column_0', 'column_1', 'column_2', 'output'],
    ...         'type': ['source', 'source', 'source', 'sink'],
    ...         'input': [None, None, None, None]
    ...     }
    ... )
    >>> space_constraints_1 = {
    ...     'min_loc': [0, 0],
    ...     'max_loc': [100, 100]
    ... }
    >>> base_container = refresh_experiment_container(
    ...     metadata_df,
    ...     **space_constraints_1
    ... )
    >>> experiment_space_eq(base_container, base_container)
    True
    >>> function_bank
             id      type                 input  exp_loc_0  exp_loc_1
    0  column_0    source                  None        0.0        0.0
    1  column_1    source                  None       50.0        0.0
    2  column_2    source                  None      100.0        0.0
    3    output      sink                  None        0.0      100.0
    0     steve  composed            [column_0]       25.0       50.0
    1       bob  composed  [column_0, column_1]       50.0       75.0
    >>> experiment_space_eq(base_container, function_bank)
    """
    cont_new_ind_a = experiment_space_container_a.reset_index(drop=True)
    cont_new_ind_b = experiment_space_container_b.reset_index(drop=True)
    cont_new_ind_a = cont_new_ind_a.sort_values(['type', 'id'])
    cont_new_ind_b = cont_new_ind_b.sort_values(['type', 'id'])
    return cont_new_ind_a.equals(cont_new_ind_b)


def space_to_iterable(
    space: pd.DataFrame
) -> FunctionSet:
    """Deconstructs a space of functions.

    Parameters
    ----------
    space: pd.DataFrame
        Either an experiment space container or a function bank.

    Returns
    -------
    function_set: FunctionSet

    Examples
    --------
    >>> import pandas as pd
    >>> from mentalgym.utils.spaces import space_to_iterable
    >>> function_space = pd.DataFrame(
    ...     data = {
    ...         'id': ['bob','janice','dilly','dally','beans'],
    ...         'living': [True,True,True,True,True],
    ...         'extra': ['a','b','c','d','e'],
    ...         'information': ['a','b','c','d','e'],
    ...         'score_accuracy': [0.95, 0.7, 0.6, 0.5, 0.6],
    ...         'score_complexity': [0.01, 100, 10, 20, 50]
    ...     }
    ... )
    >>> function_set = space_to_iterable(function_space)
    >>> function_set[0]
    {'id': 'bob', 'living': True, 'extra': 'a', 'information': 'a', 'score_accuracy': 0.95, 'score_complexity': 0.01}
    >>> function_set[1]['score_accuracy']
    0.7
    """
    return space.to_dict(orient = 'records')


def prune_function_set(
    function_set: FunctionSet,
    sampling_func: Callable,
    population_size: int,
    random_state: Optional[int] = None
) -> FunctionSet:
    """Prunes a function set

    This draws `n` samples from the function set.
    All elements not drawn are marked as 'dead' and the original,
    mutated, set is returned.

    Parameters
    ----------
    function_set: FunctionSet
        The iterable of function objects
    sampling_func: Callable
        The sampling function to use. Please see utils.sampling
        for the expected function signature.
    population_size: int
        The number of functions to escape the cull.
    random_state: Optional[int] = None
        This is passed into the sampling function.

    Returns
    -------
    curated_set: FunctionSet
        The original bank with the 'living' status updated.

    Examples
    --------
    >>> import pandas as pd
    >>> from mentalgym.utils.spaces import (
    ...     space_to_iterable,
    ...     prune_function_set
    ... )
    >>> from mentalgym.utils.sampling import softmax_score_sample
    >>> function_space = pd.DataFrame(
    ...     data = {
    ...         'id': ['bob','janice','dilly','dally','beans'],
    ...         'living': [True,True,True,True,True],
    ...         'extra': ['a','b','c','d','e'],
    ...         'information': ['a','b','c','d','e'],
    ...         'score_accuracy': [0.95, 0.7, 0.6, 0.5, 0.6],
    ...         'score_complexity': [0.01, 100, 10, 20, 50]
    ...     }
    ... )
    >>> function_set = space_to_iterable(function_space)
    >>> pruned_set = prune_function_set(
    ...     function_set,
    ...     softmax_score_sample,
    ...     3,
    ...     0
    ... )
    >>> [_['id'] for _ in pruned_set if not _['living']]
    ['dally', 'beans']
    """
    # Turn to dataframe for ease of use
    function_bank = pd.DataFrame(function_set)
    # Sample the population size.
    sampled_ids = sampling_func(
        function_bank,
        population_size,
        random_state
    )
    # Update the data.
    function_bank.loc[
        ~function_bank.id.isin(sampled_ids),
        'living'
    ] = False
    # Cast the data back to records.
    return space_to_iterable(function_bank)

def state_from_space(experiment_space_container):
    """Creates a state representation from a container.
    """
    pass