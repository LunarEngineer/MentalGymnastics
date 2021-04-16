"""Contains functions for working with spaces.

This has functions for working with the experiment space, function
space, and the built in Gym spaces.
"""
from mentalgym.typing import Function, FunctionSet
from mentalgym.utils.validation import is_function
from mentalgym.functionbank import FunctionBank
from numpy.typing import ArrayLike

def refresh_experiment_container(
    function_bank: FunctionBank,
    min_loc: ArrayLike = np.array([0, 0]),
    max_loc: ArrayLike = np.array([100, 100])
) -> pd.DataFrame:
    """Create default experiment space from a function set.

    This creates an experiment space container from a Function Bank,
    which can be queried for Function objects. It couples that with
    the given extents of the Experiment Space to create the base set
    of functions.

    This function can be called in .reset to initialize the state.

    Parameters
    ----------
    function_bank: FunctionBank
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
             id    type input  exp_loc_0  exp_loc_1
    0  column_0  source  None        0.0        0.0
    1  column_1  source  None       50.0        0.0
    2  column_2  source  None      100.0        0.0
    3    output    sink  None        0.0      100.0
    >>> min_loc = [0, 50, 100]
    >>> max_loc = [100, 200, 300]
             id    type input  exp_loc_0  exp_loc_1  exp_loc_2
    0  column_0  source  None        0.0       50.0      100.0
    1  column_1  source  None       50.0       50.0      100.0
    2  column_2  source  None      100.0       50.0      100.0
    3    output    sink  None        0.0      200.0      300.0
    """
    # This is to allow using Numpy syntax with the min and max loc
    _min_loc = np.array(min_loc)
    _max_loc = np.array(max_loc)
    # This has id, type, and input among the base keys.
    input_functions: pd.DataFrame = function_bank.query("type=='source'")
    assert input_functions.shape[0] > 0, "No input functions available."
    output_functions: pd.DataFrame = function_bank.query("type=='sink'")
    assert output_functions.shape[0] > 0, "No output functions available."
    # This tests to make sure nothing silly came up.
    assert _min_loc.shape == _max_loc.shape, "Min and max location length mismatch."
    ndim = _min_loc.shape[0]
    assert np.all(_max_loc > _min_loc), "Maximum values should be greater than minimum."
    assert ndim >= 2, "Experiment dimensionality should be at least 2."
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
        input_0.reshape(-1,1),
        # All dimensions but the first are set to ones for the inputs 
        np.ones((num_inputs, np.amax(ndim - 1, 0)))
    ], axis = 1)
    # Then we set the values to be the experiment minimum along the other
    #   dimensions.
    input_locations[:,1:] = _min_loc[1:]
    # Output nodes will also be uniformly distributed along the first axis.
    num_outputs = output_functions.shape[0]
    output_0 = np.linspace(
        min_loc[0],
        max_loc[0],
        num_outputs
    )
    # But with all *other* dimensions set to max value.
    output_locations = np.concatenate([
        output_0.reshape(-1,1),
        # All dimensions but the first are set to ones for the inputs 
        np.ones((num_outputs, np.amax(ndim - 1, 0)))
    ], axis = 1)
    output_locations[:,1:] = _max_loc[1:]
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
        columns = [f'exp_loc_{_}' for _ in range(ndim)]
    )
    function_df = pd.concat([input_functions, output_functions])
    # The final output has a row for every input and out node,
    #   and every node is in a proper location.
    output_df = pd.concat([function_df, location_df], axis=1)
    return output_df

def experiment_space_from_container(
    container: pd.DataFrame
) -> ExperimentSpace:
    """Collapses a container to an Experiment Space.

    The gym requires a specific format for the observation space.
    This function enforces that format.

    Parameters
    ----------
    container: ExperimentSpaceContainer
        This is a Pandas DataFrame with specific fields.

    Returns
    -------
    experiment_space: ExperimentSpace
        This is in the form the gym will accept.

    Examples
    --------
    >>> from mentalgym.data import function_bank
    >>> min_loc = [0, 0]
    >>> max_loc = [100, 100]
    >>> container = refresh_experiment_container(function_bank, min_loc, max_loc)
    >>> cont = experiment_space_from_container(container)
    >>> for k, v in cont.items():
    ...     print((k,v))
    ...
    ('function_ids', array(['column_0', 'column_1', 'column_2', 'output'], dtype=object))
    ('function_locations', array([[  0.,   0.],
           [ 50.,   0.],
           [100.,   0.],
           [  0., 100.]]))
    ('function_connections', array([False, False, False, False]))
    >>> function_composed_one = {
    ...    'id': 'steve',
    ...    'type': 'composed',
    ...    'input': ['column_0'],
    ...    'location': array([[ 0., 0.2]])
    ... }
    """
    ids = container.id
    location_columns = [
        _
        for _
        in container.columns
        if _.startswith('exp_loc_')
    ]
    locations = container[location_columns]
    # A *connection* is made if a node has another node
    #   as input.
    connections = container.input.apply(lambda x: x is not None)
    # Stuff it all into the output dictionary
    experiment_space = {
        'function_ids': ids.values,
        'function_locations': locations.values,
        'function_connections': connections.values
    }
    return experiment_space

def append_to_experiment(
    experiment_space_container: pd.DataFrame,
    composed_function: Function,
    function_bank: FunctionBank
) -> pd.DataFrame:
    """Extend an experiment space container with a composed action.

    Parameters
    ----------
    experiment_space_container: pd.DataFrame
        This is the original experiment space container with a new
        and shiny composed action added in.
    composed_function: Function
        This is the function representation which has keys of id,
        type, input, and location.
    function_bank: FunctionBank
        This is the function bank, which can be queried for function
        information, and is used for validation here.
    """
    # 1) Ensure the function has the basic requirements
    # TODO: Consider reworking this is raise errors instead.
    assert is_function(composed_function), "{composed_function} is invalid."
    # 2) Ensure the function inputs all exist in the bank
    f_inputs = pd.Series(composed_function['input'])
    f_queried = function_bank.query(f'id in {f_inputs}')
    assert np.all(f_inputs.isin(f_queried)), "Missing input for function."
    # 3) 'DataFrameify' the function and append it to the container.
    raise NotImplementedError
    # TODO: Fix *location* information for inputs.
    return experiment_space_container.append(
        pd.DataFrame.from_dict(composed_function)
    )
    # This needs to take a dictionary function, add it
    #  it as a row to Pandas Dataframe. I was going to use
    #   .append()
    # experiment_space_container.append(pd.DataFrame.from_dict(composed_function))
    