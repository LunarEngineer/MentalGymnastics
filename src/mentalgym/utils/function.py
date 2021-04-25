import numpy as np
import pandas as pd
from collections import deque
from mentalgym.types import Function, FunctionSet
from mentalgym.functions.atomic._atomic import AtomicFunction
from mentalgym.functions.composed import ComposedFunction
from numpy.random import default_rng
from numpy.typing import ArrayLike
from typing import Any, Callable, Dict, Iterable, Optional, Type, Union

FunctionObject = Union[Type[AtomicFunction], Type[ComposedFunction]]

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
    for col, _ in dataset.iteritems():
        # Create an action
        # TODO: Dependent on how we're serving the data
        #   the values might need to be part of the dict.
        col_dict = {
            'i': -1,
            'id': col,
            'type': 'sink' if col == target else 'source',
            'input': None,
            'living': True
        }
        # Add it to the output list
        output.append(col_dict)
    return output


def make_id(
    id_len: int = 32,
    seed: Optional[int] = None
) -> str:
    """Create a random ID for a Function.

    # TODO: Easter Egg here with random name generation?

    This will create a random `id_len` long string of alphanumeric
    characters.

    Parameters
    ----------
    id_len: int = 32
        The length of the random string.

    Returns
    -------
    random_str: str
        A `id_len` long random string.
    """
    rng = default_rng(seed)
    alphabet = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    id = "".join(
        rng.choice(alphabet, id_len)
    )
    return id


def make_function(
    function_index: Optional[int] = -1,
    function_id: Optional[str] = None,
    function_object: Optional[FunctionObject] = None,
    function_hyperparameters: Optional[Dict[str, Any]] = None,
    function_type: str = 'composed',
    function_inputs: Optional[Iterable[str]] = None,
    function_location: Optional[ArrayLike] = None,
    max_score_len: Optional[int] = 100,
    seed: Optional[int] = None
) -> Function:
    """Creates a Function representation.

    This creates a representation of a Function and is used to
    create a standardized data structure which the gym uses to
    represent a complex action composed of a series of nested
    functions. Note that functions can have scores *added* by the 
    environment

    Parameters
    ----------
    function_index: Optional[int] = -1
        This represents the index location within the function bank.
    function_id: Optional[str] = None
        An optional 'name' representation for the function.
    function_object: Optional[Callable] = None
        An object, representing a function, that exposes a forward,
        in addition to saving and loading weights.
    function_hyperparameters: Optional[Dict[str, Any]] = None
        If this is passed then this will be used as the hyperparameters
        for the function object when it is instantiated.
    function_type: str = 'composed'
        A value in the set {'composed','atomic','input','output}
    function_inputs: Optional[Iterable[str]] = None
        This is a list of input function id's which this function
        expects as input.
    function_location: Optional[ArrayLike] = None
        If this is passed then these fields are added to the function
        and each one is keyed as 'exp_loc_i'.
        This is used in the Experiment Space, not in the Function
        Bank.
    max_score_length: Optional[int] = 100
        This creates a deque to hold scores for *this* function.

    Returns
    -------
    function_representation: Function
        This is a Function object, currently a dictionary with a
        predetermined keyset.
    """
    # Set some defaults
    if function_id is None:
        function_id = make_id(seed=seed)
    if function_hyperparameters is None:
        function_hyperparameters = {}

    function_representation = {
        'i': function_index,
        'id': function_id,
        'type': function_type,
        'input': function_inputs,
        'living': True,
        'object': function_object,
        'hyperparameters': function_hyperparameters,
        'score_default': deque([0],maxlen=max_score_len) #This is to allow sampling with all functions.
    }
    if function_location is not None:
        function_representation.update(
            {
                f'exp_loc_{i}': x for i, x
                in enumerate(function_location)
            }
        )
    return function_representation
