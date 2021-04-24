"""Contains utilities used to validate data structures."""
import numpy as np
import pandas as pd
from mentalgym.constants import (
    function_types,
    experiment_space_fields
)
from typing import Any, Dict, Iterable


def is_function(
    item: Any,
    raise_early: bool = False,
    extended: bool = False
) -> bool:
    """Tests for function-ness.

    Parameters
    ----------
    item: Any
        This can be anything. This routine will determine if the
        thing is a function.

    raise_early: bool
        This will raise meaningful errors for non-valid functions.

    Returns
    -------
    is_function: bool

    Examples
    --------
    >>> name = "Bob"
    >>> is_function(name)
    False
    >>> is_function({'name': "Bob"})
    False
    >>> composed_function = {
    ...     'id': 1,
    ...     'type': 'composed',
    ...     'input': ['column_0']
    ... }
    >>> is_function(composed_function)
    True
    >>> stupid_function = {
    ...     'type': 'composed',
    ...     'input': ['column_0']
    ... }
    >>> is_function(stupid_function)
    False
    >>> is_function(stupid_function, True)
    Traceback (most recent call last):
    ...
    Exception: Function Validation Error:
                Expected keys: ['id', 'type', 'input']
                Encountered keys: ['type', 'input']
    >>> is_function(1, True)
    Traceback (most recent call last):
    ...
    Exception: Function Validation Error:
                Expected type: Dict
                Encountered type: <class 'int'>
    """
    # 1) Check for type
    if not isinstance(item, Dict):
        if raise_early:
            err_msg = f"""Function Validation Error:
            Expected data type: Dict
            Encountered data type: {type(item)}
            """
            raise Exception(err_msg)
        return False
    # 2) Check that required keys exist.
    if not set(experiment_space_fields).issubset(item):
        if raise_early:
            err_msg = f"""Function Validation Error:
            The passed function cannot be represented in the experiment space.

            Expected keys: {experiment_space_fields}
            Encountered keys: {[_ for _ in item.keys()]}
            """
            raise Exception(err_msg)
        return False
    # 3) Assert that the types are in the allowable subset
    if not item['type'] in function_types:
        if raise_early:
            err_msg = f"""Function Validation Error:
            Expected values for type variable: {function_types}
            Encountered value for type variable: {item['type']}
            """
            raise Exception(err_msg)
        return False
    # Well, looks like a function, smells like a function...
    if extended:
        # This is testing for the function bank.
        raise NotImplementedError
    return True

# TODO: Finish and write testing and docs
def validate_function_set(function_set):
    """Validates a function set.

    This does basic validation for a function set.

    Parameters
    ----------
    function_set: FunctionSet
        An iterable of Function objects.

    Examples
    """
    # Test to ensure that all functions are Functions
    # This is as long as the function set
    is_functions: Iterable[bool] = pd.Series(
        map(
            is_function,
            function_set
        )
    )
    # If *any* of them are invalid.
    if np.any(~is_functions):
        unmasked_elements = [
            str(f) for f, _ in zip(function_set, is_functions)
            if not _
        ]
        sub_str = "\n".join(unmasked_elements)
        err_msg = f"""Invalid Functions:

        The following functions are invalid.

        {sub_str}
        """
        raise Exception(err_msg)
    # There's more to be done here, but this will work for now.
