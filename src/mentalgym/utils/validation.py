"""Contains utilities used to validate data structures."""
import pandas as pd
from typing import Any, Dict


def is_function(item: Any, raise_early: bool = False) -> bool:
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
            Expected type: Dict
            Encountered type: {type(item)}
            """
            raise Exception(err_msg)
        return False
    # 2) Check that required keys exist.
    required_keys = [
        'id',
        'type',
        'input'
    ]
    if not set(required_keys).issubset(item):
        if raise_early:
            err_msg = f"""Function Validation Error:
            Expected keys: {required_keys}
            Encountered keys: {[_ for _ in item.keys()]}
            """
            raise Exception(err_msg)
        return False
    # Well, looks like a function, smells like a function...
    return True

# TODO: Finish and write testing and docs
def validate_function_bank(function_set):
    """Validates a function bank.

    This does basic validation for a function bank.

    Parameters
    ----------
    function_set: FunctionSet
        A set of functions.

    Examples
    """
    # Test each function
    for function in function_set:
        assert is_function(function)
    # Test to ensure the basic types exist.
    assert pd.DataFrame(
        function_set
    ).types.isin(['atomic', 'composed', 'source', 'sink'])
    # There's more to be done here.
    raise NotImplementedError
