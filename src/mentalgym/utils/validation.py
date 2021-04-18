"""Contains utilities used to validate data structures."""
import pandas as pd
from typing import Any, Dict


def is_function(item: Any) -> bool:
    """Tests for function-ness

    Parameters
    ----------
    item: Any
        This can be anything. This routine will determine if the
        thing is a function.

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
    ...     'input' ['column_0']
    ... }
    >>> is_function(composed_function)
    True
    """
    # 1) Check for type
    if not isinstance(item, Dict):
        return False
    # 2) Check that required keys exist.
    required_keys = [
        'id',
        'type',
        'input'
    ]
    if not set(required_keys).issubset(item):
        return False
    # Well, looks like a function, smells like a function...
    return True


def validate_function_bank(function_set):
    """Validates a function bank.

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
    ).types.isin(['atomic', 'source', 'sink'])
