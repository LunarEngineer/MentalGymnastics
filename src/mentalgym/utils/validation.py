"""Contains utilities used to validate data structures."""
from typing import Any, Dict
from mentalgym.types import FunctionBank


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

def validate_function_bank(function_manifest: Dict):
    """Validates a function bank"""
    raise NotImplementedError