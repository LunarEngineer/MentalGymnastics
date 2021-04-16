"""Contains testing for data functions."""
import numpy as np
import pandas as pd
import pytest
from mentalgym.utils.data import dataset_to_functions

static_df = pd.DataFrame(
    data = {
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    }
)

expected_output = [
    {
        'id': 'A',
        'type': 'sink',
        'input': None
    },
    {
        'id': 'B',
        'type': 'source',
        'input': None
    },
    {
        'id': 'C',
        'type': 'source',
        'input': None
    }
]


def function_equal(a1, a2):
    """Test equality for two functions.

    Parameters
    ----------
    a1: Dict[str, Any]
        The first function
    a2: Dict[str, Any]
        The second function

    Returns
    -------
    function_equal: bool
        Whether the two functions are equal.
    """
    assert a1['id'] == a2['id']
    assert a1['type'] == a2['type']
    assert a1['input'] == a2['input']
    return True


def test_dataset_to_functions_w_tgt():
    actual = dataset_to_functions(static_df, target = 'A')
    for e, a in zip(expected_output, actual):
        assert function_equal(e, a)


def test_dataset_to_functions():
    # These two lines update the testing dataset to ensure that
    #   a regular call will use the final column.
    expected_output[0]['type'] = 'source'
    expected_output[2]['type'] = 'sink'
    actual = dataset_to_functions(static_df)
    for e, a in zip(expected_output, actual):
        assert function_equal(e, a)
