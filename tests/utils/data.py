"""Contains testing for data functions."""
import numpy as np
import pandas as pd
from mentalgym.utils.data import dataset_to_actions

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
        'input': None,
        'values': np.array([1, 2, 3], dtype = 'int64')
    },
    {
        'id': 'B',
        'type': 'source',
        'input': None,
        'values': np.array([4, 5, 6], dtype = 'int64')
    },
    {
        'id': 'C',
        'type': 'source',
        'input': None,
        'values': np.array([7, 8, 9], dtype = 'int64')
    }
]


def action_equal(a1, a2):
    """Test equality for two actions.

    Parameters
    ----------
    a1: Dict[str, Any]
        The first action
    a2: Dict[str, Any]
        The second action

    Returns
    -------
    action_equal: bool
        Whether the two actions are equal.
    """
    assert a1['id'] == a2['id']
    assert a1['type'] == a2['type']
    assert a1['input'] == a2['input']
    assert np.allclose(a1['values'],a2['values'])
    return True


def test_dataset_to_actions_w_tgt():
    actual = dataset_to_actions(static_df, target = 'A')
    for e, a in zip(expected_output, actual):
        assert action_equal(e, a)

# These two lines update the testing dataset to ensure that
#   a regular call will use the final column.
expected_output[0]['type'] = 'source'
expected_output[2]['type'] = 'sink'


def test_dataset_to_actions():
    actual = dataset_to_actions(static_df)
    for e, a in zip(expected_output, actual):
        assert action_equal(e, a)
