import numpy as np
import pandas as pd
import pytest
from collections import deque
from mentalgym.utils.function import (
    dataset_to_functions,
    make_id,
    make_function
)
from sklearn.datasets import make_classification


test_inputs = [
    {
        'n_features': 3,
        'n_informative': 2,
        'n_redundant': 1,
        'target': 'y'
    },
    {
        'n_features': 4,
        'n_informative': 2,
        'n_redundant': 1,
        'target': '0'
    },
    {
        'n_features': 5,
        'n_informative': 2,
        'n_redundant': 1,
        'target': '1'
    }
]

test_outputs = [
    [
        {'i': -1, 'id': '0', 'type': 'source', 'input': None, "living": True},
        {'i': -1, 'id': '1', 'type': 'source', 'input': None, "living": True},
        {'i': -1, 'id': '2', 'type': 'source', 'input': None, "living": True},
        {'i': -1, 'id': 'y', 'type': 'sink', 'input': None, "living": True}
    ],
    [
        {'i': -1, 'id': '0', 'type': 'sink', 'input': None, "living": True},
        {'i': -1, 'id': '1', 'type': 'source', 'input': None, "living": True},
        {'i': -1, 'id': '2', 'type': 'source', 'input': None, "living": True},
        {'i': -1, 'id': '3', 'type': 'source', 'input': None, "living": True},
        {'i': -1, 'id': 'y', 'type': 'source', 'input': None, "living": True}
    ],
    [
        {'i': -1, 'id': '0', 'type': 'source', 'input': None, "living": True},
        {'i': -1, 'id': '1', 'type': 'sink', 'input': None, "living": True},
        {'i': -1, 'id': '2', 'type': 'source', 'input': None, "living": True},
        {'i': -1, 'id': '3', 'type': 'source', 'input': None, "living": True},
        {'i': -1, 'id': '4', 'type': 'source', 'input': None, "living": True},
        {'i': -1, 'id': 'y', 'type': 'source', 'input': None, "living": True}
    ]
]


test_sets = zip(test_inputs, test_outputs)

@pytest.mark.parametrize('kwargs, expected_output', test_sets)
def test_dataset_to_functions(kwargs, expected_output):
    t = kwargs.pop('target')
    X, y = make_classification(**kwargs)

    actual_output = pd.DataFrame(
        dataset_to_functions(
            pd.DataFrame(X).assign(y=y).rename(
                columns = {_: str(_) for _ in range(X.shape[1])}
            ),
            target = t
        )
    )
    err_msg = f"""Function Space Query Error:

    There is an error between the expected structure and the actual
    on the physical disk.

    Original Data
    -------------\n{pd.DataFrame(expected_output)}

    Actual Data
    -----------\n{actual_output}

    """
    assert actual_output.equals(
        pd.DataFrame(expected_output)
    ), err_msg

test_inputs = [
    {'id_len': 12, 'seed': 4},
    {'id_len': 22, 'seed': 32},
    {'id_len': 6, 'seed': 9}
]

test_outputs = [
    'T62F688fCLrx',
    '2j1JAxruOQw8X7RPIZ5CLt',
    'A17rhL'
]

test_sets = zip(test_inputs, test_outputs)

@pytest.mark.parametrize('kwargs, expected_output', test_sets)
def test_make_id(kwargs, expected_output):
    assert make_id(**kwargs) == expected_output

callable_1 = lambda x: 1
callable_2 = lambda x: 2

test_inputs = [
    {
        'function_object': callable_1,
        'function_type': 'atomic',
        'function_location': (0,0),
        'max_score_len': 5,
        'seed': 1
    },
    {
        'function_object': callable_2,
        'function_type': 'intermediate',
        'function_inputs': ['a', 'b'],
        'max_score_len': 2,
        'seed': 1
    },
    {
        'function_inputs': ['d', 'e'],
        'function_location': np.array([1,2,3]),
        'max_score_len': 1,
        'seed': 1
    }
]

test_outputs = [
    {
        'i': -1,
        'id': 'DFU6ciZ6pt1AqZpzNIfb1UZHYuCWhshC',
        'type': 'atomic',
        'input': None,
        'living': True,
        'object': callable_1,
        'score_default': deque([0], maxlen=5),
        'exp_loc_0': 0,
        'exp_loc_1': 0
    },
    {
        'i': -1,
        'id': 'DFU6ciZ6pt1AqZpzNIfb1UZHYuCWhshC',
        'type': 'intermediate',
        'input': ['a', 'b'],
        'living': True,
        'object': callable_2,
        'score_default': deque([0], maxlen=2)
    },
    {
        'i': -1,
        'id': 'DFU6ciZ6pt1AqZpzNIfb1UZHYuCWhshC',
        'type': 'composed',
        'input': ['d', 'e'],
        'living': True,
        'object': None,
        'score_default': deque([0], maxlen=1),
        'exp_loc_0': 1,
        'exp_loc_1': 2,
        'exp_loc_2': 3
    }
]

test_sets = zip(test_inputs, test_outputs)

@pytest.mark.parametrize('kwargs, expected_outputs', test_sets)
def test_make_function(kwargs, expected_outputs):
    make_function(**kwargs)