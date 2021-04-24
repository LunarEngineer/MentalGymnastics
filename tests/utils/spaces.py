"""Spaces testing

TODO: Clean this up sometime; there's a lot going on in here and the
data could be collated.
"""
import numpy as np
import pandas as pd
import pytest
from mentalgym.functions import AtomicFunction
from mentalgym.utils.data import function_bank
from mentalgym.utils.spaces import (
    append_to_experiment,
    experiment_space_eq,
    prune_function_set,
    refresh_experiment_container,
    build_default_function_space
)
from mentalgym.utils.function import (
    dataset_to_functions,
)
from mentalgym.utils.sampling import softmax_score_sample
from sklearn.datasets import make_classification

metadata_df = pd.DataFrame(
    data = {
        'i': [-1, -1, -1, -1],
        'id': ['column_0', 'column_1', 'column_2', 'output'],
        'type': ['source', 'source', 'source', 'sink'],
        'input': [None, None, None, None],
        'object': [None, None, None, None]
    }
)

test_inputs = [
    {
        'min_loc': [0, 0],
        'max_loc': [100, 100]
    },
    {
        'min_loc': [0, 50, 100],
        'max_loc': [100, 200, 300]
    }
]

test_container_outputs = [
    {
        'exp_loc_0': [0., 50., 100., 0.],
        'exp_loc_1': [0., 0., 0., 100.]
    },
    {
        'exp_loc_0': [0., 50., 100., 0.],
        'exp_loc_1': [50., 50., 50., 200.],
        'exp_loc_2': [100., 100., 100., 300.]
    }
]

test_container_sets = zip(
    test_inputs,
    test_container_outputs
)

@pytest.mark.parametrize('kwargs,locations',test_container_sets)
def test_refresh_experiment_container(kwargs,locations):
    """Tests refreshing the experiment space"""
    actual_container = refresh_experiment_container(function_bank,**kwargs)
    expected_container = pd.concat(
        [
            metadata_df,
            pd.DataFrame(locations)
        ],
        axis=1
    )
    err_msg = f"""
    Expected Values
    ---------------
    {expected_container}

    Actual Values
    -------------
    {actual_container}
    """
    assert actual_container.equals(expected_container), err_msg


test_space_outputs = [
    {
        'function_ids': np.array(['column_0', 'column_1', 'column_2', 'output'], dtype='object'),
        'function_locations': np.array([[0., 0.], [50., 0.], [100., 0.], [0., 100.]]),
        'function_connections': np.array([False, False, False, False])
    },
    {
        'function_ids': np.array(['column_0', 'column_1', 'column_2', 'output'], dtype='object'),
        'function_locations': np.array([[0., 50., 100.], [50., 50., 100.], [100., 50., 100.], [0., 200., 300.]]),
        'function_connections': np.array([False, False, False, False])
    }
]


test_banks = [
    function_bank,
    function_bank.assign(
        exp_loc_1=[50., 50., 50., 200., 50., 200., 50., 75.],
        exp_loc_2=[100., 100., 100., 300., 100., 300., 100., 100.]
    )[[
        'i', 'id', 'type', 'input', 'object', 'exp_loc_0',
        'exp_loc_1', 'exp_loc_2', 'living', 'score_default'
    ]]
]
test_append_sets = zip(
    test_inputs,
    test_banks
)

@pytest.mark.parametrize(
    'kwargs, expected_container',
    test_append_sets
)
def test_append_to_experiment(kwargs, expected_container):
    container = refresh_experiment_container(expected_container, **kwargs)
    # Have sets of composed nodes here.
    composed_iter = expected_container.query(
        'type in ["composed","atomic"]'
    ).to_dict(orient = 'records')
    actual_container = append_to_experiment(
        experiment_space_container = container,
        function_bank = expected_container,
        composed_functions = composed_iter
    )
    err_msg = f"""Frame Validation Error:
    -------
    Actual:
    -------
    {actual_container}

    ---------
    Expected:
    ---------
    {expected_container}
    """
    assert experiment_space_eq(
        actual_container,
        expected_container
    ), err_msg


eq_sets = [
    (
        refresh_experiment_container(
            metadata_df,
            **test_inputs[0]
        ),
        refresh_experiment_container(
            metadata_df,
            **test_inputs[0]
        ),
        True
    ),
    (
        refresh_experiment_container(
            metadata_df,
            **test_inputs[1]
        ),
        refresh_experiment_container(
            metadata_df,
            **test_inputs[1]
        ),
        True
    ),
    (
        refresh_experiment_container(
            metadata_df,
            **test_inputs[0]
        ),
        refresh_experiment_container(
            metadata_df,
            **test_inputs[1]
        ),
        False
    )
]


@pytest.mark.parametrize('a,b,result', eq_sets)
def test_experiment_space_eq(a,b,result):
    err_msg = f"""Experiment Space Equality Error:

    a == b should be {result}

    -----
    - a -
    -----

    {a}

    -----
    - b -
    -----

    {b}
    """
    assert experiment_space_eq(a, b) == result, err_msg


function_space = pd.DataFrame(
    data = {
        'id': ['bob','janice','dilly','dally','beans'],
        'living': [True,True,True,True,True],
        'extra': ['a','b','c','d','e'],
        'information': ['a','b','c','d','e'],
        'score_accuracy': [0.95, 0.7, 0.6, 0.5, 0.6],
        'score_complexity': [0.01, 100, 10, 20, 50]
    }
).to_dict(orient = 'records')


test_sets = [
    (
        {
        'function_set': function_space,
        'sampling_func': softmax_score_sample,
        'population_size': 3,
        'random_state': 4
        },
        [
            {'id': 'bob', 'living': True, 'extra': 'a', 'information': 'a', 'score_accuracy': 0.95, 'score_complexity': 0.01},
            {'id': 'janice', 'living': True, 'extra': 'b', 'information': 'b', 'score_accuracy': 0.7, 'score_complexity': 100.0},
            {'id': 'dilly', 'living': False, 'extra': 'c', 'information': 'c', 'score_accuracy': 0.6, 'score_complexity': 10.0},
            {'id': 'dally', 'living': False, 'extra': 'd', 'information': 'd', 'score_accuracy': 0.5, 'score_complexity': 20.0},
            {'id': 'beans', 'living': True, 'extra': 'e', 'information': 'e', 'score_accuracy': 0.6, 'score_complexity': 50.0}]
    )
]
@pytest.mark.parametrize('kwargs, expected_results', test_sets)
def test_prune_function_set(kwargs, expected_results):
    actual_results = prune_function_set(**kwargs)
    assert actual_results==expected_results


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
        {
            'id': '0',
            'type': 'source',
            'input': None,
            'i': np.nan,
            'living': np.nan,
            'object': np.nan,
            'score_default': np.nan
        },
        {
            'id': '1',
            'type': 'source',
            'input': None,
            'i': np.nan,
            'living': np.nan,
            'object': np.nan,
            'score_default': np.nan
        },
        {
            'id': '2',
            'type': 'source',
            'input': None,
            'i': np.nan,
            'living': np.nan,
            'object': np.nan,
            'score_default': np.nan
        },
        {
            'id': 'y',
            'type': 'sink',
            'input': None,
            'i': np.nan,
            'living': np.nan,
            'object': np.nan,
            'score_default': np.nan
        },
        {
            'id': 'ReLU',
            'type': 'atomic',
            'input': None,
            'i': -2.0,
            'living': True,
            'object': AtomicFunction(),
            'score_default': deque([0], maxlen=100)
        },
        {
            'id': 'Dropout',
            'type': 'atomic',
            'input': None,
            'i': -2.0,
            'living': True,
            'object': AtomicFunction(),
            'score_default': deque([0], maxlen=100)
        }
    ],
    [
        {'id': '0', 'type': 'sink', 'input': None},
        {'id': '1', 'type': 'source', 'input': None},
        {'id': '2', 'type': 'source', 'input': None},
        {'id': '3', 'type': 'source', 'input': None},
        {'id': 'y', 'type': 'source', 'input': None}
    ],
    [
        {'id': '0', 'type': 'source', 'input': None},
        {'id': '1', 'type': 'sink', 'input': None},
        {'id': '2', 'type': 'source', 'input': None},
        {'id': '3', 'type': 'source', 'input': None},
        {'id': '4', 'type': 'source', 'input': None},
        {'id': 'y', 'type': 'source', 'input': None}
    ]
]

test_sets = zip(test_inputs, test_outputs)

@pytest.mark.parametrize('kwargs, expected_output', test_sets)
def test_build_default_function_space(kwargs, expected_output):
    t = kwargs.pop('target')
    X, y = make_classification(**kwargs)
    df = pd.DataFrame(
        build_default_function_space(
            pd.DataFrame(X).assign(y=y).rename(
                columns = {_: str(_) for _ in range(X.shape[1])}
            ),
            target = t
        )
    )







