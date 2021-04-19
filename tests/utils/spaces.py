"""Spaces testing

TODO: Clean this up sometime; there's a lot going on in here and the
data could be collated.
"""
import numpy as np
import pandas as pd
import pytest
from mentalgym.utils.data import function_bank
from mentalgym.utils.spaces import (
    append_to_experiment,
    experiment_space_eq,
    experiment_space_from_container,
    refresh_experiment_container,
    space_to_iterable
)


metadata_df = pd.DataFrame(
    data = {
        'id': ['column_0', 'column_1', 'column_2', 'output'],
        'type': ['source', 'source', 'source', 'sink'],
        'input': [None, None, None, None]
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
    assert actual_container.equals(expected_container)


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

test_space_sets = zip(
    test_inputs,
    test_space_outputs
)


@pytest.mark.parametrize('kwargs,expected_space',test_space_sets)
def test_experiment_space_from_container(kwargs, expected_space):
    container = refresh_experiment_container(function_bank,**kwargs)
    actual_space = experiment_space_from_container(container)
    # 1: Function ID's
    id_eq = np.all(
        actual_space['function_ids']==expected_space['function_ids']
    )
    loc_eq = np.all(
        actual_space['function_locations']==expected_space['function_locations'])
    con_eq = np.all(
        actual_space['function_connections']==expected_space['function_connections'])
    assert np.all([
        id_eq,
        loc_eq,
        con_eq
    ])


test_banks = [
    function_bank,
    function_bank.assign(
        exp_loc_1=[50., 50., 50., 200., 50., 200.], # , 50., 75.
        exp_loc_2=[100., 100., 100., 300., 100., 300.] #, 100., 100.]
    ),
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
    composed_funcs = expected_container.query(
        'type=="composed"'
    )
    composed_iter = [
        row.to_dict() for
        ind, row in composed_funcs.iterrows()
    ]
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


frame_1 = pd.DataFrame(
    [
        {'id':'1', 'type': 'atomic',},
        {},
    ]
)

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


test_inputs = [
    function_bank,
    pd.DataFrame(
        data = {
            'id': ['bob','janice','dilly','dally','beans'],
            'living': [True,True,True,True,True],
            'extra': ['a','b','c','d','e'],
            'information': ['a','b','c','d','e'],
            'score_accuracy': [0.95, 0.7, 0.6, 0.5, 0.6],
            'score_complexity': [0.01, 100, 10, 20, 50]
        }
    )
]

test_outputs = [
    [
        {
            'id': 'column_0',
            'type': 'source',
            'input': None,
            'exp_loc_0': 0.0,
            'exp_loc_1': 0.0
        },
        {
            'id': 'column_1',
            'type': 'source',
            'input': None,
            'exp_loc_0': 50.0,
            'exp_loc_1': 0.0
        },
        {
            'id': 'column_2',
            'type': 'source',
            'input': None,
            'exp_loc_0': 100.0,
            'exp_loc_1': 0.0
        },
        {
            'id': 'output',
            'type': 'sink',
            'input': None,
            'exp_loc_0': 0.0,
            'exp_loc_1': 100.0
        },
        {
            'id': 'steve',
            'type': 'composed',
            'input': ['column_0'],
            'exp_loc_0': 25.0,
            'exp_loc_1': 50.0
        },
        {
            'id': 'bob',
            'type': 'composed',
            'input': ['column_0', 'column_1'],
            'exp_loc_0': 50.0,
            'exp_loc_1': 75.0
        }
    ],
    [
        {
            'id': 'bob',
            'living': True,
            'extra': 'a',
            'information': 'a',
            'score_accuracy': 0.95,
            'score_complexity': 0.01
        },
        {
            'id': 'janice',
            'living': True,
            'extra': 'b',
            'information': 'b',
            'score_accuracy': 0.7,
            'score_complexity': 100.0
        },
        {
            'id': 'dilly',
            'living': True,
            'extra': 'c',
            'information': 'c',
            'score_accuracy': 0.6,
            'score_complexity': 10.0
        },
        {
            'id': 'dally',
            'living': True,
            'extra': 'd',
            'information': 'd',
            'score_accuracy': 0.5,
            'score_complexity': 20.0
        },
        {
            'id': 'beans',
            'living': True,
            'extra': 'e',
            'information': 'e',
            'score_accuracy': 0.6,
            'score_complexity': 50.0
        }
    ]
]

test_sets = zip(test_inputs, test_outputs)

@pytest.mark.parametrize('test_input, expected_output', test_sets)
def test_space_to_iterable(test_input, expected_output):
    actual_output = space_to_iterable(test_input)
    err_msg = f"""Space to Iterable Error:
    When attempting to convert a space to an iterable the expected
    output did not match.

    Expected
    --------

    {expected_output}

    Actual
    ------

    {actual_output}
    """
    assert actual_output == expected_output, err_msg