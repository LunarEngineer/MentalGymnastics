import numpy as np
import pandas as pd
import pytest
from mentalgym.utils.data import function_bank
from mentalgym.utils.spaces import refresh_experiment_container, experiment_space_from_container
from typing import Dict


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



def test_experiment_space_eq()

@pytest.mark.parametrize('kwargs',test_inputs)
def test_append_to_experiment():
    container = refresh_experiment_container(function_bank,**kwargs)
    # Have sets of composed nodes here.
    composed_funcs = function_bank.query('type=="composed"')
    composed_iter = [
        row.to_dict() for
        ind, row in composed_funcs.iterrows()
    ]
    actual_container = append_to_experiment(
        container,
        composed_iter,
        function_bank
    ).sort_values(['type','id'])
    expected_container = function_bank.sort_values(['type','id'])
    assert actual_container.equals(expected_container)