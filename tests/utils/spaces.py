import pandas as pd
import pytest
from mentalgym.utils.data import function_bank
from mentalgym.utils.spaces import refresh_experiment_container


metadata_df = pd.DataFrame(
    data = {
        'id': ['column_0', 'column_1', 'column_2', 'output'],
        'type': ['source', 'source', 'source', 'sink'],
        'input': [None, None, None, None]
    }
)

test_sets = [
    (
        {
            'min_loc': [0, 0],
            'max_loc': [100, 100]
        },
        {
            'exp_loc_0': [0., 50., 100., 0.],
            'exp_loc_1': [0., 0., 0., 100.]
        }
    ),
    (
        {
            'min_loc': [0, 50, 100],
            'max_loc': [100, 200, 300]
        },
        {
            'exp_loc_0': [0., 50., 100., 0.],
            'exp_loc_1': [50., 50., 50., 200.],
            'exp_loc_2': [100., 100., 100., 300.]
        }
    )
]

@pytest.mark.parametrize('input,locations',test_sets)
def test_refresh_experiment_container():
    """Tests refreshing the experiment space"""
    actual_container = refresh_experiment_container(function_bank)
    expected_container = pd.concat(
        [
            metadata_df,
            pd.DataFrame(locations)
        ],
        axis=1
    )