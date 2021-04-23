import numpy as np
import pytest
from mentalgym.utils.data import function_bank
from mentalgym.utils.spaces import (
    refresh_experiment_container,
    append_to_experiment
)
from mentalgym.utils.reward import (
    build_reward_function,
    connection_reward,
    linear_completion_reward,
    monotonic_reward
)
from typing import Any, Callable, Dict

container = refresh_experiment_container(function_bank)
composed_funcs = function_bank.query('type=="composed"')
composed_iter = [
    row.to_dict() for
    ind, row in composed_funcs.iterrows()
]
extended_container = append_to_experiment(
    container,
    function_bank,
    composed_iter
)


test_sets = [
    (
        'monotonic',
        monotonic_reward,
        {
            'experiment_space_container': extended_container,
            'function_set': composed_iter
        },
        np.array([5.27473208e-25])
    ),
    (
        'connection_1',
        connection_reward,
        {
            'experiment_space_container': extended_container,
            'function_set': composed_iter
        },
        np.array([10.])
    ),
    (
        'connection_0',
        connection_reward,
        {
            'experiment_space_container': container,
            'function_set': composed_iter
        },
        np.array([0.])
    ),
    (
        'linear_completion_0',
        linear_completion_reward,
        {
            'experiment_space_container': extended_container,
            'function_set': composed_iter,
            'score': 0
        },
        np.array([20.])
    ),
    (
        'linear_completion_95',
        linear_completion_reward,
        {
            'experiment_space_container': extended_container,
            'function_set': composed_iter,
            'score': 95
        },
        np.array([9520.])
    )
]

test_param_string = 'test_name, reward_f, kwargs, expected_reward'


@pytest.mark.parametrize(test_param_string, test_sets)
def test_reward(
    test_name: str,
    reward_f: Callable,
    kwargs: Dict[str, Any],
    expected_reward: float
):
    actual_reward = reward_f(**kwargs)
    assert np.allclose(actual_reward, expected_reward)