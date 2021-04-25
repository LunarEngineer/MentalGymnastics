"""Holds testing for the function bank.

This lays out simple test cases which can be used to test the function
bank.
"""
from typing import Callable
import pandas as pd
import pytest
import tempfile
from mentalgym import FunctionBank
from sklearn.datasets import make_classification


err_msg_header = "Function Bank:"

def make_data(**kwargs):
    X, y = make_classification(
        **kwargs
    )
    df = pd.DataFrame(
        X
    ).assign(y=y)
    df.columns = [str(_) for _ in df.columns]
    return df


test_case_1 = {
    'dir': '',
    'init': {
        'modeling_data': make_data(
            n_features = 3,
            n_informative = 2,
            n_redundant = 1
        ),
        'target': 'y',
        'population_size': 10
    }
}
test_case_2 = {
    'init': {
        'modeling_data': make_data(
            n_features = 4,
            n_informative = 2,
            n_redundant = 1
        ),
        'target': '0',
        'population_size': 5
    }
}

test_sets = [
    test_case_1,
    test_case_2
]

def init_tester(
    dir,
    **kwargs
) -> FunctionBank:
    function_bank = FunctionBank(
        **kwargs,
        function_bank_directory = dir
    )

    err_msg = f"""Function Bank Init Error:
    _dataset_scraper_function method is not callable.
    """
    assert isinstance(
        function_bank._dataset_scraper_function,
        Callable
    ), err_msg

    err_msg = f"""Function Bank Init Error:
    _sampling_function method is not callable.
    """
    assert isinstance(
        function_bank._sampling_function,
        Callable
    ), err_msg

    err_msg = f"""Function Bank Init Error:
    _pruning_function method is not callable.
    """
    assert isinstance(
        function_bank._pruning_function,
        Callable
    ), err_msg

    err_msg = f"""Function Bank Init Error:
    _function_bank_directory property not set correctly.
    """
    assert function_bank._function_bank_directory == dir

    err_msg = f"""Function Bank Init Error:
    _population_size property not set correctly.
    """
    mask = function_bank._population_size == kwargs['population_size']
    assert mask, err_msg

    err_msg = f"""Function Bank Init Error:
    _function_id property not set correctly.
    """
    mask = function_bank._function_id == 0
    assert mask, err_msg

    #_function_manifest

@pytest.mark.parametrize('inputs', test_sets)
def test_function_bank(inputs):
    """Test the FunctionBank.
    
    This is a set of integration tests.

    Parameters
    ----------

    """

    # Create a temporary directory so everything goes away afterwards
    # Spin up a new action bank using the temp directory.
    with tempfile.TemporaryDirectory() as d:
        # 1. Check to ensure the init functioned correctly
        function_bank = init_tester(
            d,
            **inputs['init']
        )
        # 2. Check to ensure that the default set of actions
        #   are created
        # default = function_bank._query('type=="atomic"')
        # err_msg = """Action Bank Init Error:
        # The default set of actions was created incorrectly.
        # Expected value: {}
        # Actual value: {}
        # """
        # assert ab._action_manifest == default
