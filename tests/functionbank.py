"""Holds testing for the function bank.

This lays out simple test cases which can be used to test the function
bank.
"""
from collections import deque
from mentalgym import FunctionBank
from mentalgym.functions.atomic import Linear, ReLU
from sklearn.datasets import make_classification
from typing import Callable
import numpy as np
import pandas as pd
import pytest
import tempfile

from mentalgym.types import FunctionSet


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
    'init': {
        'modeling_data': make_data(
            n_features = 3,
            n_informative = 2,
            n_redundant = 1
        ),
        'target': 'y',
        'population_size': 10
    },
    'expected_set': [
        {'i': -1, 'id': '0', 'type': 'source', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': np.nan, 'hyperparameters': None},
        {'i': -1, 'id': '1', 'type': 'source', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': np.nan, 'hyperparameters': None},
        {'i': -1, 'id': '2', 'type': 'source', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': np.nan, 'hyperparameters': None},
        {'i': -1, 'id': 'y', 'type': 'sink', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': np.nan, 'hyperparameters': None},
        {'i': 0, 'id': 'Linear', 'type': 'atomic', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': Linear, 'hyperparameters': {}},
        {'i': 1, 'id': 'ReLU', 'type': 'atomic', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': ReLU, 'hyperparameters': {}}
    ],
    'query': {
        'query_str': {'i == 0'},
        'expected_results': {'i': 0, 'id': 'Linear', 'type': 'atomic', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': Linear, 'hyperparameters': {}}
    },
    
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
    },
    'expected_set': [
        {'i': -1, 'id': '0', 'type': 'sink', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': np.nan, 'hyperparameters': None},
        {'i': -1, 'id': '1', 'type': 'source', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': np.nan, 'hyperparameters': None},
        {'i': -1, 'id': '2', 'type': 'source', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': np.nan, 'hyperparameters': None},
        {'i': -1, 'id': '3', 'type': 'source', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': np.nan, 'hyperparameters': None},
        {'i': -1, 'id': 'y', 'type': 'source', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': np.nan, 'hyperparameters': None},
        {'i': 0, 'id': 'Linear', 'type': 'atomic', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': Linear, 'hyperparameters': {}},
        {'i': 1, 'id': 'ReLU', 'type': 'atomic', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': ReLU, 'hyperparameters': {}}
    ]
}

test_sets = [
    test_case_1,
    test_case_2
]

def init_tester(
    dir,
    **kwargs
) -> FunctionBank:
    """Tests init.

    This does basic sanity checking; it makes sure that the Function
    Bank is created and *at least* that the functions that are used
    by the Bank to curate the space are Callables. It also does some
    simple sanity checks to ensure that any of the class properties
    are initialized to appropriate values.
    """
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
    # When everything is said and done, return the bank
    return function_bank

def space_tester(
    function_bank: FunctionBank,
    expected_set: FunctionSet
):
    """Tests the default function bank.

    This tests to ensure, value for value, an exact recreation of the
    original data.

    Parameters
    ----------
    function_bank: FunctionBank

    expected_set
        This is the default iterable of Functions in the function
        space.
    """
    err_msg = f"""Function Space Creation Error:

    There is an error between the testing dataset and the produced
    data.

    Original Data
    -------------\n{pd.DataFrame(expected_set)}

    Actual Data
    -----------\n{pd.DataFrame(function_bank._function_manifest)}
    """
    expected_frame = pd.DataFrame(expected_set)
    actual_frame = pd.DataFrame(function_bank._function_manifest)
    assert expected_frame.equals(actual_frame), err_msg

def persistence_tester(function_bank: FunctionBank, dir: str):
    """Tests the ability of the function bank to persist data.

    This test is likely unnecessary; this *always* happens in init.
    The value recreation is already tested. This implies persistence.

    This error function is here *in case* I'm wrong; it just needs to
    be fleshed out.

    Parameters
    ----------
    function_bank: FunctionBank

    dir: str
        The directory the Function Bank writes to.
    """
    err_msg = f"""Function Space Persistence Error:

    There is an error between the expected structure and the actual
    on the physical disk.

    Original Data
    -------------\n{1}

    Actual Data
    -----------\n{1}
    """
    pass

def query_tester(function_bank, query, expected_results):
    """Tests the ability of the function bank to query data.

    There are two versions of query: the query method 

    Parameters
    ----------
    function_bank: FunctionBank

    dir: str
        The directory the Function Bank writes to.
    """
    err_msg = f"""Function Space Persistence Error:

    There is an error between the expected structure and the actual
    on the physical disk.

    Original Data
    -------------\n{1}

    Actual Data
    -----------\n{1}
    """
    pass

@pytest.mark.parametrize('inputs', test_sets)
def test_function_bank(inputs):
    """Test the FunctionBank.
    
    This is a set of integration tests.

    Parameters
    ----------
    test_sets: Dict[str, Any]
        The inputs used in testing:
            * init: Used in the init function.
            * expected_set: Used to test the function manifest.
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
        space_tester(function_bank, inputs['expected_set'])
        # 3. Check to ensure the function bank is being persisted
        #   appropriately.
        persistence_tester(function_bank, inputs['expected_dir'])
        # 4. Ensure that query is doing the right thing
        query_tester(
            function_bank,
            inputs['query']['query_str'],
            inputs['query']['expected_results']
        )
