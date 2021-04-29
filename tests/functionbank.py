"""Holds testing for the function bank.

This lays out simple test cases which can be used to test the function
bank.
"""
from collections import deque

from numpy.random import default_rng
from mentalgym import FunctionBank
from mentalgym.constants import intermediate_i
from mentalgym.utils.function import make_function
from mentalgym.utils.validation import function_eq
from mentalgym.functions.atomic import Linear, ReLU, Dropout
from sklearn.datasets import make_classification
from typing import Callable
import numpy as np
import pandas as pd
import pytest
import tempfile

from mentalgym.types import FunctionSet
pd.options.display.max_columns = 100

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
        {'i': 1, 'id': 'ReLU', 'type': 'atomic', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': ReLU, 'hyperparameters': {}},
        {'i': 2, 'id': 'Dropout', 'type': 'atomic', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': Dropout, 'hyperparameters': {}}
    ],
    'query': {
        'query_str': 'i == 0',
        'expected_results': {'i': 0, 'id': 'Linear', 'type': 'atomic', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': Linear, 'hyperparameters': {}}
    },
    'expected_dir': None,
    'append': {
        "composed_set": [
            make_function(
                function_index = intermediate_i,
                function_id = 'steve',
                function_inputs = ['1'],
                function_type = 'intermediate',
                function_location = [1, 1]
            ),
            make_function(
                function_index = intermediate_i,
                function_id = 'bob',
                function_inputs = ['1', '2'],
                function_type = 'intermediate',
                function_location = [1, 1, 2]
            )
        ],
        'expected_results': pd.DataFrame(
            {
                'i': [-1, -1, -1, -1, 0, 1, 2, 3, 4],
                'id': ['0', '1', '2', 'y', 'Linear', 'ReLU', 'Dropout', 'steve', 'bob'],
                'type': ['source', 'source', 'source', 'sink', 'atomic', 'atomic', 'atomic', 'composed', 'composed'],
                'input': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, ['1'], ['1', '2']],
                'living': [True, True, True, True, True, True, True, True, True],
                'score_default': [deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100)],
                'object': [np.nan, np.nan, np.nan, np.nan, Linear, ReLU, Dropout, None, None],
                'hyperparameters': [None, None, None, None, {}, {}, {}, {}, {}]
            }
        )
    },
    'idxmax': 4,
    'sample': {
        'kwargs': {
            "n": 1,
            "include_base": False,
            "random_state": 0
        },
        'expected_sample': pd.DataFrame([
            {'i': 4, 'id': 'bob', 'type': 'composed', 'input': ['1', '2'], 'living': True, 'score_default': deque([0], maxlen=100), 'object': None, 'hyperparameters': {}}
        ])
    },
    'score': {
        'kwargs': {
            'function_set': ['ReLU', 'Dropout', 'steve', 'bob'],
            'score': 0.9
        },
        'expected_results': pd.DataFrame(
            {
                'i': [-1, -1, -1, -1, 0, 1, 2, 3, 4],
                'id': ['0', '1', '2', 'y', 'Linear', 'ReLU', 'Dropout', 'steve', 'bob'],
                'type': ['source', 'source', 'source', 'sink', 'atomic', 'atomic', 'atomic', 'composed', 'composed'],
                'input': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, ['1'], ['1', '2']],
                'living': [True, True, True, True, True, True, True, True, True],
                'score_default': [deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0, .9], maxlen=100), deque([0, .9], maxlen=100), deque([0, .9], maxlen=100), deque([0, .9], maxlen=100)],
                'object': [np.nan, np.nan, np.nan, np.nan, Linear, ReLU, Dropout, None, None],
                'hyperparameters': [None, None, None, None, {}, {}, {}, {}, {}]
            }
        )
    },
    'prune': {
        'seed': 0,
        'expected_results': []
    },
    'function_statistics':{
        'kwargs': {
            'ids': None
        },
        'expected_shape': (9, 25),
        'expected_results': pd.DataFrame([
            {'score_default_count': 1.0, 'score_default_std': np.nan, 'score_accuracy_count': 25.0, 'score_accuracy_std': 0.3441, 'score_complexity_count': 25.0, 'score_complexity_std': 30.5512},
            {'score_default_count': 1.0, 'score_default_std': np.nan, 'score_accuracy_count': 15.0, 'score_accuracy_std': 0.3009, 'score_complexity_count': 15.0, 'score_complexity_std': 34.087},
            {'score_default_count': 1.0, 'score_default_std': np.nan, 'score_accuracy_count': 20.0, 'score_accuracy_std': 0.3413, 'score_complexity_count': 20.0, 'score_complexity_std': 32.7882},
            {'score_default_count': 1.0, 'score_default_std': np.nan, 'score_accuracy_count': 1.0, 'score_accuracy_std': np.nan, 'score_complexity_count': 1.0, 'score_complexity_std': np.nan},
            {'score_default_count': 1.0, 'score_default_std': np.nan, 'score_accuracy_count': 24.0, 'score_accuracy_std': 0.3105, 'score_complexity_count': 24.0, 'score_complexity_std': 30.6097},
            {'score_default_count': 2.0, 'score_default_std': 0.6364, 'score_accuracy_count': 24.0, 'score_accuracy_std': 0.3063, 'score_complexity_count': 24.0, 'score_complexity_std': 32.382},
            {'score_default_count': 2.0, 'score_default_std': 0.6364, 'score_accuracy_count': 27.0, 'score_accuracy_std': 0.2755, 'score_complexity_count': 27.0, 'score_complexity_std': 31.6421},
            {'score_default_count': 2.0, 'score_default_std': 0.6364, 'score_accuracy_count': 27.0, 'score_accuracy_std': 0.3158, 'score_complexity_count': 27.0, 'score_complexity_std': 31.4006},
            {'score_default_count': 2.0, 'score_default_std': 0.6364, 'score_accuracy_count': 31.0, 'score_accuracy_std': 0.3196, 'score_complexity_count': 31.0, 'score_complexity_std': 30.1129}
        ])
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
        'population_size': 1
    },
    'expected_set': [
        {'i': -1, 'id': '0', 'type': 'sink', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': np.nan, 'hyperparameters': None},
        {'i': -1, 'id': '1', 'type': 'source', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': np.nan, 'hyperparameters': None},
        {'i': -1, 'id': '2', 'type': 'source', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': np.nan, 'hyperparameters': None},
        {'i': -1, 'id': '3', 'type': 'source', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': np.nan, 'hyperparameters': None},
        {'i': -1, 'id': 'y', 'type': 'source', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': np.nan, 'hyperparameters': None},
        {'i': 0, 'id': 'Linear', 'type': 'atomic', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': Linear, 'hyperparameters': {}},
        {'i': 1, 'id': 'ReLU', 'type': 'atomic', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': ReLU, 'hyperparameters': {}},
        {'i': 2, 'id': 'Dropout', 'type': 'atomic', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': Dropout, 'hyperparameters': {}}
    ],
    'expected_dir': None,
    'query': {
        'query_str': 'i == 1',
        'expected_results': {'i': 1, 'id': 'ReLU', 'type': 'atomic', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': ReLU, 'hyperparameters': {}}
    },
    'expected_dir': None,
    'append': {
        "composed_set": [
            make_function(
                function_index = intermediate_i,
                function_id = 'steve',
                function_inputs = ['1'],
                function_type = 'intermediate',
                function_location = [1, 1]
            ),
            make_function(
                function_index = intermediate_i,
                function_id = 'bob',
                function_inputs = ['1', '2'],
                function_type = 'intermediate',
                function_location = [1, 1, 2]
            )
        ],
        'expected_results': pd.DataFrame(
            {
                'i': [-1, -1, -1, -1, -1, 0, 1, 2, 3, 4],
                'id': ['0', '1', '2', '3', 'y', 'Linear', 'ReLU', 'Dropout', 'steve', 'bob'],
                'type': ['sink', 'source', 'source', 'source', 'source', 'atomic', 'atomic', 'atomic', 'composed', 'composed'],
                'input': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, ['1'], ['1', '2']],
                'living': [True, True, True, True, True, True, True, True, True, True],
                'score_default': [deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100)],
                'object': [np.nan, np.nan, np.nan, np.nan, np.nan, Linear, ReLU, Dropout, None, None],
                'hyperparameters': [None, None, None, None, None, {}, {}, {}, {}, {}]
            }
        )
    },
    'idxmax': 4,
    'sample': {
        'kwargs': {
            "n": 2,
            "include_base": True,
            "random_state": 0
        },
        'expected_sample': pd.DataFrame([
            {'i': 0, 'id': 'Linear', 'type': 'atomic', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': Linear, 'hyperparameters': {}},
            {'i': 1, 'id': 'ReLU', 'type': 'atomic', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': ReLU, 'hyperparameters': {}},
            {'i': 2, 'id': 'Dropout', 'type': 'atomic', 'input': np.nan, 'living': True, 'score_default': deque([0], maxlen=100), 'object': Dropout, 'hyperparameters': {}},
            {'i': 3, 'id': 'steve', 'type': 'composed', 'input': ['1'], 'living': True, 'score_default': deque([0], maxlen=100), 'object': None, 'hyperparameters': {}},
            {'i': 4, 'id': 'bob', 'type': 'composed', 'input': ['1', '2'], 'living': True, 'score_default': deque([0], maxlen=100), 'object': None, 'hyperparameters': {}}
        ])
    },
    'score': {
        'kwargs': {
            'function_set': ['1', 'bob'],
            'score': [0.9, 200], # Note these scores are positive increasing with magnitude of statistic. "Bigger is assumed better"
            'score_name': ['accuracy', 'complexity']
        },
        'expected_results': pd.DataFrame(
            {
                'i': [-1, -1, -1, -1, -1, 0, 1, 2, 3, 4],
                'id': ['0', '1', '2', '3', 'y', 'Linear', 'ReLU', 'Dropout', 'steve', 'bob'],
                'type': ['sink', 'source', 'source', 'source', 'source', 'atomic', 'atomic', 'atomic', 'composed', 'composed'],
                'input': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, ['1'], ['1', '2']],
                'living': [True, True, True, True, True, True, True, True, True, True],
                'score_default': [deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100)],
                'score_accuracy': [deque([0], maxlen=100), deque([0, 0.9], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0, .9], maxlen=100)],
                'score_complexity': [deque([0], maxlen=100), deque([0, 200], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0], maxlen=100), deque([0, 200], maxlen=100)],
                'object': [np.nan, np.nan, np.nan, np.nan, np.nan, Linear, ReLU, Dropout, None, None],
                'hyperparameters': [None, None, None, None, None, {}, {}, {}, {}, {}]
            }
        )
    },
    'prune': {
        'seed': 0,
        'expected_results': ['steve']
    },
    'function_statistics':{
        'kwargs': {
            'ids': ['1', 'bob'],
            'extended': True
        },
        'expected_shape': (2, 325),
        'expected_results': pd.DataFrame([
            {'score_default_count': 1.0, 'score_default_std': np.nan, 'score_accuracy_count': 24.0, 'score_accuracy_std': 0.3442, 'score_complexity_count': 24.0, 'score_complexity_std': 43.8367},
            {'score_default_count': 1.0, 'score_default_std': np.nan, 'score_accuracy_count': 30.0, 'score_accuracy_std': 0.3114, 'score_complexity_count': 30.0, 'score_complexity_std': 39.2997}
        ])
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
    # Sampling
    err_msg = f"""Function Bank Init Error:
    _sampling_function method is not callable.
    """
    assert isinstance(
        function_bank._sampling_function,
        Callable
    ), err_msg
    # pruning
    err_msg = f"""Function Bank Init Error:
    _pruning_function method is not callable.
    """
    assert isinstance(
        function_bank._pruning_function,
        Callable
    ), err_msg
    # dir
    err_msg = f"""Function Bank Init Error:
    _function_bank_directory property not set correctly.
    """
    assert function_bank._function_bank_directory == dir
    # pop size
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
    -------------\n{function_bank.to_df()}

    Actual Data
    -----------\n{1}
    """
    pass

def query_tester(function_bank, query_str, expected_results):
    """Tests the ability of the function bank to query data.

    Uses function_eq 

    Parameters
    ----------
    function_bank: FunctionBank
        This is the Function Bank object, which can be queried.
    query_str: str
        The query to feed into the frame.
    expected_results: Dict[str, Any]
        The dictionary representation of the expected results.
    """
    actual_results = function_bank.query(
        query_str
    ).to_dict(
        orient='records'
    )[0]
    err_msg = f"""Function Space Query Error:

    There is an error between the expected structure and the actual
    on the physical disk.

    Original Data
    -------------\n{pd.DataFrame([expected_results])}

    Query
    -----\n{query_str}

    Actual Data
    -----------\n{pd.DataFrame([actual_results])}

    """
    if not function_eq(expected_results, actual_results,raise_early=True):
        raise Exception(err_msg)

def append_tester(
    function_bank: FunctionBank,
    function_set: FunctionSet,
    expected_results
) -> FunctionBank:
    """Tests the Function Bank append method.

    Parameters
    ----------
    function_bank: FunctionBank
        This is the Function Bank object, which can be queried.
    kwargs: Dict[str, Any]
        The keyword arguments 
    expected_results: Dict[str, Any]
        The dictionary representation of the expected results.
    """
    function_bank.append(function_set)
    actual_df = pd.DataFrame(function_bank._function_manifest)
    expected_df = pd.DataFrame(expected_results)
    col_disp = ''
    for c in actual_df:
        if not actual_df[c].equals(expected_df[c]):
            col_disp += f'\n\tColumn: {c}\nActual: {actual_df[c].values}\nExpected: {expected_df[c].values}'
    err_msg = f"""Function Space Append Error:

    The expected bank, post append, did not match the actual bank.

    Original Data
    -------------\n{pd.DataFrame(expected_results)}

    Actual Data
    -----------\n{actual_df}

    Column Disparities
    ------------------\n{col_disp}
    """
    assert actual_df.equals(pd.DataFrame(expected_results)), err_msg

def idxmax_tester(function_bank, expected_imax):
    """Tests the idxmax method in the Function Bank.

    Parameters
    ----------
    function_bank: FunctionBank
        This is the Function Bank object, which can be queried.
    expected_imax: int
        The expected maximum index.
    """
    err_msg = f"""Function Space .idxmax() Error:

    The expected bank maximum index did not match the expected value.

    Expected Index
    --------------\n{expected_imax}

    Actual Index
    ------------\n{function_bank.idxmax()}
    """
    assert function_bank.idxmax() == expected_imax, err_msg

def sample_tester(function_bank, kwargs, expected_results):
    """Tests the *default* sampling method in the Function Bank.

    Parameters
    ----------
    function_bank: FunctionBank
        This is the Function Bank object, which can be queried.
    kwargs: Dict[str, Any]
        This is the dictionary inputs.
    expected_results: Dict[str, Any]
        The dictionary representation of the expected results.
    """
    actual_results = function_bank.sample(**kwargs)
    actual_df = pd.DataFrame(actual_results)
    expected_df = pd.DataFrame(expected_results)
    err_msg = f"""Function Space Sample Error:

    When sampling the function bank there was a disparity between
    expected and actual functions returned.

    Keyword Arguments
    -----------------\n{kwargs}

    Expected Sample
    ---------------\n{expected_df}

    Actual Sample
    -------------\n{actual_df}

    """
    # The values have already been tested, this simply needs to
    #   ensure that the correct set of ID's were returned.
    if not expected_df.equals(actual_df):
        raise Exception(err_msg)


def score_tester(function_bank, kwargs, expected_results):
    """Tests the Function Bank score method.

    Parameters
    ----------
    function_bank: FunctionBank
        This is the Function Bank object, which can be queried.
    kwargs: Dict[ str, Any]
        The keyword arguments for score.
    expected_results: Dict[str, Any]
        The dictionary representation of the expected results.
    """
    function_bank.score(**kwargs)
    err_msg = f"""Function Space Scoring Error:

    When updating scoring information the expected output differed
    from the actual.

    Scoring Arguments
    -----------------\n{kwargs}

    Expected Results
    -----------------\n{pd.DataFrame(expected_results)}

    Actual Results
    --------------\n{function_bank.to_df()}

    """
    assert function_bank.to_df().equals(expected_results), err_msg

def prune_tester(function_bank, seed, expected_results):
    """Tests the *default* pruning method in the Function Bank.

    Parameters
    ----------
    function_bank: FunctionBank
        This is the Function Bank object, which can be queried.
    seed: int
        This is a random seed used in the simulation below.
    expected_results: Dict[str, Any]
        The dictionary representation of the expected results.
    """
    # This needs to simulate 100 scored episodes.
    # This will not add new functions.
    rng = default_rng(seed)
    score_names = ['accuracy', 'complexity']
    ids = rng.choice(
        function_bank.to_df().query('type != "sink"').id,
        size=(100,2)
    )
    score_acc = rng.random(size=(100, 1))
    score_comp = rng.integers(low=0, high=100, size=(100, 1))
    scores = np.concatenate([score_acc, score_comp],axis=1)
    zipped = zip(ids,scores)
    for function_set, score_arr in zipped:
        function_bank.score(
            function_set = list(function_set),
            score = score_arr,
            score_name = score_names
        )
    function_bank.prune(random_state = seed)
    actual_results = function_bank.to_df().query(
        'living==False'
    ).id.to_list()
    err_msg = f"""FunctionBank Prune Error:

    When updating scoring information the expected output differed
    from the actual. These below are the expected and actual pruned
    composed functions.

    Random Seed
    -----------\n{seed}

    Expected Results
    -----------------\n{expected_results}

    Actual Results
    --------------\n{actual_results}

    """
    assert set(expected_results) == set(actual_results), err_msg

def function_statistics_tester(function_bank, kwargs, expected_shape, expected_results):
    """Tests the statistics generation.

    Parameters
    ----------
    function_bank: FunctionBank
        This is the Function Bank object, which can be queried.
    kwargs: Dict[str, Any]
        Keyword arguments for function_statistics
    expected_results: pd.DataFrame
        The expected statistics.
    """
    actual_results = function_bank.function_statistics(**kwargs)
    # The dataset we're going to test 
    # 1. What's the dataset shape? Is it the expected shape?
    err_msg = f"""FunctionBank Statistics Error:

    When calculating statistics the dataset shape was not as
    expected.

    Expected Shape
    --------------\n{expected_shape}

    Actual Shape
    ------------\n{actual_results.shape}

    """
    assert actual_results.shape == expected_shape, err_msg
    actual_results = actual_results[[
        'score_default_count',
        'score_default_std',
        'score_accuracy_count',
        'score_accuracy_std',
        'score_complexity_count',
        'score_complexity_std'
    ]].round(4)
    err_msg = f"""FunctionBank Statistics Error:

    When updating scoring information the expected output differed
    from the actual. These below are the expected and actual results
    of calling function_statistics on the function bank.

    Expected Results
    -----------------\n{expected_results}

    Actual Results
    --------------\n{actual_results.to_dict(orient='records')}

    """
    assert expected_results.equals(actual_results), err_msg

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
        # 5. Ensure that append is doing the right thing.
        append_tester(
            function_bank = function_bank,
            function_set = inputs['append']['composed_set'],
            expected_results = inputs['append']['expected_results']
        )
        # 6. Ensure that idxmax is doing the right thing.
        idxmax_tester(
            function_bank = function_bank,
            expected_imax = inputs['idxmax']
        )
        # 7. Ensure that sample is doing the right thing.
        sample_tester(
            function_bank = function_bank,
            kwargs = inputs['sample']['kwargs'],
            expected_results = inputs['sample']['expected_sample']
        )
        # 8. Ensure that score is doing the right thing.
        score_tester(
            function_bank = function_bank,
            kwargs = inputs['score']['kwargs'],
            expected_results = inputs['score']['expected_results']
        )
        # 9. Ensure that prune is doing the right thing.
        prune_tester(
            function_bank = function_bank,
            seed = inputs['prune']['seed'],
            expected_results = inputs['prune']['expected_results']
        )
        # 10. Ensure that function statistics is doing the right thing
        function_statistics_tester(
            function_bank = function_bank,
            kwargs = inputs['function_statistics']['kwargs'],
            expected_shape = inputs['function_statistics']['expected_shape'],
            expected_results = inputs['function_statistics']['expected_results']
        )
