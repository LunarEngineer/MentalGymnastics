import numpy as np
import pandas as pd
import pytest
from mentalgym.utils.sampling import softmax_score_sample


test_input = pd.DataFrame(
    data = {
        'id': ['bob','janice','dilly','dally','beans'],
        'meaningless': ['a','b','c','d','e'],
        'extra': ['a','b','c','d','e'],
        'information': ['a','b','c','d','e'],
        'score_accuracy': [0.95, 0.7, 0.6, 0.5, 0.6],
        'score_complexity': [0.01, 100, 10, 20, 50]
    }
)

test_sets = [
    (test_input, 4, 0, np.array(['janice', 'bob', 'beans', 'dally'], dtype=object)),
    (test_input, 4, 16, np.array(['janice', 'bob', 'beans', 'dilly'], dtype=object)),
    (test_input, 3, 412789, np.array(['dilly', 'janice', 'beans'], dtype=object))
]

parameter_string = 'test_data,n,random_state,expected_sample'
@pytest.mark.parametrize(parameter_string, test_sets)
def test_softmax_score_sample(
    test_data,
    n,
    random_state,
    expected_sample
):
    actual_sample = softmax_score_sample(
        test_data,
        n,
        random_state
    )
    assert np.all(actual_sample==expected_sample)