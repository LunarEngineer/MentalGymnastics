import pytest
from mentalgym.utils.validation import is_function
from mentalgym.constants import experiment_space_fields

test_inputs = [
    {_: 'composed' for _ in experiment_space_fields},
    {'id': 'not_quite', 'type': 'wrong'},
    'steve',
    1
]
test_output_bool = [
    True,
    False,
    False,
    False
]
test_sets = zip(test_inputs, test_output_bool)

@pytest.mark.parametrize('item, actual_value',test_sets)
def test_is_function(item, actual_value):
    assert is_function(item) == actual_value
    if not actual_value:
        with pytest.raises(Exception):
            is_function(item, True)