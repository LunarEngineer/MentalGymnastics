import pytest
from mentalgym.utils.validation import is_function

test_inputs = [
    {'id': 'anything', 'type': 'imaginary', 'input': 'cheese'},
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