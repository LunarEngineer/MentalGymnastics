from ._atomic import AtomicFunction
from .Linear import Linear
from .relu import ReLU
from .Dropout import Dropout
from mentalgym.constants import linear_i, relu_i, dropout_i

# Here we are going to make a Function Set for the Atomic functions.
# The id's are numerically descending.
# TODO: Add any *new* atomic functions to this list.
# TODO: Someday make this more intelligent, potentially scrape a dataset.
atomic_functions = [
    dict(
        function_index = linear_i,
        function_id = 'Linear',
        function_object = Linear,
        function_type = 'atomic'
    ),
    dict(
        function_index = relu_i,
        function_id = 'ReLU',
        function_object = ReLU,
        function_type = 'atomic'
    ),
    dict(
        function_index = dropout_i,
        function_id = 'Dropout',
        function_object = Dropout,
        function_type = 'atomic'
    )
]

atomic_constants = {
    i: _['function_object']
    for (i, _) in enumerate(atomic_functions)
}

__all__ = [
    'AtomicFunction',
    'Linear',
    'RelU',
    'Dropout',
    'atomic_functions'
    'atomic_constants'
]