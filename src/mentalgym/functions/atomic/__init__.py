from mentalgym.utils.function import make_function
from .Linear import Linear
from .relu import ReLU

# Here we are going to make a Function Set for the Atomic functions.
# The id's are numerically descending.
# TODO: Add any *new* atomic functions to this list.
# TODO: Someday make this more intelligent, potentially scrape a dataset.
atomic_functions = [
    make_function(
        function_index = 0,
        function_id = 'Linear',
        function_object = Linear,
        function_type = 'atomic'
    ),
    make_function(
        function_index = 1,
        function_id = 'ReLU',
        function_object = ReLU,
        function_type = 'atomic'
    )
]

__all__ = [
    'Linear',
    'RelU',
    'atomic_functions'
]