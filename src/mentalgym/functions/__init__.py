from .atomic import AtomicFunction
from .composed import ComposedFunction
from .intermediate import IntermediateFunction
from collections import deque

# TODO: Betterify this. This works for now.

_function_atomic_one = {
    'i': -2,
    'id': 'ReLU',
    'type': 'atomic',
    'input': None,
    'living': True,
    'object': AtomicFunction(),
    'score_default': deque([0], maxlen=100)
}

_function_atomic_two = {
    'i': -2,
    'id': 'Dropout',
    'type': 'atomic',
    'input': None,
    'living': True,
    'object': AtomicFunction(),
    'score_default': deque([0], maxlen=100)
}

atomic_functions = [
    _function_atomic_one,
    _function_atomic_two
]

__all__ = [
    AtomicFunction,
    ComposedFunction,
    IntermediateFunction,
    atomic_functions
]