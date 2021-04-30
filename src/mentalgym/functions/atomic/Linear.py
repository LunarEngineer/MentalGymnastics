from torch import Tensor
import torch.nn as nn
import time

from ._atomic import AtomicFunction

# Tim note: I refactored this to match the env.
class Linear(AtomicFunction):
    def __init__(self, input_size: int, output_size: int,
                 bias: bool = True) -> None:
        super().__init__()
        nn.Linear.__init__(self, input_size, output_size, bias)
        self.class_name = 'Linear'
        self.input_size = input_size
        self.out_features = output_size

    def reset_parameters(self) -> None:
        nn.Linear.reset_parameters(self)

    # TODO: Can we remove these from the Linear and ReLU classes and put in Atomic?
    def forward(self, input: Tensor) -> Tensor:
        # TODO: Does this need a representation to self?
        return nn.Linear.forward(input)

    def extra_repr(self) -> str:
        return nn.Linear.extra_repr(self)

