from torch import Tensor
import torch.nn as nn
import time

from ._atomic import AtomicFunction

# Tim note: I refactored this to match the env.
class Linear(nn.Linear, AtomicFunction):
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias)
        # TODO: Review this with Vahe
        # nn.Linear.__init__(self, input_size, output_size, bias)
        self.class_name = 'Linear'

    # TODO: This should not be necessary.
    # def reset_parameters(self) -> None:
    #     nn.Linear.reset_parameters(self)

    # TODO: Can we remove these from the Linear and ReLU classes and put in Atomic?
    # def forward(self, input: Tensor) -> Tensor:
    #     # TODO: Does this need a representation to self?
    #     return nn.Linear.forward(input)

    # def extra_repr(self) -> str:
    #     return nn.Linear.extra_repr(self)

