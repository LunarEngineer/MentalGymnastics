from torch import Tensor
import torch.nn as nn
import time

from ._atomic import AtomicFunction

class Linear(AtomicFunction):
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True) -> None:
        super().__init__()
        nn.Linear.__init__(self, in_features, out_features, bias)

    def reset_parameters(self) -> None:
        nn.Linear.reset_parameters(self)

    # TODO: Can we remove these from the Linear and ReLU classes and put in Atomic?
    def forward(self, input: Tensor) -> Tensor:
        return nn.Linear.forward(input)

    def extra_repr(self) -> str:
        return nn.Linear.extra_repr(self)

