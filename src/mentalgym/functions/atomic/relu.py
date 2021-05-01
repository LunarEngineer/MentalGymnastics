import torch.nn as nn

from torch import Tensor
from ._atomic import AtomicFunction


class ReLU(nn.ReLU, AtomicFunction):
    def __init__(self, in_features: int, out_features: int,
                 inplace: bool = False) -> None:
        super().__init__(inplace)
        self.class_name = 'ReLU'

    # TODO: Can we remove these from the Linear and ReLU classes and put in Atomic?
    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input)

    def extra_repr(self) -> str:
        return super().extra_repr()


