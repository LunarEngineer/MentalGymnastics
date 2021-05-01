import torch.nn as nn

from torch import Tensor
from ._atomic import AtomicFunction


class ReLU(nn.ReLU):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        inplace: bool = False
    ):
        self.input_size = input_size
        self.output_size = output_size
        super().__init__(inplace)
        self.class_name = 'ReLU'

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input)

    def extra_repr(self) -> str:
        return super().extra_repr()


