import torch.nn as nn
import torch.functional as F
from torch import Tensor
from ._atomic import AtomicFunction


class Dropout(nn.Dropout):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        p: float,
        inplace: bool = False,
        training: bool = False
    ):
        self.input_size = input_size
        self.output_size = output_size
        super().__init__(p, inplace)
        self.training = training
        self.class_name = 'Dropout'

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input)
