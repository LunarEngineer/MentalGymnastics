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
        self.p = p
        self.training = training
        self.inplace = inplace
        self.class_name = 'Dropout'

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, self.training, self.inplace)
