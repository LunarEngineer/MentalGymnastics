import torch.nn as nn
import torch.functional as F
from torch import Tensor
from ._atomic import AtomicFunction


class Dropout(nn.Dropout, AtomicFunction):
    # TODO: Can we remove these and put in Atomic?
    def __init__(self, in_features: int, out_features: int,
                 p: float, inplace: bool = False, training: bool = False) -> None:
        self.p = p
        self.training = training
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, self.training, self.inplace)
