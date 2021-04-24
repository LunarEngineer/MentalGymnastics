import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import time
import AtomicFunction


class ReLU(nn.ReLU):
    def __init__(self, inplace: bool = False):
        super().__init__(inplace)

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input)

    def extra_repr(self) -> str:
        return super().extra_repr()


