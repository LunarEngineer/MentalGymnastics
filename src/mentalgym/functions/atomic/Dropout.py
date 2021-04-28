import torch.nn as nn

from torch import Tensor
from ._atomic import AtomicFunction


class Dropout(nn.Dropout, AtomicFunction):
    # TODO: Can we remove these and put in Atomic?
    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, self.training, self.inplace)
