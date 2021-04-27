import torch.nn as nn
from torch import Tensor

class AtomicFunction():
    def __init__(self):
        self.class_name = None

    def forward(self, input: Tensor) -> Tensor:
        return nn.Linear.forward(input)

    def extra_repr(self) -> str:
        return nn.Linear.extra_repr(self)