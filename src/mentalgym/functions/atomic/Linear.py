from torch import Tensor
import torch.nn as nn
import time

from ._atomic import AtomicFunction

# Tim note: I refactored this to match the env.
class Linear(nn.Linear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True
    ):
        self.input_size = input_size
        self.output_size = output_size
        super().__init__(
            in_features = input_size,
            out_features = output_size,
            bias = bias
        )
        self.class_name = 'Linear'