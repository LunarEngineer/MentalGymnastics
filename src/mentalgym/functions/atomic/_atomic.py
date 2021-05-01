import torch.nn as nn
from torch import Tensor

# TODO: Deprecate
class AtomicFunction():
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size