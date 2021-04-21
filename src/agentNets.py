# import torch
import torch.nn as nn

num_functions = 8
max_steps = 10
num_neurons = 128


class DQNAgentNN(nn.Module):
    def __init__(self):
        super(DQNAgentNN, self).__init__()

        self.layer1 = nn.Linear(
            num_functions * max_steps + max_steps + 1, num_neurons
        )
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(num_neurons, 100)

    def forward(self, x):

        u = self.layer1(x)
        u = self.layer2(u)
        out = self.layer3(u)

        return out
