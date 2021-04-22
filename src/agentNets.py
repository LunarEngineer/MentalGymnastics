# import torch
import torch.nn as nn


class DQNAgentNN(nn.Module):
    def __init__(self, hparams):
        super(DQNAgentNN, self).__init__()

        self.layers = nn.ModuleList([])
        self.layers.append(
            nn.Linear(
                hparams["num_functions"] * hparams["max_steps"]
                + hparams["max_steps"]
                + 1,
                hparams["hidden_layers"][0],
            )
        )
        for i in range(len(hparams["hidden_layers"]) - 1):
            self.layers.append(
                nn.linear(
                    hparams["hidden_layers"][i],
                    hparams["hidden_layers"][i + 1],
                )
            )
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hparams["hidden_layers"][-1], 100))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x
