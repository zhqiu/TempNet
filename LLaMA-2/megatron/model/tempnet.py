"""
    Definition of temperature network and its cross entropy loss function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .module import MegatronModule
from megatron.core import tensor_parallel

class TempNet(MegatronModule):
    def __init__(self, config, rho, feature_dim=32000, hid_size=256, tau_min=0.01): # feature_dim:: 32000 for llama, 50304 for pythia
        super(TempNet, self).__init__()

        self.rho = rho
        self.tau_min = tau_min
        self.tau_max = 2.0
        self.hid_size = hid_size

        self.proj = nn.Sequential(
                    nn.Linear(feature_dim, hid_size),
                    nn.Sigmoid(),
                    nn.Dropout(0.5),
                    nn.Linear(hid_size, hid_size)
                )

        self.scaler = nn.Parameter(torch.tensor(np.log(1.0)))

        self.last = nn.Linear(hid_size, 1)
        self.last.weight.data.fill_(1.0)
        self.last.bias.data.fill_(0.0)

    def forward(self, x):   # shape of x: [bsz*seq_len, feature_dim]
        x = F.normalize(x, dim=1)
        x = self.proj(x)

        weights = nn.Softmax(dim=1)(x / self.scaler.exp())

        x_diff = self.last((weights - 1.0/self.hid_size) * x)
        tau = (self.tau_max - self.tau_min) * torch.sigmoid(x_diff) + self.tau_min

        return tau
