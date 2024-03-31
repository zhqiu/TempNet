"""
    Definition of temperature network and its cross entropy loss function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# only use this function for training
def tau_cross_entropy(output, labels, tau, rho):      # tau: [bsz*seq_len, 1]
    labels, loss_mask = labels[0], labels[1]          # in fact, loss_mask is not used
    bsz, seq_len, vocab_size = output.shape

    logits = output.reshape(bsz*seq_len, vocab_size)
    labels = labels.reshape(-1)                        # [bsz*seq_len]

    ce_loss = F.cross_entropy(logits / tau, labels, reduction='none')
    lm_loss = tau * ce_loss.unsqueeze(1) + (rho - np.log(vocab_size)) * tau

    return lm_loss.mean(), ce_loss.mean()


class TempNet(nn.Module):
    def __init__(self, rho=0.0, feature_dim=32000, hid_size=256, tau_min=0.01): # feature_dim:: 32000 for llama, 50304 for pythia
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

