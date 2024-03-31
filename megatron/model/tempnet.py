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
    def __init__(self, rho=0.0, feature_dim=50304, hid_size=256, tau_min=0.5):
        super(TempNet, self).__init__()

        self.rho = rho
        self.tau_min = tau_min
        self.tau_max = 5.0
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
        x = F.normalize(x, dim=1, p=2.0)
        x = self.proj(x)

        weights = nn.Softmax(dim=1)(x / self.scaler.exp())

        x_diff = self.last((weights - 1.0/self.hid_size) * x)
        tau = (self.tau_max - self.tau_min) * torch.sigmoid(x_diff) + self.tau_min

        return tau



"""
# we employ Newton's method to compute taus here
class TempNet(nn.Module):
    def __init__(self, rho=None, newton_steps=10, tau_init=1.0, tau_min=0.01):
        super(TempNet, self).__init__()

        assert rho is not None

        self.rho = rho
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.newton_steps = newton_steps
        self.vocab_size = 50304
        self.eps = 1e-8

    def _nabla_tau_gi(self, logsumexp_term, weights, output_d_taus):
        return logsumexp_term + self.rho - np.log(self.vocab_size) - torch.sum(weights * output_d_taus, dim=1)


    def _nabla_tautau_gi(self, weights, taus, output_d_taus):
        E_xsquare = torch.sum(weights * output_d_taus ** 2, dim=1)
        Ex_square = torch.sum(weights * output_d_taus, dim=1) ** 2

        return (E_xsquare - Ex_square).clamp_(min=1.0) / taus
        

    def forward(self, output):   # shape of output: [bsz*seq_len, feature_dim]
        taus = torch.ones(output.shape[0], device=output.device) * self.tau_init

        for _ in range(self.newton_steps):
            output_d_taus = output / taus[:,None]
            logsumexp_term = torch.logsumexp(output_d_taus, dim=1)
            weights = F.softmax(output_d_taus, dim=1)

            taus = taus - self._nabla_tau_gi(logsumexp_term, weights, output_d_taus) \
                        / self._nabla_tautau_gi(weights, taus, output_d_taus)
           
            taus.clamp_(min=self.tau_min)
            #print("taus:", taus)
            #print("grad mean:", self._nabla_tau_gi(logsumexp_term, weights, output_d_taus).mean())

        #print("*" * 20)

        return taus[:,None].clamp_(min=0.03, max=0.03)
"""        







