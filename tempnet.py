from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import transformers
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


def tau_cross_entropy(logits, labels, tau, rho):      # logits: [bsz, vocav_size], labels: [bsz], tau: [bsz, 1]
    lm_loss = F.cross_entropy(logits / tau, labels, reduction='none')
    lm_loss = tau * lm_loss.unsqueeze(1) + (rho - np.log(32000)) * tau

    return lm_loss.mean()


class TempNet(nn.Module):
    def __init__(self, feature_dim=32000, hid_size=256, tau_min=0.01): # feature_dim: 32000 for llama
        super(TempNet, self).__init__()

        self.tau_min = tau_min
        self.tau_max = 1.2
        self.hid_size = hid_size

        self.proj = nn.Sequential(
                    nn.Linear(feature_dim, hid_size),
                    nn.Sigmoid(),
                    nn.Dropout(0.5),
                    nn.Linear(hid_size, hid_size)
                )

        self.scaler = nn.Parameter(torch.tensor([np.log(1.0)], dtype=torch.float16)) 

        self.last = nn.Linear(hid_size, 1)
        self.last.weight.data.fill_(1.0)
        self.last.bias.data.fill_(0.0)

    def forward(self, x):   # shape of x: [bsz, vocab_size]
        x = F.normalize(x, dim=-1)
        x = self.proj(x)

        weights = nn.Softmax(dim=-1)(x / self.scaler.exp())

        x_diff = self.last((weights - 1.0/self.hid_size) * x)
        tau = (self.tau_max - self.tau_min) * torch.sigmoid(x_diff) + self.tau_min

        return tau


class LLaMA_TempNet(LlamaForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tempnet = TempNet()

    def set_rho(self, rho):
        self.rho = rho

    def get_rho(self):
        return self.rho

    def print_trainable_parameters(self):
        total_trainable_params = 0

        for param in self.parameters():
            if param.requires_grad:
                total_trainable_params += param.numel()
        return total_trainable_params

    def set_tempnet(self, TempNet):
        self.tempnet = TempNet

    def fix_llama(self):
        for n, p in self.named_parameters():
            if 'tempnet' not in n:
                p.requires_grad = False


    def forward(
        self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache = False,
        output_attentions = False,
        output_hidden_states = False,
        return_dict = False,
        return_tau = False,
        cache_position = None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        assert use_cache == True

        with torch.no_grad():
            outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

            hidden_states = outputs[0]
            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.lm_head(hidden_states)
            logits = logits.float()

        if return_tau:
            with torch.no_grad():
                taus = self.tempnet(logits.half()).float()

            return taus.cpu().numpy()

        loss = None
        if labels is not None:                   # enter this branch for training
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)

            taus = self.tempnet(shift_logits)
            loss = tau_cross_entropy(shift_logits, shift_labels, taus, self.rho)

            return (loss, taus.mean(), taus.max(), taus.min())
       
        else:                                   # enter this branch for generation
            logits_max = torch.max(logits, dim=-1)[0]
            _logits = logits - logits_max.unsqueeze(dim=-1)
            with torch.no_grad():
                taus = self.tempnet(_logits.half()).float()
                logits = logits / taus

            return CausalLMOutputWithPast(
                    loss=loss, logits=logits, past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states, attentions=outputs.attentions
                )
