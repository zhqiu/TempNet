"""
    implementation of other two-way contrastive losses
"""

import pickle
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/gather.py
class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out




class CLIP_Loss(nn.Module):

    def __init__(self, world_size=8, temperature=0.01, personalized_tau=False, image_tau=None, text_tau=None):
        super(CLIP_Loss, self).__init__()
        self.world_size = world_size
        self.temperature = temperature
        self.personalized_tau = personalized_tau # if true, then temperatures are learnable
        self.image_tau = image_tau
        self.text_tau = text_tau

    def forward(self, image_features, text_features, image_idx=None, text_idx=None):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        if self.personalized_tau:
            image_temp = self.image_tau[image_idx]
            text_temp = self.text_tau[text_idx]
            sim = torch.einsum('i d, j d -> i j', text_features, image_features)
            labels = torch.arange(image_features.shape[0], device=image_features.device)
            total_loss = (F.cross_entropy(sim / text_temp, labels) + F.cross_entropy(sim.t() / image_temp, labels)) / 2

        else:
            sim = torch.einsum('i d, j d -> i j', text_features, image_features) / self.temperature
            labels = torch.arange(image_features.shape[0], device=image_features.device)
            total_loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2

        return total_loss




class SogCLR_Loss(nn.Module):
    def __init__(self, N=2900000, gamma=0.1, temperature=0.07, world_size=8):
        
        #Inputs:
        #   N is number of samples in training set
        
        super(SogCLR_Loss, self).__init__()
        self.world_size = world_size
        self.s_I = torch.zeros(N).cuda()
        self.s_T = torch.zeros(N).cuda()
        self.b_I = torch.zeros(N).cuda()
        self.b_T = torch.zeros(N).cuda()
        self.gamma = gamma
        self.temperature = temperature
        self.eps = 1e-14
        

    def forward(self, image_features, text_features, image_ids, text_ids, epoch):
        
        #Inputs:
        #    image_features, text_features is l2-normalized tensor
        #    image_features, text_features: [batch_size, emb_dim]
        
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

            #image_features = image_features.contiguous()
            #text_features = text_features.contiguous()
            #image_features_all = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0).contiguous()
            #text_features_all = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0).contiguous()
            

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum('i d, j d -> i j', image_features, text_features)
        diag_sim = torch.diagonal(sim)

        batch_size = sim.shape[0]
        mask_neg = (1.0 - torch.eye(batch_size)).to(sim.device)

        # E_I(x_i)*E_T(t) - E_I(x_i)*E_T(t_i)
        image_diffs = sim - diag_sim[:, None]
        # E_I(x)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        text_diffs = sim - diag_sim[None, :]

        image_diffs_d_temps = (image_diffs / self.temperature).clone().detach_()
        text_diffs_d_temps = (text_diffs / self.temperature).clone().detach_()
        
        # update b
        old_b_I = self.b_I[image_ids]
        new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, batch_size))
        self.b_I[image_ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T[text_ids]
        new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(batch_size, 1))
        self.b_T[text_ids] = torch.max(new_b_T, dim=0)[0]
        
        exp_image_diffs = torch.exp(image_diffs_d_temps - self.b_I[image_ids][:, None]) * mask_neg # -b to avoid exp operation overflow
        exp_text_diffs = torch.exp(text_diffs_d_temps - self.b_T[text_ids][None, :]) * mask_neg

        g_I = torch.sum(exp_image_diffs, dim=1, keepdim=True) / (batch_size-1)
        g_T = torch.sum(exp_text_diffs, dim=0, keepdim=True) / (batch_size-1)

        if epoch == 0:
            s_I = g_I
            s_T = g_T
        else:
            s_I = (1.0-self.gamma) * self.s_I[image_ids] * torch.exp(old_b_I - self.b_I[image_ids]) + self.gamma * g_I.squeeze()
            s_T = (1.0-self.gamma) * self.s_T[text_ids] * torch.exp(old_b_T - self.b_T[text_ids]) + self.gamma * g_T.squeeze()
            s_I = s_I.reshape(g_I.shape)
            s_T = s_T.reshape(g_T.shape)

        self.s_I[image_ids] = s_I.squeeze()
        self.s_T[text_ids] = s_T.squeeze()

        s_I = s_I.clamp(min=self.eps)
        s_T = s_T.clamp(min=self.eps)

        weights_image = exp_image_diffs / s_I
        weights_text = exp_text_diffs / s_T

        if torch.any(torch.isnan(weights_image)):
            assert 0, "weights_image has nan."
        if torch.any(torch.isnan(weights_text)):
            assert 0, "weights_text has nan."

        image_loss = torch.sum(weights_image * image_diffs, dim=1, keepdim=True) / (batch_size-1)
        text_loss = torch.sum(weights_text * text_diffs, dim=0, keepdim=True) / (batch_size-1)

        total_loss = image_loss.mean() + text_loss.mean()

        return total_loss




"""
    https://github.com/goel-shashank/CyCLIP/blob/52d77af2a5f1a4bff01b4c371d6b98e2d0340137/src/train.py
"""
class CyCLIP_Loss(nn.Module):
    def __init__(self, world_size, temperature, cylambda_1=0.25 , cylambda_2=0.25):
        super(CyCLIP_Loss, self).__init__()

        self.world_size = world_size
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.cylambda_1 = cylambda_1
        self.cylambda_2 = cylambda_2


    def forward(self, image_features, text_features):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        batch_size = len(image_features)

        logits_text_per_image = (image_features @ text_features.t()) / self.temperature
        logits_image_per_text = logits_text_per_image.t()

        target = torch.arange(batch_size).long().cuda()

        # contrastive loss, the same as CLIP
        contrastive_loss = (self.criterion(logits_text_per_image, target) + self.criterion(logits_image_per_text, target)) / 2.0 

        # inmodal_cyclic_loss
        logits_image_per_image = (image_features @ image_features.t()) / self.temperature
        logits_text_per_text = (text_features @ text_features.t()) / self.temperature
        inmodal_cyclic_loss = (logits_image_per_image - logits_text_per_text).square().mean() * (self.temperature ** 2) * batch_size

        # crossmodal_cyclic_loss
        crossmodal_cyclic_loss = (logits_text_per_image - logits_image_per_text).square().mean() * (self.temperature ** 2) * batch_size

        loss = contrastive_loss + self.cylambda_1 * inmodal_cyclic_loss + self.cylambda_2 * crossmodal_cyclic_loss

        return loss




"""
    VICReg
    https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
"""
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    

class VICReg_Loss(nn.Module):
    def __init__(self, world_size, dim_size, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        super(VICReg_Loss, self).__init__()

        self.world_size = world_size
        self.dim_size = dim_size
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff


    def forward(self, image_features, text_features):
        if self.world_size > 1:
            x = torch.cat(GatherLayer.apply(image_features), dim=0)
            y = torch.cat(GatherLayer.apply(text_features), dim=0)

        batch_size = len(x)

        repr_loss = F.mse_loss(x, y) # invariance term

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2  # variance term

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.dim_size
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.dim_size)  # covariance term

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )

        return loss



class iSogCLR_Loss(nn.Module):
    def __init__(self, N=2900000, gamma=0.8, tau_init=0.01, world_size=8, bsz=128, rho=8.0):      
        
        #Inputs:
        #   N is number of samples in training set
        
        super(iSogCLR_Loss, self).__init__()
        self.world_size = world_size
        self.s_I = torch.zeros(N).cuda()
        self.s_T = torch.zeros(N).cuda()
        self.b_I = torch.zeros(N).cuda()
        self.b_T = torch.zeros(N).cuda()
        self.gamma = gamma
        self.eps = 1e-14

        self.tau_min, self.tau_max = 0.005, 0.05
        self.rho = rho

        self.eta_init = 0.03

        self.beta_u = 0.5
        self.grad_clip = 5.0
        self.tau_I = torch.ones(N).cuda() * tau_init
        self.tau_T = torch.ones(N).cuda() * tau_init
        self.u_I = torch.zeros(N).cuda()
        self.u_T = torch.zeros(N).cuda()


    def forward(self, image_features, text_features, image_ids, text_ids, epoch, max_epoch):
        
        #Inputs:
        #    image_features, text_features is l2-normalized tensor
        #    image_features, text_features: [batch_size, emb_dim]
        
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum('i d, j d -> i j', image_features, text_features)
        diag_sim = torch.diagonal(sim)

        batch_size = sim.shape[0]
        mask_neg = (1.0 - torch.eye(batch_size)).cuda()

        # generate temperatures
        tau_image = self.tau_I[image_ids]
        tau_text = self.tau_T[text_ids]

        # E_I(x_i)*E_T(t) - E_I(x_i)*E_T(t_i)
        image_diffs = sim - diag_sim[:, None]
        # E_I(x)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        text_diffs = sim - diag_sim[None, :]

        image_diffs_d_temps = (image_diffs / tau_image[:, None]).detach()
        text_diffs_d_temps = (text_diffs / tau_text[None, :]).detach()

        # update b
        old_b_I = self.b_I[image_ids]
        new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, batch_size))
        self.b_I[image_ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T[text_ids]
        new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(batch_size, 1))
        self.b_T[text_ids] = torch.max(new_b_T, dim=0)[0]

        exp_image_diffs = torch.exp(image_diffs_d_temps - self.b_I[image_ids][:, None]) * mask_neg # -b to avoid exp operation overflow
        exp_text_diffs = torch.exp(text_diffs_d_temps - self.b_T[text_ids][None, :]) * mask_neg

        g_I = torch.sum(exp_image_diffs, dim=1, keepdim=True)
        g_T = torch.sum(exp_text_diffs, dim=0, keepdim=True)

        if epoch == 0:
            s_I = g_I
            s_T = g_T
        else:
            s_I = (1.0-self.gamma) * self.s_I[image_ids] * torch.exp(old_b_I - self.b_I[image_ids]) + self.gamma * g_I.squeeze()
            s_T = (1.0-self.gamma) * self.s_T[text_ids] * torch.exp(old_b_T - self.b_T[text_ids]) + self.gamma * g_T.squeeze()
            s_I = s_I.reshape(g_I.shape)
            s_T = s_T.reshape(g_T.shape)

        self.s_I[image_ids] = s_I.squeeze()
        self.s_T[text_ids] = s_T.squeeze()

        s_I = s_I.clamp(min=self.eps)
        s_T = s_T.clamp(min=self.eps)

        weights_image = exp_image_diffs / s_I
        weights_text = exp_text_diffs / s_T

        image_loss = torch.sum(weights_image * image_diffs, dim=1, keepdim=True)
        text_loss = torch.sum(weights_text * text_diffs, dim=0, keepdim=True)

        total_loss = image_loss.mean() + text_loss.mean()
 
        temp_weight_image = torch.log(s_I / (batch_size-1)) + self.b_I[image_ids][:, None] + self.rho - torch.sum(weights_image * image_diffs_d_temps, dim=1, keepdim=True)
        temp_weight_text = torch.log(s_T / (batch_size-1)) + self.b_T[text_ids][None, :] + self.rho - torch.sum(weights_text * text_diffs_d_temps, dim=0, keepdim=True)

        self.u_I[image_ids] = (1.0-self.beta_u) * self.u_I[image_ids] + self.beta_u * temp_weight_image.squeeze().clamp_(min=-self.grad_clip, max=self.grad_clip)
        self.u_T[text_ids] = (1.0-self.beta_u) * self.u_T[text_ids] + self.beta_u * temp_weight_text.squeeze().clamp_(min=-self.grad_clip, max=self.grad_clip)

        self.tau_I[image_ids] = (tau_image - self.eta_init * self.u_I[image_ids]).clamp_(min=self.tau_min, max=self.tau_max)
        self.tau_T[text_ids] = (tau_text - self.eta_init * self.u_T[text_ids]).clamp_(min=self.tau_min, max=self.tau_max)

        return total_loss, tau_image.cpu().detach().numpy(), tau_text.cpu().detach().numpy()





class TempGenerator(torch.nn.Module):
    def __init__(self, feature_dim, M=256, tau_min=0.005, dropout_rate=0.5, rho=6.0):
        super(TempGenerator, self).__init__()

        self.feature_dim = feature_dim
        self.M = M
        self.tau_min = tau_min
        self.tau_max = 0.05
        self.rho = rho

        self.proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.scaler = nn.Parameter(torch.tensor(np.log(0.01)))

        self.prototypes = nn.Parameter(torch.empty((self.M, self.feature_dim)))
        nn.init.normal_(self.prototypes, 0.0, 1.0)

        self.linear_1 = nn.Linear(self.M, 1)
        self.linear_1.weight.data.fill_(1.0)
        self.linear_1.bias.data.fill_(0.0)

        self.dropout = nn.Dropout(dropout_rate)

    def _init_prototypes(self, feats):
        self.prototypes.data.copy_(feats)

    def forward(self, x, return_feats=False):

        x = self.dropout(torch.sigmoid(self.proj(x)))

        if return_feats:
            return x

        normed_protos = F.normalize(self.prototypes, p=2.0, dim=1)

        prods = x @ normed_protos.t()

        weights = nn.Softmax(dim=1)(prods / self.scaler.exp())
        sims = torch.sigmoid(prods)

        tau = self.linear_1((weights - 1.0/self.M) * sims).squeeze()

        return (self.tau_max - self.tau_min) * torch.sigmoid(tau) + self.tau_min



class iSogCLR_TempNet_Loss(nn.Module):  # using TempGenerator
    def __init__(self, N=2900000, gamma=0.8, world_size=8, 
                       rho=8.0, feature_dim=256, tau_min=0.01):  # use temperature network      
        
        #Inputs:
        #   N is number of samples in training set
        
        super(iSogCLR_TempNet_Loss, self).__init__()
        self.world_size = world_size
        self.s_I = torch.zeros(N).cuda()
        self.s_T = torch.zeros(N).cuda()
        self.b_I = torch.zeros(N).cuda()
        self.b_T = torch.zeros(N).cuda()
        self.gamma = gamma
        self.eps = 1e-14

        self.rho = rho

        self.image_temp_gen = TempGenerator(feature_dim=feature_dim, rho=self.rho).cuda()
        self.text_temp_gen = TempGenerator(feature_dim=feature_dim, rho=self.rho).cuda()


    def forward(self, image_features, text_features, image_ids, text_ids, epoch, max_epoch):
        
        #Inputs:
        #    image_features, text_features is l2-normalized tensor
        #    image_features, text_features: [batch_size, emb_dim]
        
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum('i d, j d -> i j', image_features, text_features)
        diag_sim = torch.diagonal(sim)

        batch_size = sim.shape[0]
        mask_neg = (1.0 - torch.eye(batch_size)).cuda()

        # generate temperatures
        tau_image = self.image_temp_gen(image_features.detach())
        tau_text = self.text_temp_gen(text_features.detach())

        # E_I(x_i)*E_T(t) - E_I(x_i)*E_T(t_i)
        image_diffs = sim - diag_sim[:, None]
        # E_I(x)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        text_diffs = sim - diag_sim[None, :]

        image_diffs_d_temps = (image_diffs / tau_image[:, None]).detach()
        text_diffs_d_temps = (text_diffs / tau_text[None, :]).detach()

        # update b
        old_b_I = self.b_I[image_ids]
        new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, batch_size))
        self.b_I[image_ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T[text_ids]
        new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(batch_size, 1))
        self.b_T[text_ids] = torch.max(new_b_T, dim=0)[0]

        exp_image_diffs = torch.exp(image_diffs_d_temps - self.b_I[image_ids][:, None]) * mask_neg # -b to avoid exp operation overflow
        exp_text_diffs = torch.exp(text_diffs_d_temps - self.b_T[text_ids][None, :]) * mask_neg

        g_I = torch.sum(exp_image_diffs, dim=1, keepdim=True)
        g_T = torch.sum(exp_text_diffs, dim=0, keepdim=True)

        if epoch == 0:
            s_I = g_I
            s_T = g_T
        else:
            s_I = (1.0-self.gamma) * self.s_I[image_ids] * torch.exp(old_b_I - self.b_I[image_ids]) + self.gamma * g_I.squeeze()
            s_T = (1.0-self.gamma) * self.s_T[text_ids] * torch.exp(old_b_T - self.b_T[text_ids]) + self.gamma * g_T.squeeze()
            s_I = s_I.reshape(g_I.shape)
            s_T = s_T.reshape(g_T.shape)

        self.s_I[image_ids] = s_I.squeeze()
        self.s_T[text_ids] = s_T.squeeze()

        s_I = s_I.clamp(min=self.eps)
        s_T = s_T.clamp(min=self.eps)

        weights_image = exp_image_diffs / s_I
        weights_text = exp_text_diffs / s_T

        image_loss = torch.sum(weights_image * image_diffs, dim=1, keepdim=True)
        text_loss = torch.sum(weights_text * text_diffs, dim=0, keepdim=True)

        clip_loss = image_loss.mean() + text_loss.mean()
  
        temp_weight_image = torch.log(s_I / (batch_size-1)) + self.b_I[image_ids][:, None] + self.rho - torch.sum(weights_image * image_diffs_d_temps, dim=1, keepdim=True)
        temp_weight_text = torch.log(s_T / (batch_size-1)) + self.b_T[text_ids][None, :] + self.rho - torch.sum(weights_text * text_diffs_d_temps, dim=0, keepdim=True)

        temp_image_loss = torch.mean(temp_weight_image * tau_image[:, None])
        temp_text_loss = torch.mean(temp_weight_text * tau_text[None, :])

        temp_loss = temp_image_loss + temp_text_loss

        return (clip_loss, temp_loss), tau_image.cpu().detach().numpy(), tau_text.cpu().detach().numpy()



"""
    from dixian
"""
class onlineCLR_Loss(nn.Module):
    def __init__(self, temperature=0.01, world_size=8, gamma=0.5):
        """
        Inputs:
           N is number of samples in training set
        """
        super(onlineCLR_Loss, self).__init__()
        self.world_size = world_size
        self.pT = temperature*10
        self.nT = temperature
        
        self.u_p = torch.zeros(1).cuda() 
        self.u_n = torch.zeros(1).cuda()
        self.c_p = torch.zeros(1).cuda()
        self.c_n = torch.zeros(1).cuda() 
             
        self.gamma = gamma

    def forward(self, image_features, text_features):
        if self.world_size > 1:
            hidden1 = torch.cat(GatherLayer.apply(image_features), dim=0)
            hidden2 = torch.cat(GatherLayer.apply(text_features), dim=0)

        batch_size = hidden1.shape[0]
        
        labels = torch.eye(batch_size).cuda() # identity matrix

        logits_ab = torch.matmul(hidden1, hidden2.T)
        logits_ba = torch.matmul(hidden2, hidden1.T)

        #  online contrastive learning
        neg_mask = 1-labels

        neg_logits1 = logits_ab*neg_mask  
        pos_logits1 = logits_ab*labels   
        neg_logits2 = logits_ba*neg_mask
        pos_logits2 = logits_ba*labels   

        max_neg_logits = torch.maximum(torch.max(neg_logits1), torch.max(neg_logits2)).detach()
        max_pos_logits = torch.maximum(torch.max(-pos_logits1), torch.max(-pos_logits2)).detach()

        neg_logits1_exp = torch.exp((neg_logits1-max_neg_logits)/self.nT)*neg_mask  
        pos_logits1_exp = torch.exp((-pos_logits1-max_pos_logits)/self.pT)*labels   
        neg_logits2_exp = torch.exp((neg_logits2-max_neg_logits)/self.nT)*neg_mask
        pos_logits2_exp = torch.exp((-pos_logits2-max_pos_logits)/self.pT)*labels

        self.u_n = (1 - self.gamma) * self.u_n.cuda() * torch.exp((self.c_n-max_neg_logits)/self.nT) + self.gamma * torch.sum(neg_logits1_exp+neg_logits2_exp).detach()
        self.u_p = (1 - self.gamma) * self.u_p.cuda() * torch.exp((self.c_p-max_pos_logits)/self.pT) + self.gamma * torch.sum(pos_logits1_exp+pos_logits2_exp).detach()
        self.c_n = max_neg_logits.cuda()
        self.c_p = max_pos_logits.cuda()

        p_neg_weights1 = (neg_logits1_exp/self.u_n).detach()
        p_pos_weights1 = (pos_logits1_exp/self.u_p).detach()
        p_neg_weights2 = (neg_logits2_exp/self.u_n).detach()
        p_pos_weights2 = (pos_logits2_exp/self.u_p).detach()

        def softmax_cross_entropy_with_logits_v2(pos_logits, pos_weights, neg_logits, neg_weights): 
            expsum_neg_logits = torch.sum(neg_weights*neg_logits)  # loss on negative pairs
            expsum_pos_logits = torch.sum(pos_weights*pos_logits)  # loss on positive pairs
            normalized_logits = expsum_neg_logits - expsum_pos_logits
            return normalized_logits

        loss_a = softmax_cross_entropy_with_logits_v2(pos_logits1, p_pos_weights1, neg_logits1, p_neg_weights1)
        loss_b = softmax_cross_entropy_with_logits_v2(pos_logits2, p_pos_weights2, neg_logits2, p_neg_weights2)
        loss = (loss_a + loss_b).mean()

        return loss

