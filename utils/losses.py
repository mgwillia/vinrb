### code adapted from https://github.com/abhyantrika/mediainr ###

import torch
import torch.nn as nn
import torch.nn.functional as F


class l1(nn.Module):
    def __init__(self):
        super(l1, self).__init__()
        self.loss = {}

    def forward(self, input, target):
        self.loss = F.l1_loss(input, target)
        return self.loss
        
class mse(nn.Module):
    def __init__(self):
        super(mse, self).__init__()
        self.loss = {}

    def forward(self, input, target):
        self.loss = F.mse_loss(input, target)
        return self.loss

class psnr(nn.Module):
    def __init__(self):
        super(psnr, self).__init__()
        self.loss = {}

    def forward(self, input, target):
        mse = F.mse_loss(input, target)
        #self.loss = -10*torch.log10(mse)
        self.loss = 20. * torch.log10(torch.tensor(1., device='cuda')) - 10. * torch.log10(mse)
        self.loss = -1*self.loss
        return self.loss


def self_information(weight:float, prob_model, is_single_model:bool=False, is_val:bool=False, g=None):
    weight = (weight + torch.rand(weight.shape, generator=g).to(weight)-0.5) if not is_val else torch.round(weight)
    weight_p = weight + 0.5
    weight_n = weight - 0.5
    if not is_single_model:
        prob = prob_model(weight_p) - prob_model(weight_n)
    else:
        prob = prob_model(weight_p.reshape(-1,1))-prob_model(weight_n.reshape(-1,1))
    total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / np.log(2.0), 0, 50))
    return total_bits, prob
    
class entropy_reg(nn.Module):
    def __init__(self):
        super(entropy_reg, self).__init__()
        self.loss = {}

    def forward(self,latents, prob_models, single_prob_model, lambda_loss):
        bits = num_elems = 0
        for group_name in latents:
            if torch.any(torch.isnan(latents[group_name])):
                raise Exception('Weights are NaNs')
            cur_bits, prob = self_information(latents[group_name],prob_models[group_name], single_prob_model, is_val=False)
            bits += cur_bits
            num_elems += prob.size(0)
        self.loss = bits/num_elems*lambda_loss #{'ent_loss': bits/num_elems*lambda_loss}
        return self.loss, bits.float().item()/8
