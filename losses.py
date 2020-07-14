import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from scipy.special import softmax
from torch.autograd import Function

class FocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)

    def forward(self, pred, target):
        if pred.dim()>2:
            pred = pred.contiguous().view(pred.size(0),pred.size(1),-1)  # N,C,H,W => N,C,H*W
            pred = pred.transpose(1,2)    # N,C,H*W => N,H*W,C
            pred = pred.contiguous().view(-1, pred.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(pred, dim = 1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=pred.data.type():
                self.alpha = self.alpha.type_as(pred.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        
        return loss.mean()