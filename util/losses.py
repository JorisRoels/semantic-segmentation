
import torch
import torch.nn as nn
import torch.nn.functional as F
from distutils.version import LooseVersion

def cross_entropy2d(input, target, weight=None, size_average=False):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight)
    if size_average:
        loss /= mask.data.sum()
    return loss

class JaccardLoss(nn.Module):

    def forward(self, input, target):

        eps = 1e-10

        predicted_probabilities = F.softmax(input, dim=1)[:, 1:2, ...]
        target = target.float()

        intersection = (predicted_probabilities * target).sum()
        union = predicted_probabilities.sum() + target.sum() - intersection

        return - (intersection+eps) / (union+eps)