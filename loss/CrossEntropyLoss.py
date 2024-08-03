import torch 
from torch import nn
from torch.nn import functional as F

class CrossEntropyOH(nn.Module):
    def __init__(self):
        super(CrossEntropyOH, self).__init__()

    def forward(self, input, label, weight=None):
        log_prob = F.log_softmax(input, dim=1)
        if weight == None:
            print(log_prob.shape)
            print(label.shape)
            loss = -torch.sum(log_prob * label) / len(input)
        else:
            loss = -torch.sum(weight.unsqueeze(-1).repeat(0, log_prob.shape[-1]) *log_prob * label) / len(input)
        return loss