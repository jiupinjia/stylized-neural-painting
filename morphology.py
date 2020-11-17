import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


class Erosion2d(nn.Module):

    def __init__(self, m=1):

        super(Erosion2d, self).__init__()
        self.m = m
        self.pad = [m, m, m, m]
        self.unfold = nn.Unfold(2*m+1, padding=0, stride=1)

    def forward(self, x):
        batch_size, c, h, w = x.shape
        x_pad = F.pad(x, pad=self.pad, mode='constant', value=1e9)
        for i in range(c):
            channel = self.unfold(x_pad[:, [i], :, :])
            channel = torch.min(channel, dim=1, keepdim=True)[0]
            channel = channel.view([batch_size, 1, h, w])
            x[:, [i], :, :] = channel

        return x



class Dilation2d(nn.Module):

    def __init__(self, m=1):

        super(Dilation2d, self).__init__()
        self.m = m
        self.pad = [m, m, m, m]
        self.unfold = nn.Unfold(2*m+1, padding=0, stride=1)

    def forward(self, x):
        batch_size, c, h, w = x.shape
        x_pad = F.pad(x, pad=self.pad, mode='constant', value=-1e9)
        for i in range(c):
            channel = self.unfold(x_pad[:, [i], :, :])
            channel = torch.max(channel, dim=1, keepdim=True)[0]
            channel = channel.view([batch_size, 1, h, w])
            x[:, [i], :, :] = channel

        return x
