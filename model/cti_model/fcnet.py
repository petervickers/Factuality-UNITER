### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F

### CLASS DEFINITION ###
class SuperSimpleNet(nn.Module):
    """
        Simple class for non-linear fully-connected network
    """

    def __init__(self, dims, act="ReLU", dropout=0):
        super(SuperSimpleNet, self).__init__()

        self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(768, 256)

    def forward(self, x):
        x = self.drop(x)
        x = self.fc1(x.float())
        return (x)


### CLASS DEFINITION ###
class FCNet(nn.Module):
    """
        Simple class for non-linear fully-connected network
    """

    def __init__(self, dims, act="ReLU", dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]

            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if "" != act:
                layers.append(getattr(nn, act)())

        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if "" != act:
            layers.append(getattr(nn, act)())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x.cuda())
