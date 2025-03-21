
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionAdapter(torch.nn.Module):
    def __init__(self, c_in, reduction=4):
        super(VisionAdapter, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(c_in, c_in // reduction, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(c_in // reduction, c_in, bias=False),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    

class Prototypes(torch.nn.Module):
    def __init__(self, k, d):
        super().__init__()
        self.prototypes = torch.nn.Parameter(torch.randn(k, d), requires_grad=True)

    def forward(self,):
        normalized_prototypes = F.normalize(self.prototypes, dim=1)
        return normalized_prototypes

