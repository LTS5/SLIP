import torch.nn as nn
import torch
from tqdm import tqdm

from .utils.losses import NLLSurvLoss, CoxLoss, SurvRankingLoss

def create_mlp(in_dim=None, hid_dims=[], act=nn.ReLU(), dropout=0.,
               out_dim=None, end_with_fc=True, bias=True):

    layers = []
    if len(hid_dims) < 0:
        mlp = nn.Identity()
    elif len(hid_dims) >= 0:
        if len(hid_dims) > 0:
            for hid_dim in hid_dims:
                layers.append(nn.Linear(in_dim, hid_dim, bias=bias))
                layers.append(act)
                layers.append(nn.Dropout(dropout))
                in_dim = hid_dim
        layers.append(nn.Linear(in_dim, out_dim))
        if not end_with_fc:
            layers.append(act)
        mlp = nn.Sequential(*layers)
    return mlp

def create_mlp_with_dropout(in_dim=None, hid_dims=[], act=nn.ReLU(), dropout=0.,
               out_dim=None, end_with_fc=True, bias=True):

    layers = []
    if len(hid_dims) < 0:
        mlp = nn.Identity()
    elif len(hid_dims) >= 0:
        if len(hid_dims) > 0:
            for hid_dim in hid_dims:
                layers.append(nn.Linear(in_dim, hid_dim, bias=bias))
                layers.append(act)
                layers.append(nn.Dropout(dropout))
                in_dim = hid_dim
        layers.append(nn.Linear(in_dim, out_dim))
        if not end_with_fc:
            layers.append(act)
            layers.append(nn.Dropout(dropout))
        mlp = nn.Sequential(*layers)
    return mlp

#
# Model processing
#
def predict_emb(self, dataset, use_cuda=True, permute=False):
    """
    Create prototype-based slide representation

    Returns
    - X (torch.Tensor): (n_data x output_set_dim)
    - y (torch.Tensor): (n_data)
    """

    X = []

    for i in tqdm(range(len(dataset))):
        batch = dataset.__getitem__(i)
        data = batch['img'].unsqueeze(dim=0)
        if use_cuda:
            data = data.cuda()
        
        with torch.no_grad():
            out = self.representation(data)
            out = out['repr'].data.detach().cpu()

        X.append(out)

    X = torch.cat(X)

    return X

def predict_clf(self, dataset, use_cuda=True, permute=False):
    """
    Create prototype-based slide representation

    Returns
    - X (torch.Tensor): (n_data x output_set_dim)
    - y (torch.Tensor): (n_data)
    """

    X, y = [], []

    for i in tqdm(range(len(dataset))):
        batch = dataset.__getitem__(i)
        data = batch[0].unsqueeze(dim=0)
        label = batch[1]
        if use_cuda:
            data = data.cuda()
        
        with torch.no_grad():
            out = self.representation(data)
            out = out['repr'].data.detach().cpu()

        X.append(out)
        y.append(label)

    X = torch.cat(X)
    y = torch.Tensor(y).to(torch.long)

    return X, y

def process_clf(logits, label, loss_fn):
    results_dict = {'logits': logits}
    log_dict = {}

    if loss_fn is not None and label is not None:
        cls_loss = loss_fn(logits, label)
        loss = cls_loss
        log_dict.update({
            'cls_loss': cls_loss.item(),
            'loss': loss.item()})
        results_dict.update({'loss': loss})
    
    return results_dict, log_dict


#
# Attention networks
#
class Attn_Net(nn.Module):
    """
    Attention Network without Gating (2 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: dropout
        n_classes: number of classes
    """

    def __init__(self, L=1024, D=256, dropout=0., n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(D, n_classes)]

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


class Attn_Net_Gated(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: dropout
        n_classes: number of classes
    """

    def __init__(self, L=1024, D=256, dropout=0., n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout)]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid(),
                            nn.Dropout(dropout)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A
