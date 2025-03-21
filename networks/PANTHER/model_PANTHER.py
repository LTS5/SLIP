# Model initiation for PANTHER

from torch import nn
import numpy as np

from .components import predict_clf, predict_emb
from .layers import PANTHERBase
from .utils.proto_utils import check_prototypes


class PANTHER(nn.Module):
    """
    Wrapper for PANTHER model
    """
    def __init__(self, emb_dim, heads, out_size, load_proto, proto_path, 
                 em_iter, tau, out_type, ot_eps, fix_proto, mode):
        super(PANTHER, self).__init__()

        emb_dim = emb_dim

        self.emb_dim = emb_dim
        self.heads = heads
        self.outsize = out_size
        self.load_proto = load_proto
        self.mode = mode

        check_prototypes(out_size, self.emb_dim, self.load_proto, proto_path)
        # This module contains the EM step
        self.panther = PANTHERBase(self.emb_dim, p=out_size, L=em_iter,
                         tau=tau, out=out_type, ot_eps=ot_eps,
                         load_proto=load_proto, proto_path=proto_path,
                         fix_proto=fix_proto)

    def representation(self, x):
        """
        Construct unsupervised slide representation
        """
        out, qqs = self.panther(x)
        return {'repr': out, 'qq': qqs}

    def forward(self, x):
        out = self.representation(x)
        return out['repr']
    
    def predict(self, data_loader, use_cuda=True):
        if self.mode == 'classification':
            output, y = predict_clf(self, data_loader.dataset, use_cuda=use_cuda)
        elif self.mode == 'survival':
            output, y = predict_surv(self, data_loader.dataset, use_cuda=use_cuda)
        elif self.mode == 'emb':
            output = predict_emb(self, data_loader.dataset, use_cuda=use_cuda)
            y = None
        else:
            raise NotImplementedError(f"Not implemented for {self.mode}!")
        
        return output, y