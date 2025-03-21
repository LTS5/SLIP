
import os
import torch
import numpy as np
import pandas as pd


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricTracker:
    def __init__(self):
        self.metrics = {}

    def reset(self):
        self.metrics = {}

    def update_metrics(self, metric_dict, batch_size=1, compute_avg=True):
        for k, v in metric_dict.items():
            if k in self.metrics and compute_avg:
                self.metrics[k].update(v, batch_size)
            else:
                self.metrics[k] = AverageMeter()
                self.metrics[k].update(v, batch_size)

    def log(self, prefix="", suffix=""):
        str_ = prefix
        for k in self.metrics:
            str_ += " |{:s}:{:.4f}".format(k, self.__getitem__(k))
        str_ += suffix
        return str_

    def __getitem__(self, k):
        if k in self.metrics:
            return self.metrics[k].avg
        else:
            return 0.0

    def current_metrics(self):
        return {k:v.avg for k,v in self.metrics.items()}
    

class MetadataTracker:
    def __init__(self):
        self.metadata = {}

    def update_metadata(self, metadata_dict):
        for k, v in metadata_dict.items():
            if k in self.metadata:
                if isinstance(v, torch.Tensor):
                    self.metadata[k] = torch.cat([self.metadata[k], v])
                else:
                    self.metadata[k] = np.concatenate((self.metadata[k], np.array(v)))                   
            else:
                if isinstance(v, torch.Tensor):
                    self.metadata[k] = v.clone().detach()
                else:
                    self.metadata[k] = np.array(v)

    def to_pkl(self, keys, path):
        dict_to_pkl = {k : v for k,v in self.metadata.items() if k in keys}
        for k, v in dict_to_pkl.items():
            save_file = os.path.join(path,"{:s}.pth".format(k))
            torch.save(v, save_file)
        return dict_to_pkl

    def to_csv(self, keys, path, suffix=""):
        dict_to_csv = {k : v.numpy() for k,v in self.metadata.items() if k in keys and not isinstance(v, np.ndarray)}
        dict_to_csv.update({k : v for k,v in self.metadata.items() if k in keys and isinstance(v, np.ndarray)})
        df = pd.DataFrame.from_dict(dict_to_csv)
        df.to_csv(os.path.join(path, f"prediction{suffix}.csv"))
        return df

    def reset(self):
        self.metadata={}

    def __getitem__(self, key):
        return self.metadata[key]
