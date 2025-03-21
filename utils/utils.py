
import os, re
import random
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as tvF
import torchvision

import numpy as np
import json

from tqdm import tqdm
from .distributed import init_distributed_mode
from typing import List, Union


def bold_str(str):
    return '\033[1m' + str + '\033[0m'

def normalize_fn(mean, std):
    return lambda x : tvF.normalize(x, mean, std)


def denormalize_fn(mean, std):
    return lambda x : tvF.normalize(x, -torch.tensor(mean)/torch.tensor(std), 1/torch.tensor(std))


def clip_by_norm(feats):
    with torch.no_grad():
        norm_feats = torch.norm(feats, p=2, dim=-1, keepdim=True)
        norm_feats = torch.clip(norm_feats, min=1.0)

    feats = feats / norm_feats
    return feats


def flatten_list(list_ : List[Union[str, List[str]]]):

    flatten_list = []
    for elem in list_:
        if isinstance(list_, str):
            flatten_list.append(elem)
        else:
            flatten_list += elem

    return flatten_list

def clip_gradient(optimizer):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                torch.nn.utils.clip_grad_norm_(param, 1.0, norm_type=2.0, error_if_nonfinite=True)


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]


def natural_sort(items):
    new_items = items.copy()
    new_items.sort(key=natural_keys)
    return new_items


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_split_experiment_dir(opt):
    experiment_dir = os.path.join(opt.experiment_dir, opt.dataset, opt.method, opt.arch, opt.n_shots, str(opt.seed))
    return experiment_dir


def setup_experiment(opt, seed:int, experiment_dir:str, batch_size:int):
    # Setup SEED
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    torchvision.torch.manual_seed(seed)
    torchvision.torch.cuda.manual_seed(seed)

    # Create directory
    ensure_dir(experiment_dir)

    # Init distribution
    opt.dist=False
    if torch.cuda.is_available():
        init_distributed_mode(opt)
        opt.dist = opt.world_size>1
        opt.batch_size = batch_size//opt.world_size
        cudnn.benchmark = True
    else:
        raise NotImplementedError("This script needs GPU. No GPU available or torch.cuda.is_available is False !")


def json_load(file_path):
    with open(file_path, 'r') as fp:
        return json.load(fp)
