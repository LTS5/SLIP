
import argparse
import os

import torch
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_datasets
from networks import get_clip_network, get_ctranspath
from utils.utils import seed_worker, ensure_dir
from utils.tracker import MetadataTracker


def get_opt_parser():

    parser = argparse.ArgumentParser("Baselines Parameters", add_help=False)

    # Experiment variables
    parser.add_argument("--dataroot", type=str, help="Root Directory of the Dataset", default="./data")
    parser.add_argument("--experiment-dir", type=str, help="Path to the experiment directory", default="./experiments/")
    parser.add_argument("--seed", type=int, default=0)

    # Dataset variables
    parser.add_argument("--dataset", type=str, default="PatchGastricADC22")
    parser.add_argument("--image-size", type=int, default=224)

    # Optimization parameters
    parser.add_argument("--arch", type=str, choices=("CLIP", "CTransPath", "BiomedCLIP", "PLIP"), default="CLIP")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=12)

    # Distributed training parameters
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--dist-url", type=str, default="env://")
    parser.add_argument("--port", type=str, default="29500")

    return parser

@torch.no_grad()
def save_features(forward_fn, loader, feats_folder : str, feats_filename : str):
    feats_filepath = os.path.join(feats_folder, feats_filename)
    metadata_tracker = MetadataTracker()
    if not os.path.exists(feats_filepath):
        ensure_dir(feats_folder)
        for i, (image, label, wsi) in enumerate(tqdm(loader)):
            image = image.cuda()
            feat = forward_fn(image)
            metadata_tracker.update_metadata({
                "feat" : feat.cpu(),
                "label" : label,
                "wsi_id" : wsi
            })

        unique_wsi_ids = np.unique(metadata_tracker["wsi_id"])
        labels = metadata_tracker["label"]
        feats = metadata_tracker["feat"]
        wsi_ids = metadata_tracker["wsi_id"]
        feats_dict = {}
        for wsi_id in unique_wsi_ids:
            wsi_label = labels[wsi_ids==wsi_id][0]
            wsi_feats = feats[wsi_ids==wsi_id]
            feats_dict[wsi_id] = {"feats" : wsi_feats, "label" : wsi_label}

        torch.save(feats_dict, feats_filepath)


def run(opt):

    # Set model
    opt.dataroot = os.path.join(opt.dataroot, opt.dataset)
    if opt.arch == "CTransPath":
        model = get_ctranspath("./ctranspath.pth")
        preprocess_test = None
    else:
        model, _, _, _, preprocess_test = get_clip_network(opt.arch)


    # Set datasets
    datasets = get_datasets(root_dir=opt.dataroot, dataset=opt.dataset, n_shots="all", image_size=opt.image_size, transform=preprocess_test)
    test_dataset = datasets["test"]
    train_dataset = datasets["train"]

    # Set dataloaders
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False,
                        drop_last=False, num_workers=opt.num_workers, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False,
                        drop_last=False, num_workers=opt.num_workers, worker_init_fn=seed_worker)

    # Set forward function
    if opt.arch == "CLIP" or opt.arch == "BiomedCLIP":
        forward_fn = lambda image : model.encode_image(image)
    elif opt.arch  == "PLIP":
        forward_fn = lambda image : model.get_image_features(image)
    elif opt.arch == "CTransPath":
        forward_fn = lambda image : model(image)
    else:
        raise NotImplementedError

    # Extract the features
    model.cuda()
    model.eval()
    save_folder = os.path.join(opt.dataroot, "features")
    ensure_dir(save_folder)
    print(f"Extracting train features with {opt.arch} !")
    save_features(forward_fn, train_loader, feats_folder=save_folder, feats_filename=f"{opt.arch}_train2.pth")
    print(f"Extracting test features with {opt.arch} !")
    save_features(forward_fn, test_loader, feats_folder=save_folder, feats_filename=f"{opt.arch}_test2.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Extract Features Main', parents=[get_opt_parser()])
    opt = parser.parse_args()
    run(opt)
