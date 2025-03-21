
import os

import torch
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image


def get_gastric_samples(root : str, samples_filename : str):
    img_files = []
    labels = []
    wsis = []

    with open(os.path.join(root, "data/gastric_cls3_ann", samples_filename), "r") as f:
        for line in f.readlines():
            img_file, label = line.split(" ")
            img_files.append(os.path.join(root, "data/patches_captions", img_file))
            labels.append(int(label.replace("\n", "")))
            wsis.append(str(img_file.split("_")[0]))

    return img_files, labels, wsis


class GastricPatch(Dataset):

    def __init__(self, root : str, samples_filename : str, transform) -> None:
        super().__init__()

        self.root = root
        self.transform = transform
        self.img_files, self.labels, self.wsi_ids = get_gastric_samples(root, samples_filename)
        

    def __getitem__(self, index):
        img_file = self.img_files[index]
        wsi_id = self.wsi_ids[index]
        img = Image.open(img_file).convert("RGB")
        img = self.transform(img)
        label = self.labels[index]
        return img, label, wsi_id
    
    def __len__(self):
        return len(self.img_files)
    

class GastricSlideFeatures(Dataset):

    def __init__(self, root : str, feats_filename : str, samples_filename : str) -> None:
        super().__init__()

        feats_filepath = os.path.join(root, "features", feats_filename)
        wsi_feats = torch.load(feats_filepath)
        _, _, wsi_ids = get_gastric_samples(root, samples_filename)
        unique_wsi_ids = np.unique(np.array(wsi_ids))

        self.wsi_ids = unique_wsi_ids
        self.feats = [wsi_feats[wsi_id]["feats"] for wsi_id in unique_wsi_ids]
        self.labels = [wsi_feats[wsi_id]["label"] for wsi_id in unique_wsi_ids]

    def __len__(self):
        return len(self.feats)
    
    def __getitem__(self, index):
        wsi_id = self.wsi_ids[index]
        label = self.labels[index]
        feats = self.feats[index]
        return feats, label, wsi_id
    

class GastricPatchFeatures(GastricSlideFeatures):

    def __init__(self, root : str, feats_filename : str, samples_filename : str) -> None:
        super().__init__(root, feats_filename, samples_filename)

        self.wsi_ids = np.concatenate([wsi_id.repeat(feats.size(0)) for (wsi_id, feats) in zip(self.wsi_ids, self.feats)])
        self.labels = torch.cat([label.repeat(feats.size(0)) for (label, feats) in zip(self.labels, self.feats)])
        self.feats = torch.cat(self.feats)


def get_gastric_datasets(root_dir : str, n_shots : str, image_size : int = 150, transform = None):

    datasets = {}

    # Load train and validatin files
    train_file = f"train_{n_shots}_0.2.txt" if n_shots != "0" else "train_all_0.2.txt"
    test_file = "val_0.8.txt"

    # Create the datasets
    if transform is None:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
        ])
    
    datasets["train"] = GastricPatch(root_dir, train_file, transform)
    datasets["test"] = GastricPatch(root_dir, test_file, transform)

    return datasets


def get_features_gastric_datasets(root_dir : str, features_type : str, n_shots : str):

    datasets = {}

    # Load train and validatin files
    train_file = f"train_{n_shots}_0.2.txt" if n_shots != "0" else "train_all_0.2.txt"
    test_file = "val_0.8.txt"

    datasets["train"] = GastricSlideFeatures(root_dir, f"{features_type}_train.pth", train_file)
    datasets["train_patch"] = GastricPatchFeatures(root_dir, f"{features_type}_train.pth", train_file)
    datasets["test"] =GastricSlideFeatures(root_dir, f"{features_type}_test.pth", test_file)

    return datasets






