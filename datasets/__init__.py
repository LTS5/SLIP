
from .gastric import get_gastric_datasets, get_features_gastric_datasets
from .lung import get_lung_datasets, get_features_lung_datasets
from .tcga import get_tcga_datasets
from .prompt import PROMPTS


def get_datasets(root_dir : str, dataset : str, n_shots : int, image_size : int, transform = None):

    if dataset == "PatchGastricADC22":
        datasets = get_gastric_datasets(root_dir, n_shots, image_size, transform)
        datasets["n_classes"] = 3
    elif dataset == "DHMC":
        datasets = get_lung_datasets(root_dir, n_shots, image_size, transform)
        datasets["n_classes"] = 3
    else:
        raise NotImplementedError

    return datasets


def get_features_datasets(root_dir : str, dataset : str, features_type : str, n_shots : int):

    if dataset == "PatchGastricADC22":
        datasets = get_features_gastric_datasets(root_dir, features_type, n_shots)
        datasets["n_classes"] = 3
    elif dataset == "DHMC":
        datasets = get_features_lung_datasets(root_dir, features_type, n_shots)
        datasets["n_classes"] = 3
    elif dataset == "TCGA":
        datasets = get_tcga_datasets(root_dir, n_shots)
        datasets["n_classes"] = 2
    else:
        raise NotImplementedError

    return datasets


def get_dataset_prompts(dataset : str):
    prompts = {"templates" : PROMPTS[dataset]["templates"]}
    prompts.update(PROMPTS[dataset])
    return prompts
