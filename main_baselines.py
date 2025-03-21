
import argparse
import os
import torch
import time
import numpy as np

from torch.utils.data import DataLoader

from methods import CLIP, CLIPTopK, CLAM, LinearProbe, TransMILTrainer, SlideCLIP, CITE, PatchCoOp, PantherTrainer
from datasets import get_features_datasets, get_dataset_prompts, get_datasets
from networks import get_clip_network, get_ctranspath
from networks.clam import CLAM_MB, CLAM_SB
from networks.coop import PLIPModel
from networks.transmil import TransMIL
from networks.PANTHER.utils.proto_utils import cluster
from networks.PANTHER.tokenizer import PrototypeTokenizer
from networks.PANTHER.model_PANTHER import PANTHER
from utils.utils import setup_experiment, seed_worker, get_split_experiment_dir


def get_opt_parser():

    parser = argparse.ArgumentParser("Baselines Parameters", add_help=False)

    # Experiment variables
    parser.add_argument("--dataroot", type=str, help="Root Directory of the Dataset", default="./data")
    parser.add_argument("--experiment-dir", type=str, help="Path to the experiment directory", default="./experiments/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", type=str, choices=("CLIP", "CLIPTopk", "CLAM", "Linear", "TransMIL", "SlideCLIP", "CITE", 
                                                       "PatchCoOp", "Panther"), default="CLIP")

    # Dataset variables
    parser.add_argument("--dataset", type=str, choices=("PatchGastricADC22", "DHMC", "TCGA"), default="PatchGastricADC22")
    parser.add_argument("--n-shots", type=str, choices=("1", "2", "4", "8", "16", "all"), default="1")
    parser.add_argument("--image-size", type=int, default=224)

    # Optimization parameters
    parser.add_argument("--arch", type=str, choices=("CLIP", "CTransPath", "BiomedCLIP", "PLIP", "CLIP-RN50"), default="CLIP")
    parser.add_argument("--num-workers", type=int, default=8)

    # Distributed training parameters
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--dist-url", type=str, default="env://")
    parser.add_argument("--port", type=str, default="29500")

    # Proto Prompt parameters
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--n-iters", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--topk", type=int, default=50)

    return parser


def is_zeroshot(method):
    return method in ["CLIP", "CLIPTopk", "SlideCLIP"]

def is_features_method(method):
    return method in ["CLIP", "CLIPTopk", "CLAM", "Linear", "TransMIL", "SlideCLIP", "PatchCoOp", "Panther"]


def run(opt):

    # Set default args
    if is_zeroshot(opt.method):
        opt.n_shots = "0"

    # Setup experiments
    experiment_dir = get_split_experiment_dir(opt)
    setup_experiment(opt, seed=opt.seed, experiment_dir=experiment_dir, batch_size=opt.batch_size)

    # Get network
    if opt.arch == "CTransPath":
        model = get_ctranspath("./ctranspath.pth")
        model.dtype = torch.float32
        model.cuda()
        preprocess_test = None
        embed_dim = 768
    else:
        model, embed_dim, tokenizer, preprocess_train, preprocess_test = get_clip_network(opt.arch)
        model.cuda()
        prompts = get_dataset_prompts(opt.dataset)

        if opt.arch == "BiomedCLIP":
            model.dtype = torch.float32 # type: ignore

    if opt.arch == "PLIP":
        model = PLIPModel(model, model.dtype)

    # Set datasets
    opt.dataroot = os.path.join(opt.dataroot, opt.dataset)
    if is_features_method(opt.method):
        datasets = get_features_datasets(root_dir=opt.dataroot, dataset=opt.dataset, features_type = opt.arch, n_shots=opt.n_shots) # type: ignore

        test_dataset = datasets["test"]
        train_dataset = datasets["train"]
        # train_patch_dataset = datasets["train_patch"]

        if int(opt.n_shots) > 0:
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                            drop_last=False, num_workers=opt.num_workers, worker_init_fn=seed_worker)
        # train_patch_loader = DataLoader(train_patch_dataset, batch_size=128, shuffle=True,
        #                     drop_last=False, num_workers=opt.num_workers, worker_init_fn=seed_worker)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                            drop_last=False, num_workers=opt.num_workers, worker_init_fn=seed_worker)
    else:
        datasets = get_datasets(root_dir=opt.dataroot, dataset=opt.dataset, n_shots=opt.n_shots, image_size=opt.image_size, transform=preprocess_test) # type: ignore

        test_dataset = datasets["test"]
        train_dataset = datasets["train"]
        import torchvision.transforms as transforms
        import random
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, 
                        drop_last=False, num_workers=opt.num_workers, worker_init_fn=seed_worker)
        transform = [
                transforms.Resize((224, 224)),  # seems original code does not contain resize
                # transforms.CenterCrop(224),  # try this?
                transforms.ColorJitter(brightness=0.2, saturation=(0, 0.2), hue=0.1), # type: ignore
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5)
        ]
        r = random.randint(0, 3)
        for _ in range(r):
            transform.append(transforms.RandomRotation((90, 90)))
        transform.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transform = transforms.Compose(transform)
        train_loader.dataset.transform = transform # type: ignore
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False,
                            drop_last=False, num_workers=opt.num_workers, worker_init_fn=seed_worker)

    t1 = time.time()
    ################################## (a) MIL Methods #############################################
    if opt.method == "Linear":
        model = torch.nn.Linear(embed_dim, datasets["n_classes"], dtype=model.dtype).cuda().to(torch.float32)
        trainer = LinearProbe(model, experiment_dir)
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        trainer.train(train_loader, opt.n_epochs, test_loader, optimizer)

    elif opt.method == "CLAM":
        from networks.utils import RAdam, Lookahead
        model = CLAM_SB(n_classes=datasets["n_classes"], embed_dim=embed_dim, subtyping=True).to(model.dtype).cuda() # type: ignore
        trainer = CLAM(model=model, experiment_dir=experiment_dir)
        optimizer = RAdam(model.parameters(), lr=1e-4, weight_decay=1e-5) # type: ignore
        optimizer = Lookahead(optimizer)
        trainer.train(train_loader, opt.n_epochs, test_loader, optimizer)

    elif opt.method == "TransMIL":
        from networks.utils import RAdam, Lookahead
        model = TransMIL(n_classes=datasets["n_classes"], embed_dim=embed_dim).cuda()
        trainer = TransMILTrainer(model, experiment_dir)
        optimizer = RAdam(model.parameters(), lr=1e-4, weight_decay=1e-5) # type: ignore
        optimizer = Lookahead(optimizer)
        trainer.train(train_loader, opt.n_epochs, test_loader, optimizer)

    elif opt.method == "Panther":
        from networks.PANTHER.model_PANTHER import PANTHER
        from networks.utils import RAdam, Lookahead
        proto_path = os.path.join(*experiment_dir.split("/")[:-1], "protos.npy")
        # if not os.path.exists(proto_path):
        _, weights = cluster(train_loader,
                        n_proto=2,
                        n_iter=5000,
                        n_init=50,
                        feature_dim=embed_dim,
                        mode="kmeans",
                        n_proto_patches=10000,
                        use_cuda=True if torch.cuda.is_available() else False)
        np.save(proto_path, weights.squeeze())
        encoder = PANTHER(embed_dim, 1, 2, True, proto_path, 1, 0.001,
                        "allcat", 0.1, False, "classification").cuda().to(model.dtype) # type: ignore
        model = torch.nn.Linear(2050, datasets["n_classes"], bias=False).cuda()
        trainer = PantherTrainer(model, encoder, experiment_dir)
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-5)
        optimizer = RAdam(model.parameters(), lr=1e-4, weight_decay=1e-5) # type: ignore
        optimizer = Lookahead(optimizer)
        trainer.train(train_loader, opt.n_epochs, test_loader, optimizer)

    ################################# (b) VLM Methods ###############################################

    elif opt.method == "CLIP":
        # Set Network
        labels = []
        for data in test_loader:
            _, label, _ = data
            labels.append(label[0].item())
            
        import numpy as np
        print(np.bincount(np.array(labels), minlength=3))
        trainer = CLIP(model=model, tokenizer=tokenizer, templates=prompts["templates"],
                        classnames=prompts["slide_classnames"], experiment_dir=experiment_dir)

    elif opt.method == "CLIPTopk":
        trainer = CLIPTopK(model=model, tokenizer=tokenizer, templates=prompts["templates"],
                       classnames=prompts["slide_classnames"], experiment_dir=experiment_dir, k=opt.topk)

    elif opt.method == "SlideCLIP":
        trainer = SlideCLIP(model=model, tokenizer=tokenizer, templates=prompts["templates"],
                            slide_classnames=prompts["slide_classnames"], tissue_classnames=prompts["tissue_classnames"],
                                experiment_dir=experiment_dir)

    elif opt.method == "CITE":
        import math
        trainer = CITE(model=model, tokenizer=tokenizer, templates=["{}"],
                            classnames=prompts["slide_classnames"], experiment_dir=experiment_dir)
        n_epochs = int(math.ceil(opt.n_iters/len(train_loader)))
        trainer.train(train_loader, test_loader, n_epochs=n_epochs)

    elif opt.method == "PatchCoOp":
        import math
        trainer = PatchCoOp(model, tokenizer, prompts["templates"], prompts["slide_classnames"], experiment_dir, 1,
                            context_gain=0.01, arch=opt.arch)
        n_epochs = int(math.ceil(opt.n_iters/len(train_patch_loader)))
        trainer.train(train_patch_loader, test_loader, n_epochs=n_epochs, lr=2e-3, wd=0.0)

    else:
        raise NotImplementedError
    t2 = time.time()
    print("TIME : ", t2-t1)
    trainer.test(test_loader, save_model=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Baselines Main', parents=[get_opt_parser()])
    opt = parser.parse_args()
    run(opt)
