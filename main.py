
import argparse
import os

import torch
from torch.utils.data import DataLoader

from methods import SlideCoOp, SlideCoOpTopK, SLIP
from datasets import get_features_datasets, get_dataset_prompts
from networks import get_clip_network
from utils.utils import setup_experiment, seed_worker, get_split_experiment_dir
from networks.coop import PLIPModel
from networks.utils import build_lr_scheduler, RAdam, Lookahead


def get_opt_parser():

    parser = argparse.ArgumentParser("Baselines Parameters", add_help=False)

    # Experiment variables
    parser.add_argument("--dataroot", type=str, help="Root Directory of the Dataset", default="./data")
    parser.add_argument("--experiment-dir", type=str, help="Path to the experiment directory", default="./experiments/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", choices=("SlideCoOp", "SlideCoOpTopK", "SLIP"), default="SLIP")

    # Dataset variables
    parser.add_argument("--dataset", type=str, choices=("PatchGastricADC22", "DHMC", "TCGA"), default="PatchGastricADC22")
    parser.add_argument("--n-shots", type=str, choices=("1", "2", "4", "8", "16", "all"), default="1")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--context-size", type=int, default=1)
    parser.add_argument("--context-gain", type=float, default=0.01)

    # Optimization parameters
    parser.add_argument("--arch", type=str, choices=("CLIP", "BiomedCLIP", "PLIP", "CLIP-RN50"), default="CLIP")
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
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--temp", type=float, default=0.01)
    return parser


def run(opt):

    # Setup experiments
    experiment_dir = get_split_experiment_dir(opt)
    setup_experiment(opt, seed=opt.seed, experiment_dir=experiment_dir, batch_size=1)

    # Set datasets
    opt.dataroot = os.path.join(opt.dataroot, opt.dataset)
    datasets = get_features_datasets(root_dir=opt.dataroot, dataset=opt.dataset, features_type = opt.arch, n_shots=opt.n_shots)
    test_dataset = datasets["test"]
    train_dataset = datasets["train"]

    model, embed_dim, tokenizer, preprocess_train, preprocess_test = get_clip_network(opt.arch)
    model.cuda()
    prompts = get_dataset_prompts(opt.dataset)

    if opt.arch == "BiomedCLIP":
        model.dtype = torch.float32

    if opt.arch == "PLIP":
        model = PLIPModel(model, model.dtype)

    model.eval()

    # Set dataloaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=opt.n_shots not in ["1", "2"],
                        drop_last=False, num_workers=opt.num_workers, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                        drop_last=False, num_workers=opt.num_workers, worker_init_fn=seed_worker)

    if opt.method == "SLIP":
        trainer = SLIP(model, tokenizer, prompts["templates"], prompts["slide_classnames"],
                            prompts["tissue_classnames"], experiment_dir, context_size=opt.context_size,
                            context_gain=opt.context_gain, arch=opt.arch, temperature=opt.temp)
    elif opt.method == "SlideCoOp":
        trainer = SlideCoOp(model, tokenizer, prompts["templates"], prompts["slide_classnames"],
                            prompts["tissue_classnames"], experiment_dir, context_size=opt.context_size,
                            context_gain=opt.context_gain, arch=opt.arch, temperature=opt.temp)
    elif opt.method == "SlideCoOpTopK":
        trainer = SlideCoOpTopK(model, tokenizer, prompts["templates"], prompts["slide_classnames"],
                            prompts["tissue_classnames"], experiment_dir, context_size=opt.context_size,
                            context_gain=opt.context_gain, arch=opt.arch, k=opt.topk, temperature=opt.temp)

    trainer.test(test_loader, save_model=False)
    optimizer = torch.optim.SGD(trainer.prompt_learner.parameters(), lr=opt.lr)
    scheduler = None #build_lr_scheduler(optimizer=optimizer, max_epoch=opt.n_epochs, warmup_epoch=1, warmup_cons_lr=1e-5)
    trainer.train(train_loader, opt.n_epochs, test_loader, optimizer, scheduler=scheduler, save_every=20)
    trainer.test(test_loader, save_model=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Baselines Main', parents=[get_opt_parser()])
    opt = parser.parse_args()
    run(opt)
