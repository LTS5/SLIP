
from typing import List
from tqdm import tqdm

import torch
import torch.nn.functional as F

from .baselines import MILClassifier, CLIP
from networks.coop import PromptLearner, TextEncoder, BiomedPromptLearner, BiomedTextEncoder, PLIPPromptLearner, PLIPTextEncoder


class SlideCoOp(MILClassifier):
    """Baseline module based on clip
    """
    def __init__(self, model, tokenizer, templates : List[str], slide_classnames : List[str],
                 tissue_classnames : List[str], experiment_dir : str, context_size : int, temperature : float = 0.01,
                 imgsize : int = 224, context_init=None, cls_token_position : str = "end", context_gain=1.0, arch="CLIP") -> None:
        super(MILClassifier, self).__init__(model, experiment_dir)

        # Set attributes
        self.model.eval()
        self.tokenizer = tokenizer
        self.templates = templates
        self.classnames = [[templates[0].format(x[0])] for x in slide_classnames]
        self.tissue_classnames = tissue_classnames
        self.n_classes = len(self.classnames)
        self.experiment_dir = experiment_dir

        # Set tissue classifier
        self.model.eval()
        self.tissue_weights = self.zeroshot_classifier(self.tissue_classnames) # (D, C)
        self.temperature = temperature

        # Set prompt learner
        if arch == "CLIP" or arch=="CLIP-RN50":
            self.prompt_learner = PromptLearner(self.classnames, self.model.cpu(), context_size=context_size,
                                imgsize=imgsize, context_init=context_init, cls_token_position=cls_token_position, context_gain=context_gain).cuda()
        elif arch == "BiomedCLIP":
            self.prompt_learner = BiomedPromptLearner(self.classnames, self.model.cpu(),
                                                      context_size=context_size,  imgsize=imgsize, cls_token_position="end", tokenizer=tokenizer, per_class_ctx=True,
                                                      context_gain=context_gain).cuda()
        elif arch == "PLIP":
            self.prompt_learner = PLIPPromptLearner(self.classnames, self.model.cpu(),
                                                      context_size=context_size,  imgsize=imgsize, cls_token_position="end", tokenizer=tokenizer, per_class_ctx=True,
                                                      context_gain=context_gain).cuda()
        else:
            raise NotImplementedError

        self.model.cuda()
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        if arch == "CLIP" or arch == "CLIP-RN50":
            self.text_encoder = TextEncoder(self.model).cuda()
        elif arch == "PLIP":
            self.text_encoder = PLIPTextEncoder(self.model).cuda()
        elif arch == "BiomedCLIP":
            self.text_encoder = BiomedTextEncoder(self.model).cuda()


    def zeroshot_classifier(self, classnames):
        all_classnames = []
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = []
                for class_i in classname:
                    texts += [template.format(class_i) for template in self.templates] #format with class
                all_classnames += texts
                texts = self.tokenizer(texts)
                class_embeddings = self.model.encode_text(texts.cuda()) #embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
            print(all_classnames)
        return zeroshot_weights


    def compute_loss(self, wsi_feats, wsi_label):
        wsi_feats = wsi_feats.cuda().squeeze()
        wsi_label = wsi_label.cuda().squeeze()
        wsi_pred = self.forward(wsi_feats)
        loss = F.cross_entropy(wsi_pred, wsi_label)
        return loss, wsi_pred.unsqueeze(0)


    def forward(self, slide_features, eval=False):
        # Input: image_features (N, D)
        slide_features = F.normalize(slide_features, dim=1)

        # Compute slide classifier weights
        if eval:
            slide_weights = self.slide_weights
        else:
            prompts = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts)
            slide_weights = F.normalize(text_features, dim=1).T
        slide_logits = slide_features @ slide_weights / self.temperature

        return slide_logits.mean(0)

    def predict_slide(self, slide_features):
        return self.forward(slide_features.squeeze()).unsqueeze(0)

    def test(self, test_loader, epoch="last", save_model: bool = False):
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        # print(text_features.shape)
        self.slide_weights = F.normalize(text_features, dim=1).T
        return super().test(test_loader, epoch, save_model)


class SlideCoOpTopK(SlideCoOp):

    def __init__(self, model, tokenizer, templates: List[str], slide_classnames: List[str], tissue_classnames: List[str], 
                 experiment_dir: str, context_size: int, temperature : float = 0.01, 
                 imgsize: int = 224, context_init=None, cls_token_position: str = "end", 
                 context_gain=1, arch="CLIP", k=50) -> None:
        super().__init__(model, tokenizer, templates, slide_classnames, tissue_classnames, experiment_dir,
                         context_size, temperature, imgsize, context_init, cls_token_position, context_gain, arch)
        self.k = k

    def forward(self, slide_features, eval=False):
        # Input: image_features (N, D)
        n, _ = slide_features.size()
        slide_features = F.normalize(slide_features, dim=1)

        # Compute slide classifier weights
        if eval:
            slide_weights = self.slide_weights
        else:
            prompts = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts)
            slide_weights = F.normalize(text_features, dim=1)
        tissue_class_sim = slide_features @ slide_weights
        top_k_sim = torch.topk(tissue_class_sim, k=min(self.k,n) , dim=0).values

        return top_k_sim.mean(0) / self.temperature


class SLIP(SlideCoOp):
    """Baseline module based on clip
    """
    def __init__(self, model, tokenizer, templates : List[str], slide_classnames : List[str],
                 tissue_classnames : List[str], experiment_dir : str, context_size : int, temperature : float = 0.01,
                 imgsize : int = 224, context_init=None, cls_token_position : str = "end", context_gain=1.0, arch="CLIP") -> None:
        super().__init__(model, tokenizer, templates, slide_classnames, tissue_classnames, experiment_dir,
                         context_size, temperature, imgsize, context_init, cls_token_position, context_gain, arch)


    def compute_loss(self, wsi_feats, wsi_label):
        wsi_feats = wsi_feats.cuda().squeeze()
        wsi_label = wsi_label.cuda().squeeze()
        wsi_pred, cross_corr = self.forward(wsi_feats)

        # contrastive
        logits = cross_corr.view(-1) # c x c
        n_cls = cross_corr.size(0)
        positive_index = wsi_label * n_cls + wsi_label
        negative_index = [i for i  in range(n_cls * n_cls) if i != positive_index]
        positive_logit = logits[positive_index:(positive_index + 1)]
        negative_logits = logits[negative_index]
        _logits = torch.cat([positive_logit, negative_logits])
        contrastive = F.cross_entropy(_logits / self.temperature, torch.tensor(0, device=_logits.device))
        loss = contrastive

        return loss, wsi_pred.unsqueeze(0)


    def forward(self, slide_features, eval=False, return_metadata=False):
        # Input: image_features (N, D)
        slide_features = F.normalize(slide_features, dim=1)
        n, d = slide_features.size()

        # Compute slide classifier weights
        if eval:
            slide_weights = self.slide_weights
        else:
            prompts = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts)
            slide_weights = F.normalize(text_features, dim=1).T
        tissue_weights = self.tissue_weights

        with torch.no_grad():
            s_patch_tissue = F.softmax(slide_features @ tissue_weights / self.temperature, dim=1) # (N, C')
            s_slide_tissue = F.softmax(tissue_weights.T @ slide_weights / self.temperature, dim=0) # (C', C)
            s_attention = s_patch_tissue @ s_slide_tissue # (N, C)
            slide_class_preds = F.normalize(slide_features.T @ s_attention, dim=0) # (D, C)

        cross_corr_pred_gt = slide_class_preds.T @ slide_weights
        slide_logits = torch.diag(cross_corr_pred_gt) / self.temperature

        if return_metadata:
            return slide_logits, cross_corr_pred_gt, s_attention, s_slide_tissue

        return slide_logits, cross_corr_pred_gt

    def predict_slide(self, slide_features):
        return self.forward(slide_features.squeeze(), eval=True)[0].unsqueeze(0)
