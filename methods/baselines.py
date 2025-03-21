
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch
import numpy as np

from tqdm import tqdm
from typing import List
from functools import reduce

import torch
import torch.nn.functional as F
from  torch.cuda.amp import autocast
from operator import mul

from networks.coop import PromptLearner, TextEncoder, BiomedPromptLearner, BiomedTextEncoder
from networks.utils import build_lr_scheduler
from utils.tracker import MetadataTracker, MetricTracker
from utils.result import classification_results, accuracy


class SlideClassifier:

    def __init__(self, model, experiment_dir : str) -> None:

        self.model=model
        self.experiment_dir = experiment_dir
        self.metadata_tracker = MetadataTracker()
        self.metric_tracker = MetricTracker()

    def predict_patch(self, image_feature):
        return self.model.fc(image_feature)

    def predict_slide(self, slide_features):
        preds = F.softmax(self.predict_patch(slide_features), dim=1)
        return preds.mean(0) # Default Soft-Voting Predictions

    def load_state_dict(self, ckpt_path):
        state_dict = torch.load(ckpt_path)
        self.model = state_dict["model"]

    def state_dict(self):
        return {"model" : self.model.state_dict()}

    def train(self, train_loader, n_epochs, test_loader, optimizer, scheduler=None):
        pass

    @torch.no_grad()
    def test(self, test_loader, epoch="last", save_model : bool = False):

        # Prepare outout folder
        self.model.eval()
        self.metadata_tracker.reset()
        all_logit = []

        for wsi_feats, wsi_label, wsi_id in test_loader:

            wsi_feats = wsi_feats.cuda()
            wsi_logit = self.predict_slide(wsi_feats)
            wsi_pred = wsi_logit.argmax(1)
            all_logit.append(wsi_logit.cpu())

            self.metadata_tracker.update_metadata({
                "label" : wsi_label,
                "pred" : wsi_pred.cpu(),
                "conf" : wsi_logit[:,1].cpu(),
                "wsi" : wsi_id,
            })

        suffix = f"_{epoch}"
        self.metadata_tracker.to_csv(["label", "pred", "conf", "wsi"], self.experiment_dir, suffix=suffix)
        print(roc_auc_score(self.metadata_tracker["label"], self.metadata_tracker["conf"]))

        # Compute results
        results = classification_results(self.metadata_tracker["pred"], self.metadata_tracker["label"], self.metadata_tracker["conf"])
        print(results, "\n")

        # Save model
        if save_model:
            torch.save(self.state_dict(), f"{self.experiment_dir}/model{suffix}.pth")


########################################### (A) MIL BASELINES ############################################

class MILClassifier(SlideClassifier):

    def __init__(self, model, experiment_dir: str) -> None:
        super().__init__(model, experiment_dir)

    def compute_loss(self, wsi_feats, wsi_label):
        return

    def train(self, train_loader, n_epochs, test_loader, optimizer, scheduler=None, save_every : int = 20):
        # Train for n_epochs
        for e in range(n_epochs):
            self.metric_tracker.reset()
            pbar = tqdm(train_loader, dynamic_ncols=True)

            for wsi_feats, wsi_label, _ in pbar:

                loss, pred = self.compute_loss(wsi_feats, wsi_label)
                if loss is None:
                    continue

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.metric_tracker.update_metrics({
                    "loss": loss.item(),
                    "acc" : accuracy(pred.argmax(1).cpu(), wsi_label)
                }, batch_size=1)
                pbar.set_description(self.metric_tracker.log(prefix=f"Epoch {e+1:d} - lr={optimizer.param_groups[0]['lr']:.7f}"))

            if scheduler is not None:
                scheduler.step()
            if ((e+1)%save_every)==0:
                print(f"Test Epoch {e+1:d}")
                self.test(test_loader, epoch=e+1, save_model=False)


class LinearProbe(MILClassifier):

    def __init__(self, model, experiment_dir: str) -> None:
        super().__init__(model, experiment_dir)

    def predict_slide(self, slide_features):
        global_feature = torch.mean(slide_features.to(torch.float32), dim=1)
        return self.model(global_feature)

    def compute_loss(self, wsi_feats, wsi_label):
        pred = self.predict_slide(wsi_feats.cuda().to(torch.float32))
        loss = F.cross_entropy(pred, wsi_label.cuda())
        return loss, pred

# class LinearProbe(MILClassifier):

#     def __init__(self, model, experiment_dir: str) -> None:
#         super().__init__(model, experiment_dir)


#     def train(self, train_loader, n_epochs, test_loader, optimizer, scheduler=None, save_every: int = 20):
#         all_feats = []
#         all_labels = []
#         for feats, labels, _ in train_loader:
#             feats = feats.to(torch.float32)
#             all_feats.append(feats.squeeze().mean(dim=0))
#             all_labels.append(labels.squeeze())

#         all_feats = torch.stack(all_feats)
#         all_labels = torch.stack(all_labels)
#         self.classifier = LogisticRegression(random_state=0, C=0.3, max_iter=100000, verbose=1)
#         self.classifier.fit(all_feats, all_labels)

#     def predict_slide(self, slide_features):
#         global_feature = torch.mean(slide_features, dim=1)
#         return torch.from_numpy(self.classifier.predict_proba(global_feature.cpu().numpy()))


class CLAM(MILClassifier):

    def __init__(self, model, experiment_dir: str) -> None:
        super().__init__(model, experiment_dir)

    def predict_slide(self, slide_features):
        self.model.eval()
        logits, _, _, _, _ = self.model(slide_features.squeeze())
        return logits

    def compute_loss(self, wsi_feats, wsi_label):
        self.model.train()
        bag_weight = 0.7
        if wsi_feats.size(1) < self.model.k_sample:
            return None, None

        feats = wsi_feats.cuda().squeeze()
        label = wsi_label.cuda()
        logits, _, _, _, instance_dict = self.model(feats, label=label, instance_eval=True)
        loss = F.cross_entropy(logits, label)
        instance_loss = instance_dict['instance_loss']
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss
        return total_loss, logits


class TransMILTrainer(MILClassifier):

    def __init__(self, model, experiment_dir: str) -> None:
        super().__init__(model, experiment_dir)

    def predict_slide(self, slide_features):
        self.model.eval()
        return self.model(data=slide_features.cuda())

    def compute_loss(self, wsi_feats, wsi_label):
        self.model.train()
        results_dict = self.model(data=wsi_feats.cuda())
        loss = F.cross_entropy(results_dict['logits'], wsi_label.cuda())
        return loss, results_dict["logits"]


class PantherTrainer(MILClassifier):

    def __init__(self, model, encoder, experiment_dir: str) -> None:
        super().__init__(model, experiment_dir)
        self.encoder = encoder
        self.encoder.eval()
        self.encoder.requires_grad_(False)

    def predict_slide(self, slide_features):
        self.model.eval()
        wsi_feats = self.encoder(slide_features.cuda().float())
        out = self.model(wsi_feats)
        return out

    def compute_loss(self, wsi_feats, wsi_label):
        self.model.train()
        wsi_feats = self.encoder(wsi_feats.cuda().float())
        out = self.model(wsi_feats)
        loss = F.cross_entropy(out, wsi_label.cuda())
        return loss, out


######################################## (b) VLM Method #########################################


class CLIP(SlideClassifier):
    """Baseline module based on clip
    """
    def __init__(self, model, tokenizer, templates : List[str],
                 classnames : List[str], experiment_dir : str) -> None:
        super().__init__(model, experiment_dir)

        # Set attributes
        self.model.eval()
        self.tokenizer = tokenizer
        self.templates = templates
        self.classnames = classnames
        self.n_classes = len(self.classnames)

        # Set classifier
        self.zero_shot_weights = self.zeroshot_classifier(self.classnames)

    def zeroshot_classifier(self, classnames):
        all_classnames = []
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = []
                for class_i in classname:
                    texts += [template.format(class_i) for template in self.templates] #format with class
                all_classnames += texts
                tokenized_texts = self.tokenizer(texts)
                class_embeddings = self.model.encode_text(tokenized_texts.cuda()) #embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        print("Prompts", all_classnames)
        return zeroshot_weights

    def predict_slide(self, slide_features):
        slide_features = F.normalize(slide_features.squeeze(), dim=1)
        logits = self.model.logit_scale.exp() * slide_features @ self.zero_shot_weights
        probs = F.softmax(logits, dim=1)
        return probs.mean(0, keepdim=True)


class CLIPTopK(CLIP):

    def __init__(self, model, tokenizer, templates: List[str], classnames: List[str], experiment_dir: str, k : int = 10) -> None:
        super().__init__(model, tokenizer, templates, classnames, experiment_dir)
        self.k = k

    def predict_slide(self, slide_features):
        # Input: slide_features (N, D)
        slide_features = slide_features.squeeze()
        n, d = slide_features.size()
        slide_features = F.normalize(slide_features, dim=1)
        tissue_class_sim = slide_features @ self.zero_shot_weights # (N, C)

        # Choose topk per class
        top_k_sim = torch.topk(tissue_class_sim, k=min(self.k,n) , dim=0).values
        return top_k_sim.mean(0, keepdim=True)


class SlideCLIP(CLIP):
    def __init__(self, model, tokenizer, templates: List[str], slide_classnames : List[str],
                 tissue_classnames : List[str], experiment_dir: str) -> None:
        super().__init__(model, tokenizer, templates, slide_classnames, experiment_dir)
        self.tissue_weights = self.zeroshot_classifier(tissue_classnames)

    def predict_slide(self, slide_features):
        # Input: image_features (N, D)
        slide_features = slide_features.squeeze()
        slide_features = F.normalize(slide_features, dim=1)
        slide_weights = self.zero_shot_weights
        tissue_weights = self.tissue_weights

        with torch.no_grad():
            s_patch_tissue = F.softmax(self.model.logit_scale.exp() * slide_features @ tissue_weights, dim=1) # (N, C')
            s_slide_tissue = F.softmax(self.model.logit_scale.exp() * tissue_weights.T @ slide_weights, dim=0) # (C', C)
            s_attention = s_patch_tissue @ s_slide_tissue # (N, C)
            slide_class_preds = F.normalize(slide_features.T @ s_attention, dim=0) # (D, C)

        cross_corr_pred_gt = slide_class_preds.T @ slide_weights
        slide_logits = self.model.logit_scale.exp() * torch.diag(cross_corr_pred_gt)

        return slide_logits.unsqueeze(0)


class CoOp(CLIP):
    """Baseline module based on clip
    """
    def __init__(self, model, tokenizer, templates : List[str], slide_classnames : List[str],
                 tissue_classnames : List[str], experiment_dir : str, context_size : int,
                 imgsize : int = 224, context_init=None, cls_token_position : str = "end", context_gain=1.0, arch="CLIP") -> None:
        super().__init__(model, tokenizer, templates, slide_classnames, experiment_dir)
        
        # Set attributes
        self.classnames = [[templates[0].format(x[0])] for x in slide_classnames]
        self.tissue_classnames = tissue_classnames
        self.n_classes = len(self.classnames)
        self.experiment_dir = experiment_dir

        # Set prompt learner
        if arch == "CLIP":
            self.prompt_learner = PromptLearner(self.classnames, self.model.cpu(), context_size=context_size,
                            imgsize=imgsize, context_init=context_init, cls_token_position=cls_token_position, context_gain=context_gain).cuda()
        
        elif arch == "BiomedCLIP":
            self.prompt_learner = BiomedPromptLearner(self.classnames, self.model.cpu(),
                                                    context_size=context_size,  imgsize=imgsize, cls_token_position="end", tokenizer=tokenizer, per_class_ctx=True,
                                                    context_gain=context_gain).cuda()

        self.model.cuda()
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        if arch == "CLIP":
            self.text_encoder = TextEncoder(self.model).cuda()
        elif arch == "BiomedCLIP":
            self.text_encoder = BiomedTextEncoder(self.model).cuda()

    def predict_slide(self, image_features):
        # Input: image_features (N, D)
        image_features = F.normalize(image_features, dim=1)
        n, d = image_features.size()

        # Compute slide classifier weights
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        slide_weights = F.normalize(text_features, dim=1).T
        slide_logits = self.model.logit_scale.exp() * image_features @ slide_weights
        
        return slide_logits.mean(0)
    
    def train(self, train_loader, test_loader, n_epochs : int, lr : float, wd : float):
        # Build optimizer
        params = list(self.prompt_learner.parameters())
        optimizer = torch.optim.SGD(params, lr=lr, weight_decay=wd)
        # scheduler = build_lr_scheduler(optimizer=optimizer,
        #                                max_epoch=n_epochs,
        #                                warmup_epoch=1,
        #                                warmup_cons_lr=1e-5)

        # Train for n_epochs
        for e in range(n_epochs):
            self.metric_tracker.reset()
            pbar = tqdm(train_loader, dynamic_ncols=True)
            
            for wsi_feats, wsi_label, _ in pbar:
                wsi_feats = wsi_feats.cuda().squeeze()
                wsi_label = wsi_label.cuda().squeeze()

                wsi_pred = self.predict_slide(wsi_feats)
                loss = F.cross_entropy(wsi_pred, wsi_label)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                self.metric_tracker.update_metrics({
                    "loss": loss.item(),
                    "acc" : (wsi_pred.argmax() == wsi_label).float()*100.
                }, batch_size=1)
                pbar.set_description(self.metric_tracker.log(prefix=f"Epoch {e+1:d} - lr={optimizer.param_groups[0]['lr']:.7f}"))
                
            # scheduler.step()


class CITE(CLIP):
    """Baseline module based on clip
    """
    def __init__(self, model, tokenizer, templates : List[str], classnames : List[str], 
                 experiment_dir : str, context_size : int = 2) -> None:
        super().__init__(model, tokenizer, templates, classnames, experiment_dir)

        # Set attributes
        self.n_classes = len(self.classnames)
        self.experiment_dir = experiment_dir

        # Set prompt learner
        val = math.sqrt(6. / float(3 * reduce(mul, (16, 16), 1) + 768)) 
        self.prompt_embeddings = torch.nn.Parameter(torch.zeros(
            1, context_size, 768, device="cuda:0"), requires_grad=True)

        torch.nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        self.model.cuda()

    def patch_embeddings(self, x: torch.tensor):
        return self.model.visual.embeddings_patch(x)


    def embeddings_after_prompts(self, x: torch.tensor):
        return self.model.visual.forward_after_patch_embeddings(x)


    def incorporate_prompt(self, x):
        B = x.shape[0]
        x = self.patch_embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
        x = torch.cat((
            x[:, :1, :],
            self.prompt_embeddings.expand(B, -1, -1),
            x[:, 1:, :]
        ), dim=1)

        return x

    def encode_image(self, images):
        features = self.incorporate_prompt(images)
        features = self.embeddings_after_prompts(features)
        return F.normalize(features, dim=-1)


    def softmax_cross_entropy(self, p, labels):
        loss = -torch.log(p[torch.arange(p.size(0)), labels] + 1e-6)
        return loss.mean()


    def train(self, train_loader, test_loader, n_epochs : int):

        # Build optimizer
        n_iters = max(len(train_loader), 1000)
        optimizer = torch.optim.SGD([self.prompt_embeddings], lr=1e-2)
        scheduler = build_lr_scheduler(optimizer=optimizer, max_epoch=n_iters, warmup_epoch=10, warmup_cons_lr=1e-5)

        # Train for n_epochs
        i = 0
        with autocast():
            for e in range(n_epochs):
                self.model.train()
                self.metric_tracker.reset()

                pbar = tqdm(train_loader, dynamic_ncols=True)

                for img, label, _ in pbar:
                    optimizer.zero_grad()

                    img = img.cuda()
                    label = label.cuda()

                    patch_feats = self.encode_image(img) # (B, D)
                    logits = self.model.logit_scale.exp() * (patch_feats @ self.zero_shot_weights)

                    ce_loss = F.cross_entropy(logits, label)
                    ce_loss.backward()

                    optimizer.step()
                    scheduler.step()

                    self.metric_tracker.update_metrics({
                        "ce" : ce_loss.item(),
                        "acc" : 100 * (logits.argmax(dim=1) == label).float().mean()
                    })

                    pbar.set_description(self.metric_tracker.log(prefix=f"Epoch {e+1:d} - lr={optimizer.param_groups[0]['lr']:.7f}"))
                    i += 1
                    if i == n_iters:
                        break
                if i == n_iters:
                        break

    @torch.no_grad()
    def test(self, test_loader, save_model=False):
        # Prepare outout folder
        self.model.eval()
        self.metadata_tracker.reset()

        # Predict based on vision features
        all_labels = torch.tensor(test_loader.dataset.labels).cuda()
        all_wsi = np.array(test_loader.dataset.wsi_ids)
        all_vision_feats = []
        with autocast():
            for images, _, _ in tqdm(test_loader):
                images = images.cuda()
                feats = self.encode_image(images)
                all_vision_feats.append(feats)

            all_vision_feats = torch.cat(all_vision_feats, dim=0)

            # Compute slide level accuracy
            wsi_labels = []
            wsi_pred   = []

            unique_wsi = np.unique(all_wsi)
            for wsi_id in unique_wsi:
                wsi_labels.append(all_labels[all_wsi == wsi_id][0])
                wsi_feats = all_vision_feats[all_wsi == wsi_id]

                out = self.predict_slide(wsi_feats)

                if isinstance(out, tuple):
                    out = out[0]
                wsi_pred.append(out.argmax())

            wsi_labels = torch.stack(wsi_labels).cpu()
            wsi_pred = torch.stack(wsi_pred).cpu()
            self.metadata_tracker.update_metadata({
                "label" : wsi_labels,
                "pred" : wsi_pred,
                "wsi" : unique_wsi
            })
            self.metadata_tracker.to_csv(["label", "pred", "wsi"], self.experiment_dir, suffix="_last")

            # Compute results
            print(classification_results(wsi_pred, wsi_labels))

    def predict_slide(self, image_features):
        logits = self.model.logit_scale.exp() * (image_features @ self.zero_shot_weights)
        probs = F.softmax(logits, dim=1)
        return probs.mean(dim=0)


class PatchCoOp(CLIP):
    """Baseline module based on clip
    """
    def __init__(self, model, tokenizer, templates : List[str], slide_classnames : List[str], experiment_dir : str, context_size : int,
                 imgsize : int = 224, context_init=None, cls_token_position : str = "end", context_gain=1.0, arch="CLIP") -> None:
        super().__init__(model, tokenizer, templates, slide_classnames, experiment_dir)
        
        # Set attributes
        self.classnames = [[templates[0].format(x[0])] for x in slide_classnames]
        # self.tissue_classnames = tissue_classnames
        self.n_classes = len(self.classnames)
        self.experiment_dir = experiment_dir

        # Set prompt learner
        if arch == "CLIP":
            self.prompt_learner = PromptLearner(self.classnames, self.model.cpu(), context_size=context_size,
                            imgsize=imgsize, context_init=context_init, cls_token_position=cls_token_position, context_gain=context_gain).cuda()
        
        elif arch == "BiomedCLIP":
            self.prompt_learner = BiomedPromptLearner(self.classnames, self.model.cpu(),
                                                      context_size=context_size,  imgsize=imgsize, cls_token_position="end", tokenizer=tokenizer, per_class_ctx=True,
                                                      context_gain=context_gain).cuda()

        self.model.cuda()
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        if arch == "CLIP":
            self.text_encoder = TextEncoder(self.model).cuda()
        elif arch == "BiomedCLIP":
            self.text_encoder = BiomedTextEncoder(self.model).cuda()

    def predict_batch(self, image_features):
        # Input: image_features (N, D)
        image_features = F.normalize(image_features, dim=1)
        n, d = image_features.size()

        # Compute slide classifier weights
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        slide_weights = F.normalize(text_features, dim=1).T
        slide_logits = self.model.logit_scale.exp() * image_features @ slide_weights
        
        return slide_logits
    

    def train(self, train_loader, test_loader, n_epochs : int, lr : float, wd : float):
        # Build optimizer
        params = list(self.prompt_learner.parameters())
        optimizer = torch.optim.SGD(params, lr=lr, weight_decay=wd)
        n_iters = max(len(train_loader), 1000)
        scheduler = build_lr_scheduler(optimizer=optimizer,
                                       max_epoch=n_iters,
                                       warmup_epoch=10,
                                       warmup_cons_lr=1e-5)

        # Train for n_epochs
        i = 0
        for e in range(n_epochs):
            self.metric_tracker.reset()
            pbar = tqdm(train_loader, dynamic_ncols=True)
            
            for feats, label, _ in pbar:
                feats = feats.cuda().squeeze()
                label = label.cuda().squeeze()

                pred = self.predict_batch(feats)
                loss = F.cross_entropy(pred, label)

                # print(pred.shape, label.shape)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                self.metric_tracker.update_metrics({
                    "loss": loss.item(),
                    "acc" : (pred.argmax(dim=-1) == label).float().mean()*100.
                }, batch_size=1)
                pbar.set_description(self.metric_tracker.log(prefix=f"Epoch {e+1:d} - lr={optimizer.param_groups[0]['lr']:.7f}"))
                
                scheduler.step()
                i += 1
                if i == n_iters:
                    break
            if i == n_iters:
                    break

    # for testing only
    def predict_slide(self, image_features):
        # Input: image_features (N, D)
        image_features = image_features.squeeze()
        image_features = F.normalize(image_features, dim=1)
        n, d = image_features.size()

        # Compute slide classifier weights
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        slide_weights = F.normalize(text_features, dim=1).T
        slide_logits = self.model.logit_scale.exp() * image_features @ slide_weights
        predictions = torch.softmax(slide_logits, dim=-1)
        return predictions.mean(0, keepdim=True)