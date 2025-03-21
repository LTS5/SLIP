
import torch
import clip
import open_clip
from .ctranspath import ctranspath
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer

def get_clip_network(model_name : str):

    if model_name == "CLIP":
        model, embed_dim, preprocess = clip.load("ViT-B/16")
        preprocess_train = preprocess_test = preprocess
        tokenizer = clip.tokenize

    elif model_name == "CLIP-RN50":
        model, embed_dim, preprocess = clip.load("RN50")
        preprocess_train = preprocess_test = preprocess
        tokenizer = clip.tokenize

    elif model_name == "PLIP":
        model = CLIPModel.from_pretrained("vinid/plip")
        preprocess = CLIPProcessor.from_pretrained("vinid/plip")
        preprocess_train = preprocess_test = lambda x : preprocess.image_processor(x, return_tensors="pt")["pixel_values"][0]
        tokenizer = lambda x : preprocess.tokenizer(x, return_tensors="pt", padding=True)["input_ids"]
        embed_dim = 768
        model = model.cuda()

    elif model_name == "BiomedCLIP":
        model, preprocess_train, preprocess_test = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', precision='fp32')
        tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        # print(model.state_dict().keys())
        # print(model)
        embed_dim = 512 #model.state_dict()["text_projection"].shape[1]
        model = model.cuda()
    else:
        raise NotImplementedError
    
    return model, embed_dim, tokenizer, preprocess_train, preprocess_test

def get_ctranspath(save_path):
    return ctranspath(save_path)
