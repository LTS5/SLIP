import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask

_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, context_size : int = 16, imgsize : int = 224, 
                 context_init : str = "a photo of a", cls_token_position : str = "end", context_gain=1.0):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = context_size
        ctx_init = context_init
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = imgsize
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02 * context_gain)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name[0].replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name for name in classnames]
        print(prompts)

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cls_token_position

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model, context_size : int = 16, imgsize : int = 224, 
                 context_init : str = "a photo of a", cls_token_position : str = "end"):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner = PromptLearner(classnames, clip_model, context_size=context_size,
                                imgsize=imgsize, context_init=context_init, cls_token_position=cls_token_position)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, return_feats : bool = False):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        if return_feats:
            return logits, image_features
        return logits
    


class BiomedTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.config = clip_model.text.config
        self.transformer = clip_model.text.transformer
        self.pooler = clip_model.text.pooler
        self.proj = clip_model.text.proj

    def forward(self, prompts, tokenized_prompts):
        attn_mask = (tokenized_prompts != self.config.pad_token_id).long().cuda()
        out = self.transformer(inputs_embeds=prompts, attention_mask=attn_mask)
        pooled_out = self.pooler(out, attn_mask)
        x = self.proj(pooled_out)

        return x


class PLIPTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        # print(clip_model.model.text_model)
        self.config = clip_model.model.text_model.config
        self.transformer = clip_model.model.text_model.encoder
        self.text_projection = clip_model.model.text_projection
        self.final_layer_norm = clip_model.model.text_model.final_layer_norm
        self.eos_token_id = 46810
       
    def forward(self, prompts, tokenized_prompts):

        input_shape = prompts.size()[:-1]
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        hidden_states = prompts
        # print(tokenized_prompts)
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )
        attention_mask = (tokenized_prompts != 49407).long().cuda()
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.transformer(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        if self.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                tokenized_prompts.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                (tokenized_prompts.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]


        text_features = self.text_projection(pooled_output)
        return text_features

class PLIPModel(nn.Module):
    def __init__(self, model, dtype) -> None:
        super().__init__()
        self.model = model
        self.dtype = dtype
        self.logit_scale = model.logit_scale

    def encode_text(self, input_ids):
        text_outputs = self.model.text_model(
            input_ids=input_ids,
            attention_mask=None,
            position_ids=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )

        pooled_output = text_outputs[1]
        text_features = self.model.text_projection(pooled_output)

        return text_features

class BiomedPromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, context_size : int = 16, imgsize : int = 224, cls_token_position : str = "end", tokenizer=_tokenizer, per_class_ctx=False, context_gain=1.0):
        super().__init__()
        classnames = [j for i in classnames for j in i ]
        n_cls = len(classnames)
        n_ctx = context_size
        ctx_dim = 768
        clip_imsize = 224
        cfg_imsize = imgsize
        dtype = torch.float32
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        print("Initializing a generic context")
        if per_class_ctx:
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02*context_gain)
        prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name for name in classnames]

        tokenized_prompts = torch.cat([tokenizer(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.text.transformer.embeddings.word_embeddings(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cls_token_position

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class PLIPPromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, context_size : int = 16, imgsize : int = 224, cls_token_position : str = "end", tokenizer=_tokenizer, per_class_ctx=False, context_gain=1.0):
        super().__init__()
        classnames = [j for i in classnames for j in i ]
        n_cls = len(classnames)
        n_ctx = context_size
        ctx_dim = 512
        clip_imsize = 224
        cfg_imsize = imgsize
        dtype = clip_model.dtype
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        print("Initializing a generic context")
        if per_class_ctx:
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02*context_gain)
        prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name for name in classnames]

        # tokenized_prompts = torch.cat([tokenizer(p) for p in prompts])
        # print(prompts)
        tokenized_prompts = tokenizer(prompts)
        # print(tokenized_prompts)
        with torch.no_grad():
            embedding = clip_model.model.text_model.embeddings(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cls_token_position

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class VisionAdapter(torch.nn.Module):
    def __init__(self, c_in, reduction=4):
        super(VisionAdapter, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(c_in, c_in // reduction, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(c_in // reduction, c_in, bias=False),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class CustomBiomedCLIP(nn.Module):
    def __init__(self, classnames, clip_model, context_size : int = 16, imgsize : int = 224, cls_token_position : str = "end", tokenizer=_tokenizer, per_class_ctx=False, use_adapter=False):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner = BiomedPromptLearner(classnames, clip_model, context_size=context_size, imgsize=imgsize, cls_token_position=cls_token_position, tokenizer=tokenizer, per_class_ctx=per_class_ctx)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = BiomedTextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.tokenizer = tokenizer
        self.classnames = classnames

        # vision adaptor
        self.vision_adapter = VisionAdapter(clip_model.text.output_dim)
        self.image_encoder = clip_model.visual
        self.use_adapter = use_adapter


    def encode_image(self, image):
        image_features = self.image_encoder(image.to(torch.float))
        return F.normalize(image_features, dim=-1)
    
    def encode_image_via_adapter(self, image):
        image_features = self.image_encoder(image.to(torch.float))
        residual = self.vision_adapter(image_features)
        image_features = image_features + residual
        return F.normalize(image_features, dim=-1)
    
    def get_text_prototypes(self,):
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        return F.normalize(text_features, dim=-1)


    def forward(self, image):
        if self.use_adapter:
            image_features = self.encode_image_via_adapter(image.to(torch.float))
        else:
            image_features = self.encode_image(image.to(torch.float))

        text_features = self.get_text_prototypes()
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits
    
class VisionPrototypes(torch.nn.Module):
    def __init__(self, k, d):
        super().__init__()
        self.vision_prototypes = torch.nn.Parameter(torch.randn(k, d), requires_grad=True)

    def forward(self,):
        normalized_prototypes = F.normalize(self.vision_prototypes, dim=1)
        return normalized_prototypes