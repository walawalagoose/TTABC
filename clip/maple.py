import torch
import torch.nn as nn
from torch.nn import functional as F
import copy
import clip
from simple_tokenizer import SimpleTokenizer as _Tokenizer

from data.imagnet_prompts import imagenet_classes
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets

# import os
_tokenizer = _Tokenizer()

DOWNLOAD_ROOT='~/.cache/clip'


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx=16, ctx_init=None, prompt_depth=1):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.clip_model = clip_model
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.name_lens = None
        
        # Default is 1, which is compound shallow prompting
        assert prompt_depth >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = prompt_depth  # max=12, but will create 11 such shared prompts
        
        if ctx_init and (n_ctx <= 4):
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
            
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        if dtype == torch.float16:
            self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        
        # Define the compound prompts for the deeper layers
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
            
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)
        
        self.prompt_prefix = prompt_prefix
        self.reset_classnames(classnames)

    def reset_classnames(self, classnames):
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.ctx.device)  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).to(self.ctx.device)

        self.register_buffer("token_prefix", embedding[:, :1, :], persistent=False)  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:, :], persistent=False)  # CLS, EOS

        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.n_cls = len(classnames)

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # 为视觉侧准备深层提示
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
            
        # 返回文本提示和视觉提示
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts


class MaPLeCLIP(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx=2, ctx_init="a photo of a", prompt_depth=9):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(classnames, clip_model, n_ctx, ctx_init, prompt_depth)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        return {
            'image_features': image_features,
            'text_features': text_features,
            'logits': logits
        }


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MaPLe(nn.Module):
    def __init__(self,
                 classnames,
                 clip_model,
                 n_ctx=2,                  # prompt context length
                 ctx_init="a photo of a",             # initial context tokens
                 prompt_depth=9,            # depth of prompts
                 prec="fp32",               # precision: fp16, fp32, amp
                 device="cuda"):
        super().__init__()
        self.device = device
        self.prec = prec
        
        if prec == "fp32" or prec == "amp":
            clip_model.float()
            
        self.model = MaPLeCLIP(
            classnames=classnames,
            clip_model=clip_model,
            n_ctx=n_ctx,
            ctx_init=ctx_init,
            prompt_depth=prompt_depth
        )
        
        self.model.to(device)
        if device == "cuda" and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            
    def forward(self, image):
        if self.prec == "amp":
            with torch.cuda.amp.autocast():
                return self.model(image)
        return self.model(image)
    
    def reset_classnames(self, classnames):
        if isinstance(self.model, (MaPLeCLIP, nn.DataParallel)):
            module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            module.prompt_learner.reset_classnames(classnames)
            module.tokenized_prompts = module.prompt_learner.tokenized_prompts
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint["state_dict"]
        
        # 过滤固定的token
        if "prompt_learner.token_prefix" in state_dict:
            del state_dict["prompt_learner.token_prefix"]
        if "prompt_learner.token_suffix" in state_dict:
            del state_dict["prompt_learner.token_suffix"]
            
        self.model.load_state_dict(state_dict, strict=False)
        
        
def get_maple(clip_arch, test_set, device, n_ctx, ctx_init=None, prompt_depth=9):
    if test_set in fewshot_datasets:
        classnames = eval(f"{test_set.lower()}_classes")
    elif test_set == 'bongard':
        classnames = ['True', 'False']
    else:
        classnames = imagenet_classes
        
    clip_model, _, _ = clip.load(clip_arch, device=device, download_root=DOWNLOAD_ROOT, mode='maple')
    
    model = MaPLe(classnames, clip_model, n_ctx=n_ctx, ctx_init=ctx_init, prompt_depth=prompt_depth, device=device)

    return model