import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import sys
import os
import numpy as np
sys.path.append(os.path.abspath('/home/hfd24/lzm_self/TTABC'))
# Use the copied PromptAlign CLIP package
from ttabc.pa_clip import clip
from ttabc.pa_clip import tokenize
from ttabc.pa_clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

# Base method
from .base_method import BaseMethod
from ttabc.utils.tools import accuracy

_tokenizer = _Tokenizer()

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def load_clip_to_cpu(args):
    backbone_name = args.arch
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    
    # Adapt design details from PromptAlign config structure to TTABC args
    design_details = {
        "trainer": 'PromptAlign',
        "vision_depth": 0,
        "language_depth": 0, 
        "vision_ctx": 0,
        "language_ctx": 0,
        "maple_length": args.n_ctx # cfg.TRAINER.PROMPTALIGN.N_CTX
    }
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

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
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        self.learned_cls = False
        n_cls = len(classnames)
        n_ctx = args.n_ctx
        ctx_init = args.ctx_init
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = args.resolution # Assuming args.resolution exists
        
        # Default is 1, which is compound shallow prompting
        assert args.prompt_depth >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = args.prompt_depth

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
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
            
        print('PromptAlign design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        
        if dtype == torch.float16:
            self.proj.half()
            
        self.ctx = nn.Parameter(ctx_vectors)
        self.proj_weight_init_state = self.proj.weight.detach().clone()
        self.proj_bias_init_state = self.proj.bias.detach().clone()
        self.ctx_init_state = ctx_vectors.detach().clone()

        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Copy init state
        self.compound_prompts_text_init_state = [txt_prompt.detach().clone() for txt_prompt in self.compound_prompts_text]

        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)
        self.compound_prompt_projections_init_state = [(module.weight.detach().clone(), module.bias.detach().clone()) for module in self.compound_prompt_projections]

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
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

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts
    
    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors) 

        with torch.no_grad():
            self.proj.weight.copy_(self.proj_weight_init_state)
            self.proj.bias.copy_(self.proj_bias_init_state)

            for idx, prompt in enumerate(self.compound_prompts_text):
                prompt.copy_(self.compound_prompts_text_init_state[idx])
            
            for idx, module in enumerate(self.compound_prompt_projections):
                module.weight.copy_(self.compound_prompt_projections_init_state[idx][0])
                module.bias.copy_(self.compound_prompt_projections_init_state[idx][1])

    def reset_classnames(self, classnames, args):
        self.device = self.ctx.device
        self.n_cls = len(classnames)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        
        # Re-construct prompts
        # Assuming prompt_prefix is available or reconstructable. 
        # But PromptAlign uses internal buffer/state implicitly.
        # Wait, PromptAlign's reset_classnames regenerates prompts.
        # But it needs prompt_prefix. 
        # In original code, prompt_prefix was a local var in __init__, not saved in self.
        # But reset_classnames in original code uses self.prompt_prefix ? 
        # Ah, original code line 260: prompts = [self.prompt_prefix + ...].
        # But lines 134-148 define `prompt_prefix` BUT don't save it to `self`.
        # Wait, check PromptAlign code again. 
        # Line 197-260 in original file... 
        # Oh, the original code had a bug if `self.prompt_prefix` wasn't saved? 
        # Or maybe it relies on `ctx_init`.
        # Let's check original code line 188 -> wait, I don't see `self.prompt_prefix = ...`.
        # Ah, actually I might have missed it or it relies on `ctx` being reset properly.
        # But reset_classnames constructs new strings.
        # Let's assume for now we don't need reset_classnames heavily if classes don't change dynamically per sample (only per dataset, which reloads model).
        # Actually TTABC calls reset_classnames in main loop.
        
        # Workaround: Re-derive prompt_prefix from args
        n_ctx = args.n_ctx
        ctx_init = args.ctx_init
        if ctx_init and (n_ctx) <= 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt_prefix = ctx_init
        else:
             prompt_prefix = " ".join(["X"] * n_ctx)
             
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        clip_model = load_clip_to_cpu(args).to(self.device)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.ctx.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]
        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts

    def set_prompt_init_states(self):
        ctx_vectors = self.ctx.detach().clone()
        self.ctx_init_state = ctx_vectors
        self.proj_weight_init_state = self.proj.weight.detach().clone()
        self.proj_bias_init_state = self.proj.bias.detach().clone()

        self.compound_prompts_text_init_state = [txt_prompt.detach().clone() for txt_prompt in self.compound_prompts_text]
        self.compound_prompt_projections_init_state = [(module.weight.detach().clone(), module.bias.detach().clone()) for module in self.compound_prompt_projections]


class CustomCLIP(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        self.args = args
        self.prompt_learner = MultiModalPromptLearner(args, classnames, clip_model)
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

        if self.prompt_learner.training and label is not None:
             return F.cross_entropy(logits, label)
               
        return logits
    
    def get_text_features(self):
        with torch.no_grad():
            tokenized_prompts = self.tokenized_prompts
            prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
            text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames):
        self.prompt_learner.reset_classnames(classnames, self.args)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

    def set_prompt_inits(self):
        print("Re-updating prompt initializations to current prompts.")
        self.prompt_learner.set_prompt_init_states()


class PromptAlign(BaseMethod):
    def __init__(self, args):
        self.args = args
        # ========= 从命令行恢复 --load，避免被 config_hparams 覆盖 =========
        import sys
        if getattr(self.args, 'load', None) in [None, ""]:
            if '--load' in sys.argv:
                idx = sys.argv.index('--load')
                if idx + 1 < len(sys.argv):
                    self.args.load = sys.argv[idx + 1]
        print('PromptAlign using args.load =', getattr(self.args, 'load', None))
        # ===============================================================

        self.batch_size = args.batch_size
        self.temperature_value = {'ViT': 1.16, 'RN': 1.15}
        
        # Create model using internal functionality to match PromptAlign requirements
        # We don't use BaseMethod's create_model because it doesn't support the custom CLIP needed here
        print(f"Creating PromptAlign model with backbone: {args.arch}")
        
        # Using the same logic as Create_model for determining classes (simplified)
        # Assuming args.test_sets is single string in this context or handled by main loop
        # TTABC main loop handles classnames and passes them.
        # But BaseMethod.__init__ calls create_model which needs detailed args.
        # Here we skip BaseMethod.__init__ logic for model creation.
        
        # Load CLIP to CPU
        clip_model = load_clip_to_cpu(args)
        
        # We need classnames to initialize MultiModalPromptLearner
        # We'll use a placeholder and rely on reset_classnames later, or load standard ImageNet classes
        # TTABC usually initializes with ImageNet classes or target dataset classes
        from data.imagnet_prompts import imagenet_classes
        self.classnames = imagenet_classes
        
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            
        self.model = CustomCLIP(args, self.classnames, clip_model)
        
        # Load pretrained MaPLe weights if provided
        if hasattr(args, 'load') and args.load is not None:
            print(f"Loading pretrained MaPLe weights from: {args.load}")
            checkpoint = torch.load(args.load, map_location="cpu")
            
            # Handle different checkpoint formats
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            # Load weights
            msg = self.model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded pretrained weights. Missing keys: {msg.missing_keys}")
            print(f"Unexpected keys: {msg.unexpected_keys}")
            
            # Update init states after loading pretrained weights
            self.model.set_prompt_inits()
        else:
            print("WARNING: No pretrained MaPLe weights loaded!")
            print("PromptAlign requires pretrained MaPLe weights to achieve reported performance.")
            print("Current setup will use random initialization, which significantly degrades performance.")
        
        self.model.cuda(args.gpu)
        
        # Setup optimizer
        trainable_param = self.model.prompt_learner.parameters()
        self.optimizer = torch.optim.AdamW(trainable_param, lr=args.lr) # args.lr corresponds to TPT.LR
        self.optim_state = copy.deepcopy(self.optimizer.state_dict())
        
        self.scaler = GradScaler(init_scale=1000)
        
        # Load stats for distribution alignment
        # Need to know where VIS_VARS and VIS_MEANS are.
        # PromptAlign/trainers/prompt_align.py uses self.cfg.TPT.VIS_VARS
        # We need to find these files or arguments.
        # Assuming user provides paths or we default to something.
        # BUT PromptAlign relies on pre-computed stats!
        # If user didn't provide them, this method fails.
        # Let's check args structure again. 
        # We might need to add arguments for these paths or assume they are in 'stats/' folder.
        
        # Note: In PromptAlign code, get_tpt_dataloader loads these.
        # We need to load them here.
        # Assuming args has these attributes or we construct paths?
        # Let's try to locate them or warn.
        # For now, placeholder or check if args has them.
        
        self.visual_means = getattr(args, 'vis_means', None)
        self.visual_vars = getattr(args, 'vis_vars', None)
        
        if self.visual_means is None or self.visual_vars is None:
            # 假设文件在: /home/hfd24/lzm_self/TTABC/outputs/features/ImgNet_vis_means.pt
            default_means = "/home/hfd24/lzm_self/TTABC/outputs/features/ImgNet_vis_means.pt"
            default_vars  = "/home/hfd24/lzm_self/TTABC/outputs/features/ImgNet_vis_vars.pt"

            if os.path.isfile(default_means) and os.path.isfile(default_vars):
                # 把路径写回 args，后面 test_time_tuning 里的 lazy-load 会用到
                self.args.vis_means_path = default_means
                self.args.vis_vars_path  = default_vars
            else:
                # 找不到就保持 None，后面会自动关闭 distr_align
                self.args.vis_means_path = None
                self.args.vis_vars_path  = None

    def test_time_tuning(self, inputs):
        # This matches PromptAlign's logic in test_time_tuning
        # inputs: (images, target) or just images depending on how it's called
        # BaseMethod calls it with `inputs`.
        
        # In PromptAlign: test_time_tuning(model, inputs, optimizer, scaler, args)
        
        # Logic adaptation:
        
        image_feature = None # Not using CoCoOp 
        pgen_ctx = None
        
        selected_idx = None
        
        # Check if we need to load stats (lazy loading or if provided)
        if hasattr(self.args, 'vis_means_path') and self.visual_means is None:
             self.visual_means = torch.load(self.args.vis_means_path)
             self.visual_vars = torch.load(self.args.vis_vars_path)
        
        # If still None, we can't do distribution alignment properly -> disable it
        if self.visual_means is None and self.args.distr_align:
             print("Warning: Visual means/vars not loaded. Disabling distribution alignment.")
             self.args.distr_align = False
        
        for j in range(self.args.tta_steps):
            with autocast():
                output = self.model(inputs)

                if selected_idx is not None:
                    output = output[selected_idx]
                else:
                    output, selected_idx = self.select_confident_samples(output, self.args.tpt_threshold, self.args.align_threshold)

                # Initialize loss properly
                loss = None
                if self.args.tpt_loss:
                    loss = self.avg_entropy(output)

                # Distribution alignment
                if self.args.distr_align and self.visual_means is not None and len(selected_idx) > 0:
                    target_feat_distr = (self.visual_means, self.visual_vars)
                    
                    transformer_blocks = self.model.image_encoder.transformer.resblocks
                    
                    # Compute feature statistics from selected samples
                    out_visual_mean = torch.cat([
                        torch.mean(res.visual_feat[:, selected_idx, :], dim=1, keepdim=True).permute(1,0,2) 
                        for res in transformer_blocks
                    ])
                    
                    out_visual_var = torch.cat([
                        torch.mean(((res.visual_feat[:, selected_idx, :] - out_visual_mean[i, :, :].unsqueeze(0).permute(1,0,2))**2), dim=1, keepdim=True).permute(1,0,2) 
                        for i, res in enumerate(transformer_blocks)
                    ])
                    
                    out_feat_distr = (out_visual_mean, out_visual_var)

                    DISTR_LOSS_W = self.args.distr_loss_w / (self.args.align_layer_to - self.args.align_layer_from)
                    
                    distr_loss = DISTR_LOSS_W * self.distr_align_loss(
                         out_feat_distr, target_feat_distr, 
                         layers_from=self.args.align_layer_from, 
                         layers_to=self.args.align_layer_to
                    )
                    
                    if loss is None:
                        loss = distr_loss
                    else:
                        loss += distr_loss
                
                # Skip optimization if no loss computed
                if loss is None:
                    continue
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

    def select_confident_samples(self, logits, topTPT, topAlign):
        batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        idxTPT = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * topTPT)]
        idxAlign = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * topAlign)]
        return logits[idxTPT], idxAlign

    def avg_entropy(self, outputs):
        if outputs.shape[0] == 0:
            return outputs.sum() * 0
        logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) 
        avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) 
        min_real = torch.finfo(avg_logits.dtype).min
        avg_logits = torch.clamp(avg_logits, min=min_real)
        return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

    def distr_align_loss(self, out_feat, targ_feat, layers_from=0, layers_to=12, moments=5):
        '''
        A feature distribution alignment L1 loss between mean and variance of the features
        '''
        distr_loss = 0
        out_means, out_vars = out_feat
        targ_means, targ_vars = targ_feat
        
        # Ensure devices match
        targ_means = targ_means.to(out_means.device)
        targ_vars = targ_vars.to(out_vars.device)
        
        transf_layers = layers_to
        for l in range(layers_from, transf_layers - 1):  # Note: transf_layers-1 as in original
            out_mean, out_var = out_means[l], out_vars[l]
            targ_mean, targ_var = targ_means[l], targ_vars[l]
            distr_loss += 0.5 * F.l1_loss(out_mean, targ_mean) + 0.5 * F.l1_loss(out_var, targ_var)
        
        return distr_loss

    def test_time_adapt_eval(self, val_loader, result_dict=None):
        # Adapted from PromptAlign test_time_adapt_eval
        # TTABC main loop calls this, but BaseMethod expects slightly different signature?
        # Actually in TTABC main.py: tta_methods.test_time_adapt_eval(val_loader, results_for_ece[set_id])
        
        self.model.eval()
        # Reset prompt to initial state at start of evaluation if using TTA
        # Assuming we reset per batch in TPT?
        # PromptAlign resets model per batch if args.TTA_STEPS > 0
        
        # Note: CustomCLIP has reset()
        
        # Need to re-init optimizer?
        # PromptAlign re-inits optimizer state dict per batch
        
        # Main evaluation loop
        top1_acc = []
        top5_acc = []
        
        for i, (images, target) in enumerate(val_loader):
            # Setup device - DON'T flatten the list before checking!
            if isinstance(images, list):
                 images = [img.cuda(self.args.gpu, non_blocking=True) for img in images]
                 image_test = images[0]
                 if self.args.tpt:
                     images_concat = torch.cat(images, dim=0)
                 else:
                     images_concat = images[0]
            else:
                 images = images.cuda(self.args.gpu, non_blocking=True)
                 image_test = images
                 images_concat = images
                 
            target = target.cuda(self.args.gpu, non_blocking=True)
            
            # Reset model and optimizer
            if self.args.tta_steps > 0:
                with torch.no_grad():
                    self.model.reset()
                self.optimizer.load_state_dict(self.optim_state)
                
                # Adapt
                self.test_time_tuning(images_concat)
                
            # Inference
            with torch.no_grad():
                with autocast():
                    output = self.model(image_test)
            
            # Metrics
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1_acc.append(acc1.item())
            top5_acc.append(acc5.item())
            
            if result_dict is not None:
                # Store results logic similar to batclip
                softmax_output = output.softmax(1)
                max_confidence, max_index = torch.max(softmax_output, 1)
                
                if max_confidence.numel() == 1:
                    result_dict['max_confidence'].append(max_confidence.item())
                    result_dict['prediction'].append(max_index.item())
                    result_dict['label'].append(target.item())
                else:
                    for k in range(max_confidence.size(0)):
                         result_dict['max_confidence'].append(max_confidence[k].item())
                         result_dict['prediction'].append(max_index[k].item())
                         result_dict['label'].append(target[k].item())
                         
        return np.mean(top1_acc), np.mean(top5_acc)

    def reset_classnames(self, classnames, args):
        # TTABC main calls reset_classnames(args, tta_methods, classnames)
        # We need to expose this method or implementing it as part of model
        self.model.reset_classnames(classnames)


# Export PromptAlign as PROMPTALIGN
PROMPTALIGN = PromptAlign
