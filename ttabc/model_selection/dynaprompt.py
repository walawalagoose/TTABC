"""
    DynaPrompt: Dynamic Test-Time Prompt Tuning,
    https://arxiv.org/abs/2501.16404,
    https://github.com/zzzx1224/DynaPrompt
"""

import time

import numpy as np
import torch
import torch.nn as nn

from clip import load, tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_prompts import imagenet_classes
from ttabc.utils.tools import (
    Summary,
    AverageMeter,
    ProgressMeter,
    accuracy,
    build_optimizer,
)


_tokenizer = _Tokenizer()
DOWNLOAD_ROOT = "~/.cache/clip"


def _avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def _entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    return -(logits * logits.exp()).sum(dim=-1)


def _select_confident_samples_dynap(logits, top):
    batch_entropy = -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)
    if batch_entropy.dim() > 1:
        batch_entropy = batch_entropy.mean(-1)
    topk = int(max(logits.size(0) * top, 1))
    idx = torch.argsort(batch_entropy, descending=False)[:topk]
    return logits[idx], idx


class _TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        if prompts.dim() > 3:
            tokenized_prompts = tokenized_prompts.unsqueeze(0).repeat(prompts.size(0), 1, 1)
            tokenized_prompts = tokenized_prompts.view(-1, tokenized_prompts.size(-1))
        x = prompts.view(-1, prompts.size(-2), prompts.size(-1)) + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        return x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection


class _DynamicPromptLearner(nn.Module):
    def __init__(
        self,
        clip_model,
        classnames,
        n_ctx=4,
        ctx_init=None,
        ctx_position="end",
        learned_cls=False,
        num_prompts=1,
        num_select_prompts=1,
    ):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        self.dtype = clip_model.dtype
        self.device = clip_model.visual.conv1.weight.device
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.num_prompts = num_prompts
        self.num_select_prompts = num_select_prompts
        self.split_idx = None

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            if "[CLS]" in ctx_init:
                ctx_list = ctx_init.split(" ")
                self.split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = torch.zeros_like(embedding[0, 1 : 1 + n_ctx, :])
            self.init_ctx = embedding[0, 1 : 1 + n_ctx, :].clone()
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.init_ctx = torch.zeros_like(ctx_vectors)
            prompt_prefix = " ".join(["X"] * n_ctx)

        if num_prompts > 1:
            ctx_vectors = ctx_vectors.unsqueeze(0).repeat(num_prompts, 1, 1)

        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors)
        self.ctx_order = list(range(num_prompts))
        self.ctx_use = [0] * num_prompts

        self.prompt_prefix = prompt_prefix
        self.ctx_init = ctx_init
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx

        if not learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(n_cls, 1, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(cls_vectors, std=0.02)
            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors)
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " X." for _ in classnames]

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        if learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1 :, :])
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.classnames = classnames

    def reset(self):
        self.ctx.copy_(self.ctx_init_state)
        self.ctx_use = [0] * self.num_prompts
        self.ctx_order = list(range(self.num_prompts))
        if self.learned_cls:
            self.cls.copy_(self.cls_init_state)

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        classnames = [name.replace("_", " ") for name in classnames]
        self.name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        clip_model, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames
        self.ctx_use = [0] * self.num_prompts
        self.ctx_order = list(range(self.num_prompts))

    def forward(self):
        if self.num_prompts > 1:
            ctx = self.ctx + self.init_ctx.unsqueeze(0)
        else:
            ctx = self.ctx + self.init_ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif ctx.size(0) != self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.num_prompts > 1:
            prefix = prefix.unsqueeze(0).repeat(self.num_prompts, 1, 1, 1)
            suffix = suffix.unsqueeze(0).repeat(self.num_prompts, 1, 1, 1)

        if self.class_token_position != "end":
            raise NotImplementedError("DynaPrompt integration currently supports only end-position prompts.")

        if self.learned_cls:
            cls = self.cls
            return torch.cat([prefix, ctx, cls, suffix], dim=-2)
        return torch.cat([prefix, ctx, suffix], dim=-2)


class _DynamicPromptCLIP(nn.Module):
    def __init__(self, device, classnames, arch, n_ctx, ctx_init, num_prompts, num_select_prompts):
        super().__init__()
        clip_model, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.image_encoder = clip_model.visual
        self.text_encoder = _TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale.data
        self.prompt_learner = _DynamicPromptLearner(
            clip_model,
            classnames,
            n_ctx=n_ctx,
            ctx_init=ctx_init,
            num_prompts=num_prompts,
            num_select_prompts=num_select_prompts,
        )
        self.num_prompts = num_prompts

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    def get_text_features(self, prompt_selection=None):
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        if prompt_selection is not None:
            prompt_selection = prompt_selection.reshape(-1).to(dtype=torch.long, device=prompts.device)
            prompts = prompts[prompt_selection - 1]
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def forward(self, image, prompt_selection=None):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = self.get_text_features(prompt_selection)
        logit_scale = self.logit_scale.exp()
        return logit_scale * image_features @ text_features.t()


def _get_classnames(test_set):
    if test_set in fewshot_datasets:
        return eval(f"{test_set.lower()}_classes")
    return imagenet_classes


class DYNAPROMPT:
    def __init__(self, args):
        self.args = args
        assert self.args.prompt_type == "coop", "DynaPrompt is integrated as a CoOp-style method."
        device = torch.device(f"cuda:{self.args.gpu}" if torch.cuda.is_available() else "cpu")
        classnames = _get_classnames(args.test_sets)
        self.model = _DynamicPromptCLIP(
            device=device,
            classnames=classnames,
            arch=args.arch,
            n_ctx=args.n_ctx,
            ctx_init=args.ctx_init,
            num_prompts=args.dynap_num_prompts,
            num_select_prompts=args.dynap_num_select_prompts,
        )
        self.model = self.model.to(device)

        if args.load is not None:
            checkpoint = torch.load(args.load, map_location=device, weights_only=False)
            pretrained_ctx = checkpoint["state_dict"]["ctx"]
            assert pretrained_ctx.size(0) == args.n_ctx
            with torch.no_grad():
                self.model.prompt_learner.ctx[0].copy_(pretrained_ctx)
                self.model.prompt_learner.ctx_init_state[0].copy_(pretrained_ctx)

        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        self.optimizer = build_optimizer(args, self.model.prompt_learner.parameters())
        self.optim_state = self.optimizer.state_dict()
        self.scaler = torch.amp.GradScaler(init_scale=1000, device="cuda")

    def _forward_prompt_bank(self, images, prompt_ids, requires_grad):
        outputs = []
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        with context:
            with torch.amp.autocast(device_type="cuda"):
                for prompt_id in prompt_ids:
                    prompt_selection = torch.tensor([prompt_id + 1], device=images.device, dtype=torch.long)
                    prompt_output = self.model(images, prompt_selection)
                    outputs.append(prompt_output.view(images.size(0), 1, -1))
        return torch.cat(outputs, dim=1)

    def _dynamic_prompt_tuning(self, images):
        selected_prompt_ids = None
        raw_pred = None
        raw_entropy = None

        for _ in range(self.args.tta_steps):
            prompt_bank = self._forward_prompt_bank(
                images,
                prompt_ids=list(range(self.args.dynap_num_prompts)),
                requires_grad=False,
            ).detach()
            raw_pred = prompt_bank[0, 0]

            if self.args.dynap_onlinetpt and self.args.dynap_num_prompts > 1:
                prompt_entropy = _entropy(prompt_bank).mean(0)
                prompt_gap = prompt_bank[0].max(-1)[0].unsqueeze(0) - prompt_bank[1:].max(dim=-1)[0]
                prompt_gap = prompt_gap.mean(0)

                ent_order = prompt_entropy.topk(self.args.dynap_num_prompts)[1]
                init_prompt_position = self.model.prompt_learner.ctx_order[0]
                exact_idx = torch.where(ent_order == init_prompt_position)[0].item()
                gap_order = prompt_gap.topk(self.args.dynap_num_prompts)[1]
                ent_tail = ent_order[min(exact_idx + 1, self.args.dynap_num_prompts - 1) :]
                gap_head = gap_order[:exact_idx]
                overlap = np.intersect1d(ent_tail.cpu().numpy(), gap_head.cpu().numpy())

                if len(overlap) > 0:
                    selected_prompt_ids = torch.cat(
                        [ent_order[torch.as_tensor(overlap, device=ent_order.device)], ent_order[[exact_idx]]],
                        dim=0,
                    )
                else:
                    selected_prompt_ids = ent_order[exact_idx].reshape(1)

                chosen = np.array(selected_prompt_ids.detach().cpu()).reshape(-1).tolist()
                for prompt_id in chosen:
                    self.model.prompt_learner.ctx_order.remove(prompt_id)
                    self.model.prompt_learner.ctx_use[prompt_id] += 1
                    self.model.prompt_learner.ctx_order.append(prompt_id)
                raw_pred = prompt_bank[0, selected_prompt_ids].mean(0)
            else:
                selected_prompt_ids = torch.tensor([0], device=prompt_bank.device, dtype=torch.long)

            raw_entropy = _avg_entropy(raw_pred.unsqueeze(0))

            output = self._forward_prompt_bank(
                images,
                prompt_ids=selected_prompt_ids.reshape(-1).tolist(),
                requires_grad=True,
            )
            output, _ = _select_confident_samples_dynap(output, self.args.selection_p)
            with torch.amp.autocast(device_type="cuda"):
                loss = _avg_entropy(output).mean()

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return selected_prompt_ids, raw_pred, raw_entropy

    def test_time_tuning(self, inputs):
        return self._dynamic_prompt_tuning(inputs)

    def test_time_adapt_eval(self, val_loader, result_dict=None):
        batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
        top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
        top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
        progress = ProgressMeter(len(val_loader), [batch_time, top1, top5], prefix="DynaPrompt Test: ")

        self.model.eval()
        with torch.no_grad():
            self.model.reset()
        end = time.time()

        for i, (images, target) in enumerate(val_loader):
            assert self.args.gpu is not None
            if isinstance(images, list):
                for k in range(len(images)):
                    images[k] = images[k].cuda(self.args.gpu, non_blocking=True)
                image = images[0]
                images = torch.cat(images, dim=0)
            else:
                if len(images.size()) > 4:
                    assert images.size()[0] == 1
                    images = images.squeeze(0)
                images = images.cuda(self.args.gpu, non_blocking=True)
                image = images
            target = target.cuda(self.args.gpu, non_blocking=True)

            if self.args.tta_steps > 0:
                if not self.args.dynap_onlinetpt:
                    with torch.no_grad():
                        self.model.reset()
                elif self.args.dynap_num_prompts > 1 and self.model.prompt_learner.ctx_use.count(0) == 0:
                    with torch.no_grad():
                        prompt_id = self.model.prompt_learner.ctx_order[0]
                        self.model.prompt_learner.ctx_use[prompt_id] = 0
                        self.model.prompt_learner.ctx[prompt_id].zero_()

                self.optimizer.load_state_dict(self.optim_state)
                selected_prompt_ids, _, _ = self.test_time_tuning(images)
            else:
                selected_prompt_ids = torch.tensor([0], device=image.device, dtype=torch.long)

            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda"):
                    prompt_selection = selected_prompt_ids.reshape(-1) + 1
                    output = self._forward_prompt_bank(
                        image,
                        prompt_ids=selected_prompt_ids.reshape(-1).tolist(),
                        requires_grad=False,
                    ).mean(1)

            if result_dict is not None:
                probs = output.softmax(dim=1)
                max_confidence, max_index = torch.max(probs, 1)
                for j in range(max_confidence.size(0)):
                    result_dict["max_confidence"].append(max_confidence[j].item())
                    result_dict["prediction"].append(max_index[j].item())
                    result_dict["label"].append(target[j].item())

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % self.args.print_freq == 0:
                progress.display(i)

        progress.display_summary()
        return [top1.avg, top5.avg]