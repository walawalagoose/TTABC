"""
    CALIP: Zero-Shot Enhancement of CLIP with Parameter-free Attention,
    https://arxiv.org/abs/2209.14169,
    https://github.com/ZiyuGuo99/CALIP
"""

import time
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import tokenize
from ttabc.model_selection.base_method import BaseMethod
from ttabc.utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, resize_with_CLIP
from data.prompt_utils import build_text_features, resolve_templates, build_basic_prompts


@torch.no_grad()
def _build_text_features(args, classnames: List[str], clip_model: nn.Module) -> torch.Tensor:
    device = next(clip_model.parameters()).device

    if args.prompt_setting is not None:
        _, text_features = build_text_features(
            clip_model, classnames, args.prompt_setting, dataset=args.test_sets, device=device
        )
        return text_features.t()

    templates = resolve_templates(args.test_sets) if args.test_sets else ["a photo of a {}."]
    text_features = []
    for name in classnames:
        prompts = build_basic_prompts(name.replace("_", " "), templates=templates)
        tokens = torch.cat([tokenize(p) for p in prompts]).to(device)
        feats = clip_model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feat = feats.mean(dim=0)
        feat = feat / feat.norm()
        text_features.append(feat)

    text_features = torch.stack(text_features, dim=0)
    return text_features.t()


def _encode_image_tokens(clip_model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    visual = clip_model.visual
    if not hasattr(visual, "class_embedding"):
        raise NotImplementedError("CALIP currently supports only ViT backbones with patch tokens.")

    x = visual.conv1(image.type(clip_model.dtype))
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    cls_token = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls_token, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)

    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)

    x = visual.ln_post(x)
    if visual.proj is not None:
        x = x @ visual.proj
    return x


class CALIPWrapper(nn.Module):
    def __init__(self, clip_model: nn.Module, args):
        super().__init__()
        self.clip_model = clip_model
        self.args = args

        self.beta2 = float(getattr(args, "calip_beta2", 2.0))
        self.beta3 = float(getattr(args, "calip_beta3", 0.1))
        self.logit_scale = clip_model.logit_scale.data.exp()

        self.register_buffer("_feat_t", torch.empty(0), persistent=False)

    def reset(self):
        return

    def reset_classnames(self, classnames, arch):
        feat_t = _build_text_features(self.args, classnames, self.clip_model)
        device = next(self.clip_model.parameters()).device
        feat_t = feat_t.to(device)
        self._feat_t = feat_t

    @torch.no_grad()
    def forward(self, image):
        tokens = _encode_image_tokens(self.clip_model, image)
        tokens = tokens / tokens.norm(dim=-1, keepdim=True)

        img_global = tokens[:, 0, :]  # (N, D)
        img_spatial = tokens[:, 1:, :].permute(0, 2, 1)  # (N, D, P)

        feat_t = self._feat_t  # (D, C)

        base_logits = self.logit_scale * (img_global @ feat_t)  # (N, C)

        a_weight = torch.matmul(img_spatial.permute(0, 2, 1), feat_t) * 2.0
        a_weight1 = F.softmax(a_weight, dim=1)
        a_weight2 = F.softmax(a_weight, dim=2)

        feat_t_a = torch.matmul(img_spatial, a_weight1)

        feat_t_t = feat_t.t().unsqueeze(0).expand(a_weight2.size(0), -1, -1)  # (N, C, D)
        feat_v_a_tokens = torch.matmul(a_weight2, feat_t_t)
        feat_v_a = feat_v_a_tokens.mean(dim=1) + feat_v_a_tokens.max(dim=1)[0]  # (N, D)

        logits1 = self.logit_scale * torch.einsum("nd,ndc->nc", img_global, feat_t_a)
        logits2 = self.logit_scale * (feat_v_a @ feat_t)

        return base_logits + logits1 * self.beta2 + logits2 * self.beta3


class CALIP(BaseMethod):
    def __init__(self, args):
        super().__init__(args)
        
        self.args = args
        device = torch.device(f"cuda:{self.args.gpu}" if torch.cuda.is_available() else "cpu")

        # Replace TTABC model with CALIP inference wrapper.
        self.model = CALIPWrapper(self.model.clip_model, args).to(device).eval()

    def test_time_tuning(self, inputs):
        return

    def test_time_adapt_eval(self, val_loader, result_dict=None):
        batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
        top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
        top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)

        progress = ProgressMeter(len(val_loader), [batch_time, top1, top5], prefix="CALIP Test: ")

        self.model.eval()
        end = time.time()
        softmax = torch.nn.Softmax(dim=1)

        with torch.no_grad():
            for i, (images, target) in enumerate(val_loader):
                if isinstance(images, list):
                    for k in range(len(images)):
                        images[k] = resize_with_CLIP(images[k], self.args.resolution).cuda(self.args.gpu, non_blocking=True)
                    image = images[0]
                else:
                    if len(images.size()) > 4:
                        assert images.size()[0] == 1
                        images = images.squeeze(0)
                    image = resize_with_CLIP(images, self.args.resolution).cuda(self.args.gpu, non_blocking=True)

                target = target.cuda(self.args.gpu, non_blocking=True)

                with torch.amp.autocast(device_type="cuda"):
                    output = self.model(image)

                if result_dict is not None:
                    probs = softmax(output)
                    max_confidence, max_index = torch.max(probs, 1)
                    if max_confidence.numel() == 1:
                        result_dict['max_confidence'].append(max_confidence.item())
                        result_dict['prediction'].append(max_index.item())
                        result_dict['label'].append(target.item())
                    else:
                        for j in range(max_confidence.size(0)):
                            result_dict['max_confidence'].append(max_confidence[j].item())
                            result_dict['prediction'].append(max_index[j].item())
                            result_dict['label'].append(target[j].item())

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], image.size(0))
                top5.update(acc5[0], image.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                if (i + 1) % self.args.print_freq == 0:
                    progress.display(i)

        progress.display_summary()
        return [top1.avg, top5.avg]