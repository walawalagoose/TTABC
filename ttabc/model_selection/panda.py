"""
    Panda: Test-Time Adaptation with Negative Data Augmentation (AAAI 2026)
    https://arxiv.org/abs/2511.10481
    https://github.com/ruxideng/Panda
"""

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

from ttabc.utils.tools import (
    Summary, AverageMeter, ProgressMeter,
    accuracy, build_optimizer, softmax_entropy
)
from ttabc.model_selection.base_method import BaseMethod


# ------------------------------------------------------------
# Official Panda negative augmentation (batch_hedge_v6)
# ------------------------------------------------------------
def batch_hedge_v6_images(batch, patch_size=32):
    B, C, H, W = batch.shape
    num_ph, num_pw = H // patch_size, W // patch_size
    if num_ph == 0 or num_pw == 0:
        return batch

    eff_H, eff_W = num_ph * patch_size, num_pw * patch_size

    patches = (
        batch[:, :, :eff_H, :eff_W]
        .unfold(2, patch_size, patch_size)
        .unfold(3, patch_size, patch_size)
    )
    patches = (
        patches.permute(0, 2, 3, 1, 4, 5)
        .reshape(B, num_ph * num_pw, C, patch_size, patch_size)
    )

    M = math.ceil(B / 10)
    out = torch.zeros((M, C, H, W), device=batch.device, dtype=batch.dtype)

    for m in range(M):
        base_idx = torch.randint(0, B, (1,), device=batch.device).item()

        idx_img = torch.randint(0, B, (num_ph * num_pw,), device=batch.device)
        mask = idx_img == base_idx
        while mask.any():
            idx_img[mask] = torch.randint(
                0, B, (mask.sum().item(),), device=batch.device
            )
            mask = idx_img == base_idx

        patch_idx = torch.arange(num_ph * num_pw, device=batch.device)
        new_patches = patches[idx_img, patch_idx]

        out_region = (
            new_patches
            .reshape(num_ph, num_pw, C, patch_size, patch_size)
            .permute(2, 0, 3, 1, 4)
            .reshape(C, eff_H, eff_W)
        )

        out_batch = batch[base_idx].clone()
        out_batch[:, :eff_H, :eff_W] = out_region
        out[m] = out_batch

    return out


# ------------------------------------------------------------
# Panda = Tent + logit-level NDA offset (official)
# ------------------------------------------------------------
class Panda(BaseMethod):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        # Panda hyperparameters (official-aligned)
        self.beta = getattr(args, "panda_beta", 0.2)
        self.patch_size = getattr(args, "panda_patch_size", 32)

        # Configure model for Tent adaptation
        self.configure_model()

        # Store initial model and optimizer states
        self.model_state_dict = deepcopy(self.model.state_dict())
        self.optim_state = deepcopy(self.optimizer.state_dict())

    # --------------------------------------------------------
    # Tent configuration (unchanged)
    # --------------------------------------------------------
    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)

        for m in self.model.image_encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)

        params_to_optimize, _ = self.configure_trainable_parameters()
        assert len(params_to_optimize) > 0
        self.optimizer = build_optimizer(self.args, params_to_optimize)
        self.optim_state = deepcopy(self.optimizer.state_dict())

    def configure_trainable_parameters(self):
        params = []
        names = []
        for nm, m in self.model.image_encoder.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ["weight", "bias"]:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def reset(self):
        self.model.load_state_dict(self.model_state_dict, strict=False)
        self.optimizer.load_state_dict(self.optim_state)

    # --------------------------------------------------------
    # Panda logit construction (OFFICIAL semantics)
    # --------------------------------------------------------
    def panda_logits(self, images):
        # Original logits
        orig_logits = self.model(images, return_features=False)

        # Negative augmentations
        bh6 = batch_hedge_v6_images(images, patch_size=self.patch_size)
        bh6_logits = self.model(bh6, return_features=False)

        # Mean bias logit (batch-shared)
        mean_bh6_logits = bh6_logits.mean(dim=0, keepdim=True)
        mean_bh6_logits = mean_bh6_logits.expand(orig_logits.size(0), -1)

        # Logit-level offset (official)
        return orig_logits - self.beta * mean_bh6_logits

    # --------------------------------------------------------
    # Test-time tuning (Tent + Panda)
    # --------------------------------------------------------
    def test_time_tuning(self, images):
        for _ in range(self.args.tta_steps):
            with torch.amp.autocast(device_type="cuda"):
                logits = self.panda_logits(images)
                loss = softmax_entropy(logits).mean(0)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        return

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    def test_time_adapt_eval(self, val_loader, result_dict=None):
        batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
        top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
        top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, top1, top5],
            prefix="Test: "
        )

        self.reset()
        end = time.time()

        for i, (images, target) in enumerate(val_loader):
            assert self.args.gpu is not None

            if isinstance(images, list):
                for k in range(len(images)):
                    images[k] = images[k].cuda(self.args.gpu, non_blocking=True)
                image = images[0]
                images = torch.cat(images, dim=0)
            else:
                images = images.cuda(self.args.gpu, non_blocking=True)
                image = images

            target = target.cuda(self.args.gpu, non_blocking=True)

            if self.args.tta_steps > 0:
                if self.args.episodic:
                    with torch.no_grad():
                        self.reset()
                self.test_time_tuning(images)

            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda"):
                    output = self.panda_logits(image)

            if result_dict is not None:
                softmax_output = output.softmax(dim=1)
                max_confidence, max_index = torch.max(softmax_output, 1)
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