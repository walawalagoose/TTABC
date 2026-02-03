"""
    PromptAlign: Test-Time Prompting with Distribution Alignment for Zero-Shot Generalization,
    https://arxiv.org/abs/2311.01459,
    https://github.com/jameelhassan/PromptAlign
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from ttabc.model_selection.base_method import BaseMethod
from ttabc.utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, marginal_entropy


class PromptAlign(BaseMethod):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.visual_means, self.visual_vars = self.load_visual_stats()

    def _get_backbone_core(self):
        return self.model.model if hasattr(self.model, "model") else self.model

    def _forward_logits(self, inputs):
        if self.args.prompt_type == "maple":
            return self.model(inputs)["logits"]
        return self.model(inputs)
    
    def load_visual_stats(self):
        means_path = self.args.vis_means_path
        vars_path = self.args.vis_vars_path
        if means_path and vars_path and os.path.isfile(means_path) and os.path.isfile(vars_path):
            return (torch.load(means_path, map_location="cpu"),
                torch.load(vars_path, map_location="cpu"),)
        return None, None
    
    def select_confident_samples_pa(self, logits):
        batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        idx_tpt = torch.argsort(batch_entropy, descending=False)[: int(batch_entropy.size(0) * self.args.tpt_threshold)]
        idx_align = torch.argsort(batch_entropy, descending=False)[: int(batch_entropy.size(0) * self.args.align_threshold)]
        return logits[idx_tpt], idx_align

    def distr_align_loss(self, out_feat, targ_feat):
        layers_from = self.args.align_layer_from
        layers_to = self.args.align_layer_to
        out_means, out_vars = out_feat
        targ_means, targ_vars = targ_feat
        targ_means, targ_vars = targ_means.to(out_means.device), targ_vars.to(out_vars.device)

        distr_loss = 0
        for l in range(layers_from, layers_to - 1):
            distr_loss += 0.5 * F.l1_loss(out_means[l], targ_means[l])
            distr_loss += 0.5 * F.l1_loss(out_vars[l], targ_vars[l])
        return distr_loss

    def compute_visual_stats(self, selected_idx):
        backbone = self._get_backbone_core()
        resblocks = backbone.image_encoder.transformer.resblocks
        out_visual_mean = torch.cat([
            torch.mean(res.visual_feat[:, selected_idx, :], dim=1, keepdim=True).permute(1, 0, 2)
            for res in resblocks
        ])
        out_visual_var = torch.cat([
            torch.mean(
                (res.visual_feat[:, selected_idx, :]
                 - out_visual_mean[i, :, :].unsqueeze(0).permute(1, 0, 2)) ** 2,
                dim=1,
                keepdim=True,
            ).permute(1, 0, 2)
            for i, res in enumerate(resblocks)
        ])
        return out_visual_mean, out_visual_var

    def test_time_tuning(self, inputs):
        selected_idx = None

        for _ in range(self.args.tta_steps):
            with autocast(device_type="cuda"):
                output = self._forward_logits(inputs)

                if selected_idx is not None:
                    output_selected = output[selected_idx]
                else:
                    output_selected, selected_idx = self.select_confident_samples_pa(output)
                
                loss = None
                if self.args.tpt_loss:
                    loss = marginal_entropy(output_selected)

                if self.args.distr_align:
                    out_feat = self.compute_visual_stats(selected_idx)
                    distr_loss = self.distr_align_loss(out_feat, (self.visual_means, self.visual_vars))
                    distr_loss_w = self.args.distr_loss_w / (self.args.align_layer_to - self.args.align_layer_from)
                    if loss:
                        loss += distr_loss_w * distr_loss
                    else:
                        loss = distr_loss_w * distr_loss

            if loss is None:
                continue

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

    def test_time_adapt_eval(self, val_loader, result_dict=None):
        batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
        top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
        top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, top1, top5],
            prefix="Test: ")

        self.model.eval()
        with torch.no_grad():
            self.model.reset()

        end = time.time()

        for i, (images, target) in enumerate(val_loader):
            assert self.args.gpu is not None
            if isinstance(images, list):
                images = [img.cuda(self.args.gpu, non_blocking=True) for img in images]
                image = images[0]
            else:
                if len(images.size()) > 4:
                    assert images.size()[0] == 1
                    images = images.squeeze(0)
                images = images.cuda(self.args.gpu, non_blocking=True)
                image = images

            target = target.cuda(self.args.gpu, non_blocking=True)

            if self.args.tpt:
                images_concat = torch.cat(images, dim=0)
            else:
                images_concat = images

            if self.args.tta_steps > 0:
                if self.args.episodic:
                    with torch.no_grad():
                        self.model.reset()
                self.optimizer.load_state_dict(self.optim_state)
                self.test_time_tuning(images_concat)

            with torch.no_grad():
                with autocast(device_type="cuda"):
                    output = self._forward_logits(image)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            if result_dict is not None:
                softmax_output = output.softmax(1)
                max_confidence, max_index = torch.max(softmax_output, 1)
                
                if max_confidence.numel() == 1:
                    result_dict["max_confidence"].append(max_confidence.item())
                    result_dict["prediction"].append(max_index.item())
                    result_dict["label"].append(target.item())
                else:
                    for j in range(max_confidence.size(0)):
                        result_dict["max_confidence"].append(max_confidence[j].item())
                        result_dict["prediction"].append(max_index[j].item())
                        result_dict["label"].append(target[j].item())

            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % self.args.print_freq == 0:
                progress.display(i)

        progress.display_summary()
        return top1.avg, top5.avg


PROMPTALIGN = PromptAlign