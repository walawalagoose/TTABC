"""
    Online Zero-Shot Classification with CLIP (OnZeta)
    https://arxiv.org/abs/2407.10946
"""

import math

import torch
import torch.nn.functional as F

from ttabc.model_selection.base_method import BaseMethod
from ttabc.utils.tools import accuracy


class OnZeta(BaseMethod):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        assert self.args.prompt_type == "no_prompt", "OnZeta only supports prompt_type='no_prompt'"

    def test_time_tuning(self, inputs):
        pass

    def test_time_adapt_eval(self, val_loader, result_dict=None):
        self.model.eval()

        image_feat = []
        image_label = []

        with torch.no_grad():
            for images, target in val_loader:
                assert self.args.gpu is not None
                if isinstance(images, list):
                    images = images[0]
                if len(images.size()) > 4:
                    assert images.size()[0] == 1
                    images = images.squeeze(0)
                images = images.cuda(self.args.gpu, non_blocking=True)
                target = target.cuda(self.args.gpu, non_blocking=True)

                _, features = self.model(images)
                image_feat.append(features.float())
                image_label.append(target)

        image_feat = torch.cat(image_feat, dim=0)
        image_label = torch.cat(image_label, dim=0)
        n = len(image_label)
        num_class = len(self.model.classnames)

        with torch.no_grad():
            text_classifier = self.model.get_text_features().float().t()
            logits_t = image_feat @ text_classifier

        acc1, acc5 = accuracy(logits_t, image_label, topk=(1, 5))

        onzeta_sum = torch.zeros(n, num_class, device=image_feat.device, dtype=image_feat.dtype)
        onlab_sum = torch.zeros_like(onzeta_sum)

        for _ in range(self.args.repeat):
            idx = torch.randperm(n, device=image_feat.device)
            combo_label = torch.zeros(n, num_class, device=image_feat.device, dtype=image_feat.dtype)
            text_label = torch.zeros_like(combo_label)
            w = text_classifier.clone()
            rho = torch.zeros(num_class, device=image_feat.device, dtype=image_feat.dtype)

            for i in range(n):
                lr = self.args.cw / math.sqrt(i + 1)
                rlr = self.args.cr / math.sqrt(i + 1)
                beta = self.args.beta * math.sqrt((i + 1) / n)
                x = image_feat[idx[i], :]

                tlabel = F.softmax((x @ text_classifier) / self.args.tau_t, dim=0)
                tlabel = tlabel * torch.exp(rho)
                tlabel = tlabel / torch.sum(tlabel)
                rho = rho - rlr * (tlabel - self.args.alpha / num_class)
                rho = rho.clamp_min(0.0)

                text_label[i, :] = tlabel

                vision_label = F.softmax((x @ w) / self.args.tau_i, dim=0)
                combo_label[i, :] = beta * vision_label + (1.0 - beta) * tlabel

                grad = torch.outer(x, vision_label - tlabel)
                w = w - (lr / self.args.tau_i) * grad
                w = F.normalize(w, dim=0)

            inv_idx = torch.empty_like(idx)
            inv_idx[idx] = torch.arange(n, device=idx.device)
            onzeta_sum += combo_label[inv_idx]
            onlab_sum += text_label[inv_idx]

        onzeta_output = onzeta_sum / self.args.repeat
        _ = onlab_sum / self.args.repeat

        onzeta_acc1, onzeta_acc5 = accuracy(onzeta_output, image_label, topk=(1, 5))

        if result_dict is not None:
            # softmax_output = onzeta_output.softmax(dim=1)
            # max_confidence, max_index = torch.max(softmax_output, 1)
            softmax_output = onzeta_output.softmax(dim=1)
            max_confidence, max_index = torch.max(onzeta_output, 1)
            for j in range(max_confidence.size(0)):
                result_dict["max_confidence"].append(max_confidence[j].item())
                result_dict["prediction"].append(max_index[j].item())
                result_dict["label"].append(image_label[j].item())

        return [onzeta_acc1[0], onzeta_acc5[0]]