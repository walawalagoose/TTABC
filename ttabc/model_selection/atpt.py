'''
    A-TPT: Angular Diversity Calibration Properties for Test-Time Prompt Tuning of Vision-Language Models (ICLR 2026)
    https://arxiv.org/abs/2510.26441
'''

import time
from PIL import Image

import torch
import torch.optim

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from ttabc.utils.tools import (
    Summary, AverageMeter, ProgressMeter,
    accuracy, select_confident_samples, marginal_entropy
)
from ttabc.model_selection.base_method import BaseMethod


class ATPT(BaseMethod):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.temperature_value = {'ViT': 1.16, 'RN': 1.15}

    def test_time_tuning(self, inputs):
        output = None
        output2 = None
        single_output = None

        if self.args.prompt_type == 'cocoop':
            image_feature, pgen_ctx = inputs
            pgen_ctx.requires_grad = True
            self.optimizer = torch.optim.AdamW([pgen_ctx], self.args.lr)

        selected_idx = None

        for j in range(self.args.tta_steps):
            with torch.amp.autocast(device_type='cuda'):
                if self.args.prompt_type == 'maple':
                    output = self.model(inputs)['logits']
                elif self.args.prompt_type == 'cocoop':
                    output = self.model((image_feature, pgen_ctx))
                else:
                    output = self.model(inputs)

                if selected_idx is not None:
                    output = output[selected_idx]
                else:
                    output, selected_idx = select_confident_samples(
                        output, self.args.selection_p
                    )

                loss = marginal_entropy(output)

            if self.args.two_step:
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward(retain_graph=True)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                loss = 0

                with torch.amp.autocast(device_type='cuda'):
                    if self.args.prompt_type == 'cocoop':
                        output2 = self.model((image_feature, pgen_ctx))
                    else:
                        output2 = self.model(inputs)

            if output is None and output2 is None:
                single_output = self.model(self.args.image)

            # =====================================================
            # [A-TPT] Angular Diversity Regularization
            # =====================================================

            # text features: [N, D]
            text_features = self.model.get_text_features()

            # normalize (can stay in AMP)
            text_features = torch.nn.functional.normalize(
                text_features, dim=-1
            )

            # >>> A-TPT FIX <<<
            # acos MUST be computed in FP32 for numerical stability
            with torch.cuda.amp.autocast(enabled=False):
                text_features_fp32 = text_features.float()

                # cosine similarity [N, N]
                cos_sim = text_features_fp32 @ text_features_fp32.t()

                eps = 1e-7
                cos_sim = cos_sim.clamp(-1 + eps, 1 - eps)

                # angular distance
                angles = torch.acos(cos_sim)

                # mask diagonal
                N = angles.size(0)
                diag_mask = torch.eye(
                    N, device=angles.device, dtype=torch.bool
                )
                angles = angles.masked_fill(diag_mask, float('inf'))

                # minimum angular distance per class
                min_angles, _ = angles.min(dim=1)

                # angular diversity
                angular_diversity = min_angles.mean()
            # <<< A-TPT FIX <<<

            lambda_ = self.args.lambda_term
            loss = loss + (-lambda_ * angular_diversity)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        if self.args.prompt_type == 'cocoop':
            return pgen_ctx

        return None

    def test_time_adapt_eval(self, val_loader, result_dict):
        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, top1, top5],
            prefix='Test: '
        )

        self.model.eval()
        if self.args.prompt_type != 'cocoop':
            with torch.no_grad():
                self.model.reset()

        end = time.time()
        softmax = torch.nn.Softmax(dim=1)

        for i, (images, target) in enumerate(val_loader):
            assert self.args.gpu is not None

            if isinstance(images, list):
                for k in range(len(images)):
                    images[k] = images[k].cuda(self.args.gpu, non_blocking=True)
                image = images[0]
            else:
                if len(images.size()) > 4:
                    images = images.squeeze(0)
                images = images.cuda(self.args.gpu, non_blocking=True)
                image = images

            target = target.cuda(self.args.gpu, non_blocking=True)

            if self.args.tpt:
                images = torch.cat(images, dim=0)

            self.args.image = image

            if self.args.prompt_type != 'cocoop':
                if self.args.tta_steps > 0:
                    with torch.no_grad():
                        self.model.reset()
                self.optimizer.load_state_dict(self.optim_state)
                self.test_time_tuning(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type='cuda'):
                        image_feature, pgen_ctx = self.model.gen_ctx(
                            images, self.args.tpt
                        )
                self.optimizer = None
                pgen_ctx = self.test_time_tuning((image_feature, pgen_ctx))

            if self.args.tpt and self.args.prompt_type == 'cocoop':
                image_feature = image_feature[0].unsqueeze(0)

            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda'):
                    if self.args.prompt_type == 'maple':
                        output = self.model(image)['logits']
                    elif self.args.prompt_type == 'cocoop':
                        output = self.model((image_feature, pgen_ctx))
                    else:
                        output = self.model(image)

            softmax_output = softmax(output)
            max_confidence, max_index = torch.max(softmax_output, 1)

            result_dict['max_confidence'].append(max_confidence.item())
            result_dict['prediction'].append(max_index.item())
            result_dict['label'].append(target.item())

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % self.args.print_freq == 0:
                progress.display(i)

        progress.display_summary()
        return [top1.avg, top5.avg]