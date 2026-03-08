'''
    A-TPT: Angular Diversity Calibration Properties for Test-Time Prompt Tuning of Vision-Language Models (ICLR 2026)
    https://arxiv.org/abs/2510.26441,
    https://github.com/MB-Shihab-Aaqil-Ahamed/A-TPT/tree/master
'''

import time
from PIL import Image

import torch
import torch.optim
import torch.nn.functional as F

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
            # W = text_features; normalize; compute cosine sim matrix; for each class take max
            # (excluding diagonal via subtracting 2*diag); clamp; min_ang_norm = -acos(clamped);
            # loss += (-lambda_) * mean(min_ang_norm)
            W = self.model.get_text_features()
            W_ = F.normalize(W, p=2, dim=1)  # [C, D]
            Wwt_ = torch.matmul(W_, W_.t())  # [C, C]

            # remove self-similarities from being selected by max
            Wwt_ = Wwt_ - 2.0 * torch.diag(torch.diag(Wwt_))

            max_Wwt_ = Wwt_.max(dim=1)[0]  # [C]
            tau_ = getattr(self.args, "tau_term", 0.99999)
            Wwt_constraint = max_Wwt_.clamp(-tau_, tau_)
            min_ang_norm = -torch.acos(Wwt_constraint)  # [C]
            min_ang_norm_mean = min_ang_norm.mean()

            lambda_ = getattr(self.args, "lambda_term", 0.0)
            loss = loss + ((-lambda_) * min_ang_norm_mean)
            # <<< A-TPT FIX <<<
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