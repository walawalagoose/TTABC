"""
    Entropy is not Enough for Test-Time Adaptation: From the Perspective of Disentangled Factors,
    https://arxiv.org/abs/2403.07366,
    https://whitesnowdrop.github.io/DeYO
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
import torchvision
import math
from einops import rearrange

from ttabc.utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, build_optimizer, softmax_entropy
from ttabc.model_selection.base_method import BaseMethod

class DeYO(BaseMethod):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        
        # Configure model for DeYO adaptation
        self.configure_model()
        # Store initial model and optimizer states
        self.model_state_dict = deepcopy(self.model.state_dict())
        self.optim_state = deepcopy(self.optimizer.state_dict())
        
        # DeYO hyperparameters
        self.reweight_ent = self.args.reweight_ent
        self.reweight_plpd = self.args.reweight_plpd
        self.plpd_threshold = self.args.plpd_threshold
        self.deyo_margin = self.args.deyo_margin # remember to multiply math.log(num_classes)
        self.margin_e0 = self.args.margin # remember to multiply math.log(num_classes)

        self.aug_type = self.args.aug_type
        self.occlusion_size = self.args.occlusion_size
        self.row_start = self.args.row_start
        self.column_start = self.args.column_start
        self.patch_len = self.args.patch_len
    
    def configure_model(self):
        """Only adapt normalization layers for image encoder."""
        self.model.eval()
        self.model.requires_grad_(False)

        # Enable normalization layers for adaptation
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
        
        # Re-create optimizer with only normalization layer parameters
        params_to_optimize, _ = self.configure_trainable_parameters()
        assert len(params_to_optimize) > 0, "No normalization layer parameters found."
        self.optimizer = build_optimizer(self.args, params_to_optimize)
        self.optim_state = deepcopy(self.optimizer.state_dict())
        
    def configure_trainable_parameters(self):
        params = []
        names = []
        for nm, m in self.model.named_modules():
            # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
            if 'layer4' in nm:
                continue
            if 'blocks.9' in nm:
                continue
            if 'blocks.10' in nm:
                continue
            if 'blocks.11' in nm:
                continue
            if 'norm.' in nm:
                continue
            if nm in ['norm']:
                continue

            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

        return params, names
        
    def reset(self):
        """Reset model to initial state before adaptation."""
        self.model.load_state_dict(self.model_state_dict, strict=False)
        self.optimizer.load_state_dict(self.optim_state)
        
    def loss_calculation(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        imgs_test = x[0]
        outputs = self.model(imgs_test, return_features=False)
        K = outputs.shape[1]
        margin_e0 = self.margin_e0 * math.log(K)
        deyo_margin = self.deyo_margin * math.log(K)

        entropies = softmax_entropy(outputs)
        filter_ids_1 = torch.where((entropies < deyo_margin))
        entropies = entropies[filter_ids_1]
        if len(entropies) == 0:
            loss = None  # set loss to None, since all instances have been filtered
            return outputs, loss

        x_prime = imgs_test[filter_ids_1]
        x_prime = x_prime.detach()
        if self.aug_type == 'occ':
            first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
            final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
            occlusion_window = final_mean.expand(-1, -1, self.occlusion_size, self.occlusion_size)
            x_prime[:, :, self.row_start:self.row_start + self.occlusion_size, self.column_start:self.column_start + self.occlusion_size] = occlusion_window
        elif self.aug_type == 'patch':
            resize_t = torchvision.transforms.Resize(((imgs_test.shape[-1] // self.patch_len) * self.patch_len, (imgs_test.shape[-1] // self.patch_len) * self.patch_len))
            resize_o = torchvision.transforms.Resize((imgs_test.shape[-1], imgs_test.shape[-1]))
            x_prime = resize_t(x_prime)
            x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=self.patch_len, ps2=self.patch_len)
            perm_idx = torch.argsort(torch.rand(x_prime.shape[0], x_prime.shape[1]), dim=-1)
            x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1), perm_idx]
            x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=self.patch_len, ps2=self.patch_len)
            x_prime = resize_o(x_prime)
        elif self.aug_type == 'pixel':
            x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
            x_prime = x_prime[:, :, torch.randperm(x_prime.shape[-1])]
            x_prime = rearrange(x_prime, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=imgs_test.shape[-1], ps2=imgs_test.shape[-1])

        with torch.no_grad():
            outputs_prime = self.model(x_prime)

        prob_outputs = outputs[filter_ids_1].softmax(1)
        prob_outputs_prime = outputs_prime.softmax(1)

        cls1 = prob_outputs.argmax(dim=1)

        plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1, 1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1, 1))
        plpd = plpd.reshape(-1)

        filter_ids_2 = torch.where(plpd > self.plpd_threshold)
        entropies = entropies[filter_ids_2]
        if len(entropies) == 0:
            loss = None  # set loss to None, since all instances have been filtered
            return outputs, loss

        plpd = plpd[filter_ids_2]

        if self.reweight_ent or self.reweight_plpd:
            coeff = (float(self.reweight_ent) * (1. / (torch.exp(((entropies.clone().detach()) - margin_e0)))) +
                     float(self.reweight_plpd) * (1. / (torch.exp(-1. * plpd.clone().detach())))
                     )
            entropies = entropies.mul(coeff)

        loss = entropies.mean(0)
        return outputs, loss
    
    @ torch.enable_grad()
    def test_time_tuning(self, images):
        for _ in range(self.args.tta_steps):
            with torch.amp.autocast(device_type='cuda'):
                _, loss = self.loss_calculation((images,))
            # update model only if not all instances have been filtered
            if loss is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            self.optimizer.zero_grad()
        return
    
    def test_time_adapt_eval(self, val_loader, result_dict=None):
        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, top1, top5],
            prefix='Test: ')
        
        self.reset()
        end = time.time()
    
        for i, (images, target) in enumerate(val_loader):
            assert self.args.gpu is not None
            if isinstance(images, list): # sample-wise
                for k in range(len(images)):
                    images[k] = images[k].cuda(self.args.gpu, non_blocking=True)
                image = images[0]
                images = torch.cat(images, dim=0)
            else: # batch-wise
                images = images.cuda(self.args.gpu, non_blocking=True)
                image = images
            target = target.cuda(self.args.gpu, non_blocking=True)

            # Reset the norm layers to its initial state
            if self.args.tta_steps > 0:
                if self.args.episodic:
                    # Episodic reset (default disabled)
                    with torch.no_grad():
                        self.reset()
                # Perform test-time tuning
                self.test_time_tuning(images)

            # The actual inference goes here
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda'):
                    output = self.model(image)

            if result_dict is not None:
                softmax_output = output.softmax(dim=1)
                max_confidence, max_index = torch.max(softmax_output, 1)
                if max_confidence.numel() == 1:
                    result_dict['max_confidence'].append(max_confidence.item())
                    result_dict['prediction'].append(max_index.item())
                    result_dict['label'].append(target.item())
                else:
                    for j in range(max_confidence.size(0)):
                        result_dict['max_confidence'].append(max_confidence[j].item())
                        result_dict['prediction'].append(max_index[j].item())
                        result_dict['label'].append(target[j].item())


            # Measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % self.args.print_freq == 0:
                progress.display(i)

        progress.display_summary()

        return [top1.avg, top5.avg]