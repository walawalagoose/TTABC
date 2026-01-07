"""
    BATCLIP: Bimodal Online Test-Time Adaptation for CLIP,
    https://arxiv.org/abs/2412.02837,
    https://github.com/sarthaxxxxx/BATCLIP
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

from ttabc.utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, softmax_entropy, I2TLoss, InterMeanLoss
from ttabc.model_selection.base_method import BaseMethod

class BATCLIP(BaseMethod):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        
        # Setup loss functions
        self.i2t_loss = I2TLoss()
        self.inter_mean_loss = InterMeanLoss()
        # Configure model for BATCLIP adaptation
        self.configure_model()
        # Store initial model and optimizer states
        self.model_state_dict = deepcopy(self.model.state_dict())
        self.optim_state = deepcopy(self.optimizer.state_dict())
    
    def configure_model(self):
        """Same as Tent, only adapt normalization layers."""
        self.model.eval()
        self.model.requires_grad_(False)
        params_to_optimize = []
        
        # Enable normalization layers for adaptation
        for m in self.model.modules():
            if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
                m.train()
                for param in m.parameters():
                    param.requires_grad = True
                    params_to_optimize.append(param)
            if isinstance(m, nn.BatchNorm2d):
                m.train()
                for param in m.parameters():
                    param.requires_grad = True
                    params_to_optimize.append(param)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
        
        # Re-create optimizer with only normalization layer parameters
        assert len(params_to_optimize) > 0, "No normalization layer parameters found."
        self.optimizer = torch.optim.AdamW(params_to_optimize, self.args.lr)
        self.optim_state = deepcopy(self.optimizer.state_dict())
        
    def reset(self):
        """Reset model to initial state before adaptation."""
        self.model.load_state_dict(self.model_state_dict, strict=False)
        self.optimizer.load_state_dict(self.optim_state)
    
    def test_time_tuning(self, images):
        for _ in range(self.args.tta_steps):
            with torch.amp.autocast(device_type='cuda'):
                logits, _, text_feat, img_pre_feats, _ = self.model(images, return_features=True)
                
                ent_loss = softmax_entropy(logits).mean(0)
                i2t_loss = -self.i2t_loss(logits, img_pre_feats, text_feat)
                inter_mean_loss = -self.inter_mean_loss(logits, img_pre_feats)
                loss = ent_loss + i2t_loss + inter_mean_loss
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
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
                    # images[k] = resize_with_CLIP(images[k], self.args.resolution).cuda(self.args.gpu, non_blocking=True)
                    images[k] = images[k].cuda(self.args.gpu, non_blocking=True)
                image = images[0]
                images = torch.cat(images, dim=0)
            else: # batch-wise
                # images = resize_with_CLIP(images, self.args.resolution).cuda(self.args.gpu, non_blocking=True)
                images = images.cuda(self.args.gpu, non_blocking=True)
                image = images
            target = target.cuda(self.args.gpu, non_blocking=True)

            # Reset the norm layers to its initial state
            if self.args.tta_steps > 0 and self.args.episodic:
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