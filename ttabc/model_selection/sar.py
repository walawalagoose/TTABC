"""
    Towards Stable Test-Time Adaptation in Dynamic Wild World,
    https://arxiv.org/abs/2302.12400,
    https://github.com/mr-eggplant/SAR
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
import math

from ttabc.utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, build_optimizer, softmax_entropy
from ttabc.model_selection.base_method import BaseMethod

class SAR(BaseMethod):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        
        # Configure model for SAR adaptation
        self.configure_model()
        # Store initial model and optimizer states
        self.model_state_dict = deepcopy(self.model.state_dict())
        self.optim_state = deepcopy(self.optimizer.state_dict())
        
        # SAR hyperparameters
        self.margin_e0 = self.args.sar_margin_e0  # margin E_0 for reliable entropy minimization, Eqn. (2)
        self.reset_constant_em = self.args.reset_constant_em  # threshold e_m for model recovery scheme
        self.ema = None  # to record the moving average of model output entropy, as model recovery criteria
    
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
        
        # Set up optimizer
        self.optimizer = self.set_optimizer(params_to_optimize)
        self.optim_state = deepcopy(self.optimizer.state_dict())
        
    def set_optimizer(self, params):
        arch_name = self.args.arch.lower().replace("-", "_")
        if "vit_" in arch_name or "swin_" in arch_name:
            if abs(self.args.lr - 0.001) > 1e-6:
                print("Recommend transformer settings where lr=0.001.")
        return SAM(params, torch.optim.SGD, lr=self.args.lr, momentum=0.9)
        
    def configure_trainable_parameters(self):
        """Collect the affine scale + shift parameters from norm layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
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
        
    @torch.no_grad()
    def update_ema(self, new_data, alpha=0.9):
        if self.ema is None:
            return new_data
        else:
            return alpha * self.ema + (1 - alpha) * new_data
        
    def reset(self):
        """Reset model to initial state before adaptation."""
        self.model.load_state_dict(self.model_state_dict, strict=False)
        self.optimizer.load_state_dict(self.optim_state)
        self.ema = None
    
    def test_time_tuning(self, images):
        for _ in range(self.args.tta_steps):
            # First forward-backward step
            with torch.amp.autocast(device_type='cuda'):
                logits1 = self.model(images, return_features=False)
            K = logits1.size(1)
            margin_e0 = self.margin_e0 * math.log(K)
            entropies1 = softmax_entropy(logits1)
            filter_ids_1 = torch.where(entropies1 < margin_e0)
            loss1 = entropies1[filter_ids_1].mean(0)
            # No scaler, manual step for SAM
            loss1.backward()
            self.optimizer.first_step(
                zero_grad=True
            )  # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
            
            # Second forward-backward step
            with torch.amp.autocast(device_type='cuda'):
                logits2 = self.model(images, return_features=False)
            entropies2 = softmax_entropy(logits2)
            entropies2 = entropies2[filter_ids_1]  # second time forward
            filter_ids_2 = torch.where(
                entropies2 < margin_e0
            )  # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
            loss2 = entropies2[filter_ids_2].mean(0)
            if not loss2.isnan().item():
                self.ema = self.update_ema(loss2.item())  # record moving average loss values for model recovery
            # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
            loss2.backward()
            self.optimizer.second_step(zero_grad=True)
            
            # perform model recovery
            if self.ema is not None:
                if self.ema < self.reset_constant_em:
                    self.reset()
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
    
class SAM(torch.optim.Optimizer):
    # from https://github.com/davda54/sam
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups