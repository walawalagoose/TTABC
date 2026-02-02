"""
    Dual Prototype Evolving for Test-Time Generalization of Vision-Language Models,
    https://arxiv.org/abs/2410.12790,
    https://zhangce01.github.io/DPE-CLIP/
"""

import time
import operator
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip.prototype import wrap_dpe_backbone
from data.prompt_utils import get_classnames
from ttabc.utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, resize_with_CLIP, marginal_entropy, softmax_entropy, InfoNCELoss, select_confident_samples
from ttabc.model_selection.base_method import BaseMethod



class TextResidue(nn.Module):
    """Learnable residual for text prototypes"""
    def __init__(self, clip_weights):
        super(TextResidue, self).__init__()
        self.feat_dim, self.cate_num = clip_weights.shape
        self.residual = nn.Parameter(
            torch.zeros([self.feat_dim, self.cate_num]).half().cuda(), 
            requires_grad=True
        )
        
    def forward(self, x):
        new_clip_weights = x.clone() + self.residual
        new_clip_weights = F.normalize(new_clip_weights, dim=0)
        return new_clip_weights
    
    def reset(self):
        self.residual = nn.Parameter(
            torch.zeros([self.feat_dim, self.cate_num]).half().cuda(), 
            requires_grad=True
        )


class PositiveCacheResidue(nn.Module):
    """Learnable residual for image prototypes (positive cache)"""
    def __init__(self, pos_cache_keys):
        super(PositiveCacheResidue, self).__init__()
        self.feat_dim, self.cache_size = pos_cache_keys.shape
        self.residual = nn.Parameter(
            torch.zeros([self.feat_dim, self.cache_size]).half().cuda(), 
            requires_grad=True
        )
        
    def forward(self, x):
        new_pos_cache_keys = x.clone() + self.residual
        new_pos_cache_keys = F.normalize(new_pos_cache_keys, dim=0)
        return new_pos_cache_keys


def get_entropy_normalized(loss, n_classes):
    """get entropy normalized by number of classes"""
    if isinstance(loss, torch.Tensor):
        loss = loss.item()
    return loss / torch.log(torch.tensor(n_classes, dtype=torch.float)).item()


class DPE_CLIP(BaseMethod):
    """
    DPE-CLIP: Dual Prototype Evolving for Test-Time Generalization
    
    Key features:
    1. Text Prototype Evolution: Learns residuals for CLIP text weights
    2. Image Prototype Evolution: Maintains and optimizes positive cache
    3. Dual Prototype Alignment: Aligns image and text prototypes via InfoNCE
    4. Global Prototype Update: Accumulates high-confidence updates
    """
    
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        self.classnames = get_classnames(args.test_sets)
        self.backbone = self.model
        self.model = wrap_dpe_backbone(self.backbone)

        self.pos_config = {
            'enabled': True,
            'shot_capacity': 3,
            'alpha': 6.0,
            'beta': 5.0,
        }
        if getattr(args, 'pos_config', None):
            self.pos_config.update(args.pos_config)
        self.lr_config = {
            'text': 0.0006,
            'image': 0.0006,
            'align': 0.5,
        }
        if getattr(args, 'lr_config', None):
            self.lr_config.update(args.lr_config)
        self.global_update_threshold = getattr(args, 'global_update_threshold', 0.1)

        self.pos_cache = {}
        self.clip_weights_global = None
        self.num_avg = 0
        
    def reset_cache(self):
        """Reset the positive cache (called at the start of each dataset)."""
        self.pos_cache = {}
        self.clip_weights_global = None
        self.num_avg = 0

    def reset_classnames(self, classnames, arch=None):
        self.classnames = classnames
        if hasattr(self.model, "reset_classnames"):
            self.model.reset_classnames(classnames, arch)
        self.reset_cache()

    def update_cache(self, cache, pred, features_loss, shot_capacity):
        """Update cache with new features and loss, maintaining the maximum shot capacity."""
        with torch.no_grad():
            item = features_loss
            if pred in cache:
                if len(cache[pred]) < shot_capacity:
                    cache[pred].append(item)
                elif features_loss[1] < cache[pred][-1][1]:
                    cache[pred][-1] = item
                cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
            else:
                cache[pred] = [item]
            return

    def cache_key_value(self, image_features, cache, num_classes):
        """Compute cache prototypes and values from the positive cache."""
        with torch.no_grad():
            cache_keys = []
            cache_values = []
            all_classes = []

            for class_index in sorted(cache.keys()):
                num_items = len(cache[class_index])
                image_prototype = torch.zeros_like(image_features)
                for item in cache[class_index]:
                    image_prototype += item[0] / num_items
                cache_keys.append(image_prototype)
                cache_values.append(class_index)
                all_classes.append(class_index)

            cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
            cache_values = F.one_hot(
                torch.tensor(cache_values, device=image_features.device, dtype=torch.int64),
                num_classes=num_classes,
            ).to(dtype=image_features.dtype)

            return cache_keys, cache_values, all_classes

    def compute_cache_logits(self, image_features, cache_keys, cache_values, alpha, beta):
        affinity = image_features @ cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits

    def _compute_logits(self, image_features, clip_weights):
        return 100.0 * image_features @ clip_weights

    def build_optimizer_dpe(self, text_params, image_params=None):
        opt_name = getattr(self.args, "optimizer", "AdamW").lower()
        wd = getattr(self.args, "weight_decay", 1e-1)
        param_groups = [
            {'params': text_params, 'lr': self.lr_config['text']},
        ]
        if image_params is not None:
            param_groups.append({'params': image_params, 'lr': self.lr_config['image']})

        if opt_name == "sgd":
            momentum = getattr(self.args, "momentum", 0.9)
            return torch.optim.SGD(
                param_groups, lr=self.lr_config['text'], momentum=momentum, weight_decay=wd)
        if opt_name == "adam":
            betas = getattr(self.args, "betas", (0.9, 0.999))
            return torch.optim.Adam(param_groups, lr=self.lr_config['text'], betas=betas, weight_decay=wd)
        else:  # AdamW
            betas = getattr(self.args, "betas", (0.9, 0.999))
            return torch.optim.AdamW(param_groups, lr=self.lr_config['text'], betas=betas, weight_decay=wd)

    def test_time_tuning(self, inputs, clip_weights_local):
        # Get number of classes
        num_classes = clip_weights_local.shape[1]
        
        text_residue = TextResidue(clip_weights_local)

        image_features = self.model.encode_image(inputs)
        new_clip_weights0 = text_residue(clip_weights_local)
        clip_logits = self._compute_logits(image_features, new_clip_weights0)

        if image_features.size(0) > 1:
            output, selected_idx = select_confident_samples(clip_logits, self.args.selection_p)
            image_features_x = image_features[selected_idx].mean(0).unsqueeze(0)
            clip_logits = output.mean(0).unsqueeze(0)
            loss = marginal_entropy(output)
            pred = int(output.mean(0).argmax(dim=-1).item())
        else:
            image_features_x = image_features
            loss = softmax_entropy(clip_logits)
            pred = int(clip_logits.argmax(dim=-1).item())
        
        # Update positive cache
        if self.pos_config['enabled']:
            entropy = get_entropy_normalized(loss, num_classes)
            self.update_cache(
                self.pos_cache,
                pred,
                [image_features_x, entropy],
                self.pos_config['shot_capacity'],
            )

            pos_cache_keys, pos_cache_values, all_classes = self.cache_key_value(
                image_features_x,
                self.pos_cache,
                num_classes,
            )
            pos_cache_residue = PositiveCacheResidue(pos_cache_keys)
        else:
            pos_cache_keys = None
            pos_cache_values = None
            all_classes = None
            pos_cache_residue = None
        
        # Optimization step
        for j in range(self.args.tta_steps):
            new_clip_weights = text_residue(clip_weights_local)
            if self.args.tta_steps > 1:
                clip_logits = self._compute_logits(image_features, new_clip_weights)
            final_logits = clip_logits.clone()
            
            if self.pos_config['enabled'] and self.pos_cache:
                new_pos_cache_keys = pos_cache_residue(pos_cache_keys)
                final_logits = final_logits + self.compute_cache_logits(
                    image_features_x, new_pos_cache_keys, pos_cache_values, 
                    self.pos_config['alpha'], self.pos_config['beta']
                )
                
                # Loss: entropy minimization + alignment
                loss = marginal_entropy(final_logits)
                # Alignment loss between image and text prototypes
                image2text_loss = InfoNCELoss(
                    new_pos_cache_keys.T, 
                    new_clip_weights[:, all_classes].T
                )
                loss += image2text_loss * self.lr_config['align']
            else:
                loss = marginal_entropy(final_logits)
            
            # Setup optimizer
            if self.pos_config['enabled'] and self.pos_cache:
                optimizer = self.build_optimizer_dpe(
                    text_residue.parameters(),
                    pos_cache_residue.parameters(),
                )
            else:
                optimizer = self.build_optimizer_dpe(text_residue.parameters())
            
            optimizer.zero_grad()
            if j == self.args.tta_steps - 1:
                loss.backward()
            else:
                loss.backward(retain_graph=True)
            optimizer.step()
        
        # Return updated prototypes
        text_residue.eval()
        if self.pos_config['enabled'] and self.pos_cache:
            pos_cache_residue.eval()
            with torch.no_grad():
                new_clip_weights = text_residue(clip_weights_local)
                new_pos_cache_keys = pos_cache_residue(pos_cache_keys)
            return new_clip_weights, new_pos_cache_keys, pos_cache_keys, pos_cache_values, all_classes
        else:
            with torch.no_grad():
                new_clip_weights = text_residue(clip_weights_local)
            return new_clip_weights, None, None, None, None

    def test_time_adapt_eval(self, val_loader, result_dict=None):
        """Main evaluation loop with test-time adaptation."""
        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, top1, top5],
            prefix='Test: ')

        self.reset_cache()
        self.model.eval()
        end = time.time()

        num_classes = len(self.classnames)

        with torch.no_grad():
            text_features = self.model.get_text_features()
            self.clip_weights_global = text_features.T.half()


        for i, (images, target) in enumerate(val_loader):
            assert self.args.gpu is not None
            
            # Handle different image formats
            if isinstance(images, list):
                for k in range(len(images)):
                    images[k] = resize_with_CLIP(images[k], self.args.resolution).cuda(self.args.gpu, non_blocking=True)
                image = images[0]
                images = torch.cat(images, dim=0)
            else:
                images = resize_with_CLIP(images, self.args.resolution).cuda(self.args.gpu, non_blocking=True)
                image = images
                
            target = target.cuda(self.args.gpu, non_blocking=True)
            
            # Test-time tuning
            with torch.amp.autocast(device_type='cuda'):
                # if self.args.tta_steps > 0 and self.args.episodic:
                #     # self.reset_cache()
                #     self.clip_weights_global = self.model.get_text_features().T.half()

                # Local optimization: create local copy of global text weights
                clip_weights_local = self.clip_weights_global.clone().detach()

                new_clip_weights, new_pos_cache_keys, pos_cache_keys, pos_cache_values, all_classes = (
                    self.test_time_tuning(images, clip_weights_local)
                )
                
            
            # Inference with adapted prototypes
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda'):
                    image_features = self.model.encode_image(image)
                    clip_logits = self._compute_logits(image_features, new_clip_weights)
                    
                    final_logits = clip_logits.clone()
                    
                    # Add cache logits if available
                    if self.pos_config['enabled'] and self.pos_cache and new_pos_cache_keys is not None:
                        final_logits += self.compute_cache_logits(
                            image_features, new_pos_cache_keys, pos_cache_values,
                            self.pos_config['alpha'], self.pos_config['beta']
                        )
                    
                    # Global update: accumulate high-confidence samples
                    loss = marginal_entropy(final_logits)
                    entropy_normalized = get_entropy_normalized(loss, num_classes)
                    
                    if entropy_normalized < self.global_update_threshold:
                        # Cumulative average update
                        self.num_avg += 1
                        self.clip_weights_global = (
                            self.clip_weights_global * (self.num_avg / (self.num_avg + 1)) + 
                            new_clip_weights * (1 / (self.num_avg + 1))
                        )
            
            # Record results
            if result_dict is not None:
                softmax_output = final_logits.softmax(dim=1)
                max_confidence, max_index = torch.max(softmax_output, 1)
                result_dict['max_confidence'].append(max_confidence.item())
                result_dict['prediction'].append(max_index.item())
                result_dict['label'].append(target.item())
            
            # Measure accuracy
            acc1, acc5 = accuracy(final_logits, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if (i+1) % self.args.print_freq == 0:
                progress.display(i)
        
        progress.display_summary()
        
        return [top1.avg, top5.avg]