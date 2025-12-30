# -*- coding: utf-8 -*-
"""
BATCLIP: Bimodal Online Test-Time Adaptation for CLIP
Paper: https://arxiv.org/abs/2412.02837
Based on ICCV 2025 paper by Maharana et al.

This implementation follows the official BATCLIP code and adapts it to TTABC framework.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from ttabc.utils.tools import Summary, AverageMeter, ProgressMeter, accuracy
from ttabc.model_selection.base_method import BaseMethod


# ============================================================================
# Loss Functions (from official BATCLIP implementation)
# ============================================================================

class Entropy(nn.Module):
    """Entropy loss for prediction confidence"""
    def __init__(self):
        super(Entropy, self).__init__()

    def __call__(self, logits):
        return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


class I2TLoss(nn.Module):
    """
    Image-to-Text alignment loss.
    Aligns mean image features per class with corresponding text features.
    """
    def __init__(self):
        super(I2TLoss, self).__init__()

    def __call__(self, logits, img_feats, text_norm_feats):
        """
        Args:
            logits: Model predictions [B, num_classes]
            img_feats: Image features [B, D]
            text_norm_feats: Normalized text features [num_classes, D]
        
        Returns:
            Average cosine similarity between class prototypes and text features
        """
        labels = torch.argmax(logits.softmax(1), dim=1)
        loss = 0.0
        for l in torch.unique(labels, sorted=True).tolist():
            img_idx_embeddings = img_feats[labels == l]
            mean_feats = img_idx_embeddings.mean(0).type(text_norm_feats.dtype)
            dist = torch.matmul(mean_feats.unsqueeze(0), text_norm_feats[l].unsqueeze(0).t()).mean()
            loss += dist
        return loss / len(torch.unique(labels))


class InterMeanLoss(nn.Module):
    """
    Inter-class mean loss.
    Maximizes separation between different class prototypes.
    """
    def __init__(self):
        super(InterMeanLoss, self).__init__()
        
    def __call__(self, logits, img_feats):
        """
        Args:
            logits: Model predictions [B, num_classes]
            img_feats: Image features [B, D]
        
        Returns:
            Sum of cosine similarities between different class prototypes
        """
        labels = torch.argmax(logits.softmax(1), dim=1)
        mean_feats = []
        for l in torch.unique(labels, sorted=True).tolist():
            img_idx_embeddings = img_feats[labels == l]
            mean = img_idx_embeddings.mean(0)
            mean_feats.append(mean / mean.norm())

        cosine_sim_matrix = torch.matmul(torch.stack(mean_feats), torch.stack(mean_feats).t())
        loss = 1 - cosine_sim_matrix
        loss.fill_diagonal_(0)
        return loss.sum()


# ============================================================================
# BATCLIP Method
# ============================================================================

class BATCLIP(BaseMethod):
    """
    BATCLIP: Bimodal Online Test-Time Adaptation for CLIP
    
    Key Features:
    1. Adapts only normalization layers (BatchNorm, LayerNorm, GroupNorm)
    2. Uses three loss components:
       - Entropy minimization for confident predictions
       - I2T loss for image-text alignment
       - InterMean loss for class separation
    3. Online adaptation during test-time
    """
    
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        
        # Setup loss functions (official BATCLIP losses)
        self.softmax_entropy = Entropy()
        self.i2t_loss = I2TLoss()
        self.inter_mean_loss = InterMeanLoss()
        
        # Configure model for BATCLIP adaptation
        self.configure_model_for_batclip()
        
        print(f"BATCLIP initialized with lr={self.args.lr}, tta_steps={self.args.tta_steps}")
    
    def configure_model_for_batclip(self):
        """
        Configure model for BATCLIP adaptation (official implementation).
        - Freeze all parameters
        - Enable only normalization layers for adaptation
        - Disable BatchNorm2d running statistics tracking
        - CRITICAL: Freeze prompts if using TPT to match original BATCLIP behavior
        """
        self.model.eval()
        
        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False
        
        # CRITICAL FIX: Explicitly freeze prompts for TPT models
        # The original BATCLIP uses ZeroShotCLIP (no prompts), so prompts should NOT be adapted
        if hasattr(self.model, 'prompt_learner'):
            print("BATCLIP: Freezing TPT prompts (only normalization layers will be adapted)")
            for param in self.model.prompt_learner.parameters():
                param.requires_grad = False
        
        # Collect parameters to optimize
        params_to_optimize = []
        
        # Enable normalization layers for adaptation
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
                m.train()
                for param in m.parameters():
                    param.requires_grad = True
                    params_to_optimize.append(param)
            elif isinstance(m, nn.BatchNorm2d):
                m.train()
                for param in m.parameters():
                    param.requires_grad = True
                    params_to_optimize.append(param)
                # Disable running statistics tracking (official BATCLIP)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
        
        # Re-create optimizer with only normalization layer parameters
        if len(params_to_optimize) > 0:
            self.optimizer = torch.optim.AdamW(
                params_to_optimize,
                lr=self.args.lr,
                betas=(0.9, 0.999),
                weight_decay=0.01
            )
            # Update optim_state
            from copy import deepcopy
            self.optim_state = deepcopy(self.optimizer.state_dict())
            print(f"BATCLIP: Optimizing {len(params_to_optimize)} normalization layer parameters")
        else:
            print("BATCLIP: Warning - No normalization layer parameters found!")

    
    def test_time_tuning(self, images):
        """
        Perform test-time tuning on a batch of images.
        
        Args:
            images: Input images (already concatenated if using augmentations)
        """
        for j in range(self.args.tta_steps):
            with torch.amp.autocast(device_type='cuda'):
                # CRITICAL FIX: Manually replicate the model's forward pass to extract features
                # The model's forward method processes images correctly (including normalization)
                # We need to replicate this to get the same features as the original BATCLIP
                
                # Replicate ClipTestTimeTuning.real_forward() to extract intermediate features
                # See: TTABC/clip/custom_clip.py lines 311-321
                
                # Step 1: Extract image features (same as model.real_forward)
                # The image_encoder in CLIP expects raw images (normalization is in preprocessing)
                img_pre_features = self.model.image_encoder(images.type(self.model.dtype))
                
                # Step 2: Get text features (same as model.real_forward)
                text_features = self.model.get_text_features()
                
                # Step 3: Normalize image features for logits computation
                img_features = img_pre_features / img_pre_features.norm(dim=-1, keepdim=True)
                
                # Step 4: Compute logits (same as model.real_forward)
                logit_scale = self.model.logit_scale.exp()
                logits = logit_scale * img_features @ text_features.t()
                
                # Step 5: Compute BATCLIP losses
                # IMPORTANT: Use img_pre_features (features before L2 normalization) 
                # for I2T and InterMean losses, matching original BATCLIP
                loss = self.softmax_entropy(logits).mean(0)
                i2t_loss = self.i2t_loss(logits, img_pre_features, text_features)
                inter_mean_loss = self.inter_mean_loss(logits, img_pre_features)
                
                # Combine losses (subtract to maximize similarity/separation)
                loss -= i2t_loss
                loss -= inter_mean_loss
            
            # Optimization step
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

    
    def test_time_adapt_eval(self, val_loader, result_dict=None):
        """
        Perform test-time adaptation and evaluation.
        
        Args:
            val_loader: Validation data loader
            result_dict: Dictionary to store results for ECE calculation
        
        Returns:
            List of [top1_accuracy, top5_accuracy]
        """
        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, top1, top5],
            prefix='Test: ')

        # Reset model and switch to evaluate mode
        self.model.eval()
        with torch.no_grad():
            self.model.reset()
        end = time.time()

        # Define a softmax layer
        softmax = torch.nn.Softmax(dim=1)
    
        for i, (images, target) in enumerate(val_loader):
            assert self.args.gpu is not None
            if isinstance(images, list):
                for k in range(len(images)):
                    images[k] = images[k].cuda(self.args.gpu, non_blocking=True)
                image = images[0]
            else:
                if len(images.size()) > 4:
                    # when using ImageNet Sampler as the dataset
                    assert images.size()[0] == 1
                    images = images.squeeze(0)
                images = images.cuda(self.args.gpu, non_blocking=True)
                image = images
            target = target.cuda(self.args.gpu, non_blocking=True)
            
            if self.args.tpt:
                images = torch.cat(images, dim=0)

            # Reset the tunable prompt to its initial state
            if self.args.tta_steps > 0:
                with torch.no_grad():
                    self.model.reset()
                self.optimizer.load_state_dict(self.optim_state)
                
                # Perform test-time tuning
                self.test_time_tuning(images)

            # The actual inference goes here
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda'):
                    output = self.model(image)

            if result_dict is not None:
                softmax_output = softmax(output) 
                # Maximum confidence of the softmax_output and its index
                max_confidence, max_index = torch.max(softmax_output, 1)
                # Save the max confidence, prediction, and label to the result_dict
                # Handle both single samples and batches
                if max_confidence.numel() == 1:
                    # Single sample
                    result_dict['max_confidence'].append(max_confidence.item())
                    result_dict['prediction'].append(max_index.item())
                    result_dict['label'].append(target.item())
                else:
                    # Batch of samples
                    for i in range(max_confidence.size(0)):
                        result_dict['max_confidence'].append(max_confidence[i].item())
                        result_dict['prediction'].append(max_index[i].item())
                        result_dict['label'].append(target[i].item())


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
