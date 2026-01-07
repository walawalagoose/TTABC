"""
    HisTPT: Historical Test-time Prompt Tuning for Vision Foundation Models,
    https://arxiv.org/abs/2410.20346,
    https://github.com/jingyi0000/HisTPT
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from ttabc.utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, SoftTargetCrossEntropy, select_confident_samples
from ttabc.model_selection.base_method import BaseMethod


class HistoricalMemory(nn.Module):
    """
    Historical Memory Module for HisTPT
    
    Maintains three types of memory:
    1. ST Memory (Short-Term): Recent 32 samples for capturing local distribution
    2. Hard Memory: Top-16 high-entropy (difficult) samples for focusing on weaknesses
    3. EMA Memory (Global): Exponential moving average for long-term stable information
    """
    def __init__(self, memory_size=32, hard_topk=16, ema_momentum=0.99):
        super().__init__()
        self.memory_size = memory_size
        self.hard_topk = hard_topk
        self.ema_momentum = ema_momentum
        
        # Initialization flags
        self.init_local = True
        self.init_hard = True
        self.init_global = True
        
        # Memory storage (will be initialized on first update)
        self.memory_local = None  # ST memory: [memory_size, num_classes, dim]
        self.memory_hard = None   # Hard memory: [memory_size, num_classes, dim]
        self.memory_ema = None    # EMA memory: [num_classes, dim]
        self.entropy_history = None  # Entropy history: [memory_size]
    
    def update_st_memory(self, text_embeddings, entropy):
        """
        Update Short-Term Memory (ST Memory)
        
        Args:
            text_embeddings: [num_classes, dim] current text embeddings
            entropy: scalar, entropy of current sample
        """
        if self.init_local:
            # First initialization
            self.memory_local = text_embeddings.unsqueeze(0).repeat(
                self.memory_size, 1, 1
            )  # [32, num_classes, dim]
            self.entropy_history = entropy.expand(self.memory_size)
            self.init_local = False
        else:
            # Sliding window update (FIFO queue)
            self.memory_local = torch.cat([
                text_embeddings.unsqueeze(0),
                self.memory_local[:-1]
            ], dim=0)
            
            # Update entropy history
            self.entropy_history = torch.cat([
                entropy.unsqueeze(0),
                self.entropy_history[:-1]
            ], dim=0)
    
    def get_st_prompt(self):
        """Get ST prompt (average of short-term memory)"""
        if self.memory_local is None:
            return None
        return self.memory_local.mean(dim=0)  # [num_classes, dim]
    
    def update_hard_memory(self):
        """
        Update Hard Sample Memory (Hard Memory)
        
        Selects top-k samples with highest entropy from ST memory
        """
        if self.memory_local is None:
            return
        
        if self.init_hard:
            # First initialization
            self.memory_hard = self.memory_local.clone()
            self.init_hard = False
        else:
            # Select top-k difficult samples based on entropy
            _, indices = torch.topk(self.entropy_history, self.hard_topk, dim=0)
            hard_samples = self.memory_local[indices].mean(dim=0)  # [num_classes, dim]
            
            # Update hard memory with FIFO
            self.memory_hard = torch.cat([
                hard_samples.unsqueeze(0),
                self.memory_hard[:-1]
            ], dim=0)
    
    def get_hard_prompt(self):
        """Get Hard prompt (average of hard sample memory)"""
        if self.memory_hard is None:
            return None
        return self.memory_hard.mean(dim=0)  # [num_classes, dim]
    
    def update_ema_memory(self, text_embeddings=None):
        """
        Update EMA Memory (Global Memory)
        
        Args:
            text_embeddings: [num_classes, dim] current text embeddings (optional)
        """
        if self.init_global:
            # First initialization
            if text_embeddings is not None:
                self.memory_ema = text_embeddings.clone()
            self.init_global = False
        else:
            # EMA update: new_value = old * momentum + new * (1 - momentum)
            st_prompt = self.get_st_prompt()
            hard_prompt = self.get_hard_prompt()
            
            if st_prompt is not None and hard_prompt is not None:
                # New global value is the average of ST and Hard prompts
                new_global = 0.5 * (st_prompt + hard_prompt)
                self.memory_ema = (
                    self.memory_ema * self.ema_momentum +
                    new_global * (1 - self.ema_momentum)
                )
    
    def get_ema_prompt(self):
        """Get EMA prompt (global memory)"""
        return self.memory_ema
    
    def reset(self):
        """Reset all memories to initial state"""
        self.init_local = True
        self.init_hard = True
        self.init_global = True
        self.memory_local = None
        self.memory_hard = None
        self.memory_ema = None
        self.entropy_history = None


class HisTPT(BaseMethod):
    """
    HisTPT: Historical Test-time Prompt Tuning for Vision Foundation Models
    Key Features:
    1. Maintains three types of historical memory (ST, Hard, EMA)
    2. Generates pseudo-labels by adaptively fusing three predictions
    3. Updates prompts using cross-entropy loss with pseudo-labels
    """
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.loss_fn = SoftTargetCrossEntropy()
        self.soft = True # Use soft pseudo-labels
        
        # Create historical memory module
        self.historical_memory = HistoricalMemory(
            memory_size=getattr(args, 'memory_size', 32),
            hard_topk=getattr(args, 'hard_topk', 16),
            ema_momentum=getattr(args, 'ema_momentum', 0.99)
        ).cuda(args.gpu)
    
    def compute_entropy(self, logits):
        """
        Compute entropy of predictions
        
        Args:
            logits: [B, num_classes] or [num_classes]
        Returns:
            entropy: sample-wise entropy [B]
        """
        # Ensure logits is 2D
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        
        # Compute entropy: -sum(p * log(p))
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1)
        
        return entropy
    
    def get_text_embeddings_with_memory(self, memory_type='current'):
        """
        Get text embeddings with different memory types
        
        Args:
            memory_type: 'current', 'st', 'hard', 'ema'
        Returns:
            text_embeddings: [num_classes, dim]
        """
        def _l2n(x):
            return x / x.norm(dim=-1, keepdim=True)
        
        # Get current text features from the model
        with torch.no_grad():
            text_features = self.model.get_text_features()  # [num_classes, dim]
        
        if memory_type == 'current':
            return text_features
        elif memory_type == 'st':
            st_prompt = self.historical_memory.get_st_prompt()
            return _l2n(st_prompt) if st_prompt is not None else text_features
        elif memory_type == 'hard':
            hard_prompt = self.historical_memory.get_hard_prompt()
            return _l2n(hard_prompt) if hard_prompt is not None else text_features
        elif memory_type == 'ema':
            ema_prompt = self.historical_memory.get_ema_prompt()
            return _l2n(ema_prompt) if ema_prompt is not None else text_features
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")
    
    def forward_with_memory(self, image, memory_type='current'):
        """
        Forward pass with specified memory type
        
        Args:
            image: input images
            memory_type: 'current', 'st', 'hard', 'ema'
        Returns:
            logits: [B, num_classes]
        """
        # Extract image features
        img_features = self.model.image_encoder(image.type(self.model.dtype))
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        
        # Get text features from memory
        text_features = self.get_text_embeddings_with_memory(memory_type)
        
        # Compute logits
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * img_features @ text_features.t()
        
        return logits
    
    def generate_pseudo_labels(self, image, soft=True):
        """
        Generate pseudo-labels by fusing three predictions
        
        Args:
            image: original images (not augmented)
        Returns:
            pseudo_labels: [B] pseudo labels
            confidence: [B] confidence scores
            soft: whether to return soft labels
        """
        with torch.no_grad():
            # Generate predictions with three types of memory
            logits_ema = self.forward_with_memory(image, 'ema')
            logits_st = self.forward_with_memory(image, 'st')
            logits_hard = self.forward_with_memory(image, 'hard')
            probs_ema = F.softmax(logits_ema, dim=1)   # [B, K]
            probs_st = F.softmax(logits_st, dim=1)     # [B, K]
            probs_hard = F.softmax(logits_hard, dim=1) # [B, K]
            
            # Compute entropy for each prediction (lower is better)
            entropy_ema = self.compute_entropy(logits_ema)
            entropy_st = self.compute_entropy(logits_st)
            entropy_hard = self.compute_entropy(logits_hard)
            
            # Compute weights using softmax (lower entropy gets higher weight)
            entropy_stack = torch.stack([
                -entropy_ema, -entropy_st, -entropy_hard
            ], dim=1)  # Negate so lower entropy gets higher weight
            weights = F.softmax(entropy_stack, dim=1)  # [B, 3]
            
            # Weighted fusion of logits, then generate pseudo-labels
            probs_fused = (
                weights[:, 0:1] * probs_ema +
                weights[:, 1:2] * probs_st +
                weights[:, 2:3] * probs_hard
            )  # [B, K]
            
            if soft:
                pseudo_labels = probs_fused  # Soft labels
                confidence, _ = probs_fused.max(dim=1)
            else:
                confidence, pseudo_labels = probs_fused.max(dim=1)
        
        return pseudo_labels, confidence
    
    def test_time_tuning(self, images_aug, images_ori):
        """
        HisTPT test-time tuning
        
        Args:
            images_aug: augmented images (for training) - shape [B*num_aug, C, H, W]
            images_ori: original images (for generating pseudo-labels) - shape [B, C, H, W]
        """
        num_views = images_aug.size(0) // images_ori.size(0)  # = len(images) when list mode
                
        for j in range(self.args.tta_steps):
            # === Tuning ===
            with torch.amp.autocast(device_type='cuda'):
                # 1. Forward pass with augmented images
                logits_aug = self.model(images_aug)  # [B*num_aug, num_classes]
                
                # 2. Generate pseudo-labels using original images
                pseudo_labels, _ = self.generate_pseudo_labels(images_ori, soft=self.soft)  # [B]
                
                # 3. Filter sample with entropy selection
                selected_idx = torch.arange(logits_aug.size(0))
                if self.args.tpt:
                    logits_aug, selected_idx = select_confident_samples(logits_aug, self.args.selection_p)
                
                # 4. Expand pseudo-labels to match augmented images batch size
                # 5. Compute loss (cross-entropy with pseudo-labels)
                if self.soft:  # [num_views*B, K]
                    pseudo_labels = pseudo_labels.repeat(num_views, 1)[selected_idx]
                    loss = self.loss_fn(logits_aug, pseudo_labels)
                else:  # [num_views*B]
                    pseudo_labels = pseudo_labels.repeat(num_views)[selected_idx]
                    loss = F.cross_entropy(logits_aug, pseudo_labels)
                
            # 5. Optimization step
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
        # === AFTER tuning: update memory once ===
        with torch.no_grad():    
            # 1. Compute entropy of current sample
            # TODO: take average entropy over augmentations?
            entropy = self.compute_entropy(logits_aug).mean()
            # entropy = self.compute_entropy(logits_aug)[0]
            
            # 2. Update historical memory
            current_text_features = self.model.get_text_features()
            self.historical_memory.update_st_memory(
                current_text_features, entropy
            )
            self.historical_memory.update_hard_memory()
            self.historical_memory.update_ema_memory(current_text_features)
                
    
    def test_time_adapt_eval(self, val_loader, result_dict=None):
        """
        HisTPT test-time adaptation and evaluation
        
        Args:
            val_loader: validation data loader
            result_dict: dictionary to store results for ECE calculation
        Returns:
            [top1_accuracy, top5_accuracy]
        """
        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, top1, top5],
            prefix='Test: ')

        # Reset model
        self.model.eval()
        with torch.no_grad():
            self.model.reset()
        
        # Reset historical memory (important: memory persists across samples)
        self.historical_memory.reset()
        
        end = time.time()
        softmax = torch.nn.Softmax(dim=1)
    
        for i, (images, target) in enumerate(val_loader):
            assert self.args.gpu is not None
            
            # Handle input images
            if isinstance(images, list):
                # TPT mode: images is a list of augmented images
                for k in range(len(images)):
                    images[k] = images[k].cuda(self.args.gpu, non_blocking=True)
                image_ori = images[0]  # Original image
                images_aug = torch.cat(images, dim=0)  # Augmented images
            else:
                if len(images.size()) > 4:
                    assert images.size()[0] == 1
                    images = images.squeeze(0)
                images = images.cuda(self.args.gpu, non_blocking=True)
                image_ori = images
                images_aug = images
            
            target = target.cuda(self.args.gpu, non_blocking=True)

            # Test-time tuning
            if self.args.tta_steps > 0:
                # Reset prompt (but NOT historical memory - that's the key!)
                # TODO: Test if resetting is needed
                # with torch.no_grad():
                #     self.model.reset()
                # self.optimizer.load_state_dict(self.optim_state)
                
                # Perform tuning
                self.test_time_tuning(images_aug, image_ori)

            # Inference
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda'):
                    output = self.model(image_ori)

            # Save results for ECE calculation
            if result_dict is not None:
                softmax_output = softmax(output)
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

            # Measure accuracy
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image_ori.size(0))
            top5.update(acc5[0], image_ori.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % self.args.print_freq == 0:
                progress.display(i)

        progress.display_summary()
        return [top1.avg, top5.avg]