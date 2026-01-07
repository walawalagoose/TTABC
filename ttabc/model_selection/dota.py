"""
DOTA (as in this repo's `DOTA/dota_gda_em_aug.py`) integrated into TTABC.

Key characteristics of the original implementation:
- training-free, online (streaming) updates
- uses multi-view augmentations; selects low-entropy views
- updates class-wise Gaussian statistics with soft pseudo-labels (from CLIP)
- fuses CLIP logits with DOTA's GDA logits
"""

import time
from dataclasses import dataclass
from typing import Optional

import torch

from clip import tokenize
from data.imagnet_prompts import tip_imagenet_templates
from data.cifar_prompts import cifar_templates
from ttabc.model_selection.base_method import BaseMethod
from ttabc.utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, select_confident_samples


class DOTAState(torch.nn.Module):
    """
    Online class-conditional Gaussian model with a shared (mean) covariance inverse.
    """
    def __init__(self, cfg, input_dim, num_classes, clip_weights, streaming_update_Sigma=True):
        super(DOTAState, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.streaming_update_Sigma = streaming_update_Sigma
        self.epsilon = cfg.epsilon
        self.mu = clip_weights.T.to(self.device)  # initialize mu with clip_weights
        self.c = torch.ones(num_classes, dtype=torch.float32).to(self.device)
        self.Sigma = cfg.sigma * torch.eye(input_dim, dtype=torch.float32).repeat(num_classes, 1, 1).to(self.device)
        self.overall_Sigma = torch.mean(self.Sigma, dim=0)
        self.Lambda = torch.pinverse(self.overall_Sigma.double()).to(self.device).half()

    # Update the covariance and the mean for the corresponding category
    def fit(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)  # y is now a probability distribution (soft labels)
        with torch.no_grad():
            sum_weights = torch.sum(y, dim=0)  
            weighted_x = torch.matmul(y.T, x)  
            new_mu = (weighted_x + self.c.unsqueeze(1) * self.mu) / (sum_weights.unsqueeze(1) + self.c.unsqueeze(1)) 
            new_c = self.c + sum_weights

            # Update the covariance matrix for each category
            if self.streaming_update_Sigma:
                x_minus_mu = x.unsqueeze(1) - self.mu.unsqueeze(0)  # Shape: (batch_size, num_classes, input_shape)
                weighted_x_minus_mu = y.unsqueeze(2) * x_minus_mu  # Shape: (batch_size, num_classes, input_shape)
                delta = torch.einsum('bji,bjk->jik', weighted_x_minus_mu, x_minus_mu)  # Shape: (num_classes, input_shape, input_shape)
                self.Sigma = (self.c[:, None, None] * self.Sigma + delta) / (self.c[:, None, None] + sum_weights[:, None, None])

            # Update the total covariance matrix, mean matrix, and count sections
            self.overall_Sigma = torch.mean(self.Sigma, dim=0)
            self.mu = new_mu
            self.c = new_c
            
    # Update the inverse matrix to include a small identity matrix when calculating the inverse matrix to ensure that it is full rank
    def update(self):
        self.Lambda = torch.inverse(
            (1 - self.epsilon) * self.overall_Sigma + self.epsilon * torch.eye(self.input_dim).to(
            self.device)).half()

    # Calculate the results of Dota predictions
    def predict(self, X):
        X = X.to(self.device)
        with torch.no_grad():
            Lambda = self.Lambda
            M = self.mu.transpose(1, 0).half()
            W = torch.matmul(Lambda, M)  
            c = 0.5 * torch.sum(M * W, dim=0)
            scores = torch.matmul(X, W) - c
            return scores


class DOTA(BaseMethod):
    """
    Training-free DOTA in TTABC's method interface.
    """

    def __init__(self, args):
        super().__init__(args)
        self.args = args

    def test_time_tuning(self, inputs):
        # DOTA performs no parameter updates
        return None

    def get_text_features(self):
        return self.model.get_text_features()

    def test_time_adapt_eval(self, val_loader, result_dict=None):
        batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
        top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
        top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
        
        progress = ProgressMeter(len(val_loader), [batch_time, top1, top5], prefix="Test: ")

        self.model.eval()
        with torch.no_grad():
            self.model.reset()

        softmax = torch.nn.Softmax(dim=1)
        end = time.time()

        for i, (images, target) in enumerate(val_loader):
            assert self.args.gpu is not None

            if isinstance(images, list):
                for k in range(len(images)):
                    images[k] = images[k].cuda(self.args.gpu, non_blocking=True)
                image = images[0]
                images = torch.cat(images, dim=0)
            else:
                images = images.cuda(self.args.gpu, non_blocking=True)
                image = images
            target = target.cuda(self.args.gpu, non_blocking=True)

            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda"):
                    clip_logits, image_features = self.model(images)
                    prob_map = clip_logits.softmax(dim=-1)  # (k,K)

            # Entropy selection
            selected_idx = torch.arange(clip_logits.size(0))
            if self.args.tpt:
                clip_logits, selected_idx = select_confident_samples(clip_logits, self.args.selection_p)
                image_features = image_features[selected_idx]
                prob_map = prob_map[selected_idx]

            # Initialize DOTA model
            text_features = self.get_text_features()
            K, d = int(text_features.size(0)), int(text_features.size(1))
            if self.args.init_mu == "clip_text":
                mu_init = text_features
            else:
                mu_init = torch.full((d, K), float(self.args.init_mu_value))
            dota_model = DOTAState(self.args, d, K, clip_weights=mu_init).cuda(self.args.gpu)
            dota_model.update()

            # CLIP side: average logits over selected views
            clip_logits = clip_logits.mean(dim=0, keepdim=True)  # (1,K)
            
            # DOTA side: predict from mean feature
            dota_logits = dota_model.predict(image_features.mean(0).unsqueeze(0).half())  # (1,K)
            
            # Choose a smaller weight, so that model relies more on the original clip initially
            dota_weights = torch.clamp(self.args.rho * dota_model.c.mean() / image_features.size(0), max=self.args.eta)   
            # Clip and Dota prediction weights are added to form the final prediction
            final_logits = clip_logits + dota_weights * dota_logits
            output = final_logits
            
            # Online update
            dota_model.fit(image_features, prob_map)
            dota_model.update() # Inverse covariance matrix

            if result_dict is not None:
                softmax_output = softmax(output)
                max_confidence, max_index = torch.max(softmax_output, dim=1)
                if max_confidence.numel() == 1:
                    result_dict['max_confidence'].append(max_confidence.item())
                    result_dict['prediction'].append(max_index.item())
                    result_dict['label'].append(target.item())
                else:
                    for j in range(max_confidence.size(0)):
                        result_dict['max_confidence'].append(max_confidence[j].item())
                        result_dict['prediction'].append(max_index[j].item())
                        result_dict['label'].append(target[j].item())

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.args.print_freq == 0:
                progress.display(i)

        progress.display_summary()
        return [top1.avg, top5.avg]