
"""
    Bayesian Test-Time Adaptation for Vision-Language Models,
    https://arxiv.org/abs/2503.09248,
    https://github.com/buerzlh/Bayesian-Test-Time-Adaptation-for-Vision-Language-Models
"""

import time
from dataclasses import dataclass

import torch

from ttabc.model_selection.base_method import BaseMethod
from ttabc.utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, select_confident_samples


class BCAState(torch.nn.Module):
    def __init__(self, cfg, init_centers, tem=100.0):
        super(BCAState, self).__init__()
        self.mu = init_centers.clone()
        self.M = self.mu.size(0)
        self.cluster_to_class_prob = torch.eye(self.M).cuda()
        
        ## hyperparameters
        self.threshold1 = cfg.threshold1
        self.c1 = [cfg.init_count1] * self.M
        self.threshold2 = cfg.threshold2
        self.c2 = [cfg.init_count2] * self.M
        self.tem = tem
        self.selection_p = cfg.selection_p
    
    @torch.no_grad()
    def bca_step(self, image_features):
        '''One step of BCA adaptation'''
        # assign labels
        output, image_features = self.assign_label(image_features)
        prob_max, pred = torch.max(output, dim=1)
        
        # update model
        if prob_max > self.threshold1:
            self.update_centers(image_features, pred)
        if prob_max > self.threshold2:
            self.update_prior(output, pred)
        return output
    
    def assign_label(self, image_features):
        # calculate P(x|u_m)
        P_x_um = self.tem * image_features @ self.mu.t()
        if image_features.size(0) > 1:
            output, selected_idx = select_confident_samples(P_x_um, self.selection_p)
            image_features = image_features[selected_idx].mean(0).unsqueeze(0)
            P_x_um = output.mean(0).unsqueeze(0)
        
        ## calculate P(u_m|x_i) =  P(x|u_m)/P(x)
        s1 = P_x_um.softmax(dim=1)
        ## calculate (P(x|u_m)/P(x))*P(Y|u_m)
        f1 = s1 @ self.cluster_to_class_prob
        return f1, image_features
    
    def update_prior(self, soft_prob, pred):
        self.cluster_to_class_prob[pred] = (self.c2[pred]*self.cluster_to_class_prob[pred] + soft_prob)/(self.c2[pred]+1)
        self.c2[pred] = self.c2[pred] + 1
    
    def update_centers(self, image_feature, pred):
        self.mu[pred] = self.c1[pred] * self.mu[pred] + image_feature
        self.c1[pred] = self.c1[pred] + 1
        self.mu[pred] = self.mu[pred] / self.c1[pred]
        self.mu[pred] = self.mu[pred] / torch.norm(self.mu[pred])


class BCA(BaseMethod):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

    def test_time_tuning(self, inputs):
        # BCA performs no parameter updates
        return None

    def get_text_features(self) -> torch.Tensor:
        return self.model.get_text_features()

    def test_time_adapt_eval(self, val_loader, result_dict=None):
        batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
        top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
        top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
        progress = ProgressMeter(len(val_loader), [batch_time, top1, top5], prefix="Test: ")

        self.model.eval()
        with torch.no_grad():
            self.model.reset()

        end = time.time()
        
        # Initialize BCA model
        with torch.no_grad():
            text_features = self.get_text_features()
        bca_model = BCAState(self.args, text_features, tem=self.model.logit_scale.exp()).cuda(self.args.gpu)

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
                    _, image_features = self.model(images)  # (N,K), (N,d)
            
            # BCA Steps
            z = bca_model.bca_step(image_features)
            output = torch.log(z + 1e-12)

            if result_dict is not None:
                softmax_output = z
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