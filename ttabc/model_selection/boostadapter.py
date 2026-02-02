"""
    BoostAdapter: Improving Vision-Language Test-Time Adaptation via Regional Bootstrapping,
    https://arxiv.org/abs/2410.15430,
    https://github.com/taolinzhang/BoostAdapter

"""

import time
import operator
import torch
import torch.nn.functional as F
from PIL import Image
import copy

from ttabc.utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, resize_with_CLIP, marginal_entropy, softmax_entropy, select_confident_samples, get_entropy
from ttabc.model_selection.base_method import BaseMethod

def select_confident_samples_boostadapter(feat, logits, topTPT):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idxTPT = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * topTPT)]
    # return feat[idxTPT], logits[idxTPT]
    return feat[idxTPT], logits[idxTPT], idxTPT

def get_output_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    return -(logits * torch.exp(logits)).sum(dim=-1)

class BoostAdapter(BaseMethod):
    """
    Port of training-free BoostAdapter (cache-based) into current framework.
    No parameter update; only builds positive/negative feature caches.
    """

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        
        # Default configurations for positive and negative cache
        self.pos_config = {
            'enabled': True,
            'shot_capacity': 3,
            'alpha': 2.0,
            'beta': 5.0
        }
        self.neg_config = {
            'enabled': True,
            'shot_capacity': 2,
            'alpha': 0.117,
            'beta': 1.0,
            'entropy_threshold': {'lower': 0.2, 'upper': 0.5},
            'mask_threshold': {'lower': 0.03, 'upper': 1.0}
        }
        
        # Override with user configurations if provided
        if args.pos_config:
            self.pos_config.update(self.args.pos_config)
        if args.neg_config:
            self.neg_config.update(self.args.neg_config)
            
        # Positive and negative caches
        self.pos_cache = {}
        self.neg_cache = {}

    def test_time_tuning(self, inputs):
        # BoostAdapter does not perform any test-time tuning
        pass

    def _update_cache(self, cache, pred, features_loss, shot_capacity, include_prob_map=False, fifo=True):
        with torch.no_grad():
            item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
            if pred in cache:
                cache[pred].append(item)
                if fifo: 
                    if len(cache[pred]) > shot_capacity:
                        cache[pred] = cache[pred][1:]
                else:
                    cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
                    cache[pred] = cache[pred][:shot_capacity]   
            else:
                cache[pred] = [item]

    def _compute_cache_logits(self, image_features, cache, alpha, beta, neg_mask_thresholds=None):
        """Compute logits using positive/negative cache."""
        with torch.no_grad():
            cache_keys = []
            cache_values = []
            for class_index in sorted(cache.keys()):
                for item in cache[class_index]:
                    cache_keys.append(item[0])
                    if neg_mask_thresholds:
                        cache_values.append(item[2])
                    else:
                        cache_values.append(class_index)
            if len(cache_keys) == 0:
                return torch.zeros(1)[0]

            cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
            if neg_mask_thresholds:
                cache_values = torch.cat(cache_values, dim=0)
                cache_values = ((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).to(image_features.dtype)
            else:
                cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=len(self.model.classnames))).to(image_features.device, image_features.dtype)

            affinity = image_features @ cache_keys
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            # return alpha * cache_logits, affinity
            return alpha * cache_logits
        
    def _get_clip_logits(self, image_features, clip_logits, infer_ori_image=False):
        ori_feat = image_features.detach().clone()
        ori_output = clip_logits.detach().clone()

        with torch.no_grad():
            if image_features.size(0) > 1:
                if infer_ori_image:
                    prob_map = clip_logits[:1].softmax(1)
                    pred = int(clip_logits[:1].topk(1, 1, True, True)[1].t()[0])
                    
                    loss = softmax_entropy(clip_logits[:1])
                    return image_features[:1], clip_logits[:1], loss, prob_map, pred, ori_feat, ori_output
                else:
                    output, selected_idx = select_confident_samples(clip_logits, self.args.selection_p)
                    image_features = image_features[selected_idx].mean(0).unsqueeze(0)
                    clip_logits = output.mean(0).unsqueeze(0)

                    prob_map = output.softmax(1).mean(0).unsqueeze(0)
                    pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
                    
                    loss = marginal_entropy(output)
            else:
                prob_map = clip_logits.softmax(1)
                pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])
                
                loss = softmax_entropy(clip_logits)

        return image_features, clip_logits, loss, prob_map, pred, ori_feat, ori_output

    def test_time_adapt_eval(self, val_loader, result_dict=None):
        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, top1, top5],
            prefix='Test: ')

        # There is no need to reset the model
        self.model.eval()
        end = time.time()

        for i, (images, target) in enumerate(val_loader):
            assert self.args.gpu is not None
            if isinstance(images, list):
                for k in range(len(images)):
                    images[k] = resize_with_CLIP(images[k], self.args.resolution).cuda(self.args.gpu, non_blocking=True)
                image = images[0]
            else:
                images = resize_with_CLIP(images, self.args.resolution).cuda(self.args.gpu, non_blocking=True)
                image = images
            target = target.cuda(self.args.gpu, non_blocking=True)

            # The actual inference and adaptation go here
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda'):
                    clip_logits, image_features = self.model(image)

            image_features, clip_logits, loss, prob_map, pred, ori_feat, ori_output = self._get_clip_logits(image_features, clip_logits, self.args.infer_ori_image)

            prop_entropy = get_entropy(loss, len(self.model.classnames)) # class-normalized entropy

            # update positive cache
            if self.pos_config['enabled']:
                self._update_cache(
                    self.pos_cache,
                    pred, [image_features, loss],
                    self.pos_config['shot_capacity'],
                    fifo=False)
                
                select_feat, select_output, select_idx = select_confident_samples_boostadapter(ori_feat, ori_output, self.args.selection_p)
                select_entropy = get_output_entropy(select_output)
                
                cur_pos_cache = copy.deepcopy(self.pos_cache)
                for i in range(select_entropy.shape[0]):
                        cur_pred = int(select_output[i].argmax(dim=-1).item())
                        cur_feat = select_feat[i]
                        self._update_cache(
                            cur_pos_cache, cur_pred,
                            [cur_feat.unsqueeze(0), select_entropy[i].item()],
                            self.pos_config['shot_capacity'] + self.args.delta, fifo=False)

            # update negative cache
            if self.neg_config['enabled'] and self.neg_config['entropy_threshold']['lower'] < prop_entropy < self.neg_config['entropy_threshold']['upper']:
                self._update_cache(
                    self.neg_cache,
                    pred, [image_features, loss, prob_map],
                    self.neg_config['shot_capacity'], True)

            # Initialize final logits with CLIP logits
            with torch.no_grad():
                final_logits = clip_logits.clone()
            
            # Compute adapted logits using positive cache
            if self.pos_config['enabled'] and self.pos_cache:
                pos_logits = self._compute_cache_logits(
                    image_features, 
                    self.pos_cache, 
                    self.pos_config['alpha'], 
                    self.pos_config['beta'],
                )
                final_logits += pos_logits
            
            # Compute adapted logits using negative cache
            if self.neg_config['enabled'] and self.neg_cache:
                neg_logits = self._compute_cache_logits(
                    image_features, 
                    self.neg_cache, 
                    self.neg_config['alpha'], 
                    self.neg_config['beta'],
                    (self.neg_config['mask_threshold']['lower'],
                     self.neg_config['mask_threshold']['upper'])
                )
                final_logits -= neg_logits
                
            output = final_logits
            
            if result_dict is not None:
                softmax_output = output.softmax(dim=1)
                
                #maximum confidence of the softmax_output and its index
                max_confidence, max_index = torch.max(softmax_output, 1)
                
                #save the max confidence, prediction, and label to the result_dict
                if max_confidence.numel() == 1:
                    result_dict['max_confidence'].append(max_confidence.item())
                    result_dict['prediction'].append(max_index.item())
                    result_dict['label'].append(target.item())
                else:
                    for j in range(max_confidence.size(0)):
                        result_dict['max_confidence'].append(max_confidence[j].item())
                        result_dict['prediction'].append(max_index[j].item())
                        result_dict['label'].append(target[j].item())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(final_logits, target, topk=(1, 5))
            
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % self.args.print_freq == 0:
                progress.display(i)

        progress.display_summary()

        return [top1.avg, top5.avg]