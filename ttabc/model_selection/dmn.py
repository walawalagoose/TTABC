"""
    Dual Memory Networks: A Versatile Adaptation Approach for Vision-Language Models
    https://arxiv.org/abs/2403.18293
    https://github.com/YBZh/DMN
"""

import time
import math
import torch
import torch.nn as nn

from ttabc.utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, resize_with_CLIP, marginal_entropy, get_entropy
from ttabc.model_selection.base_method import BaseMethod


def _l2_normalize(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def select_confident_samples_dmn(prob, top):
    batch_entropy = -(prob * torch.log(prob + 1e-6)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return prob[idx], idx


class DualMem(nn.Module):
    def __init__(self, args=None, beta=5.5, feat_dim=1024, class_num=1000, mapping='bias'):
        super(DualMem, self).__init__()
        self.args = args
        self.beta = beta
        self.rank = 4
        self.init_pred = 0
        if args.shared_param:
            self.global_affine = nn.Parameter(torch.zeros((feat_dim, feat_dim)))
            self.global_bias = nn.Parameter(torch.zeros((class_num, feat_dim)))  ## unknown use the category mean.
            self.global_bias_key = self.global_bias
            self.global_bias_value = self.global_bias

            self.global_ffn_affine = nn.Parameter(torch.zeros((feat_dim, feat_dim)))
            self.global_ffn_bias = nn.Parameter(torch.zeros((class_num, feat_dim)))  ## unknown use the category mean.
            self.text_affine = self.global_ffn_affine
            self.text_bias = self.global_ffn_bias
        else:
            self.global_affine = nn.Parameter(torch.zeros((feat_dim, feat_dim)))
            self.global_bias = nn.Parameter(torch.zeros((class_num, feat_dim)))  ## unknown use the category mean.
            self.global_bias_key = nn.Parameter(torch.zeros((class_num, feat_dim)))  ## unknown use the category mean.
            self.global_bias_value = nn.Parameter(torch.zeros((class_num, feat_dim)))  ## unknown use the category mean.

            self.global_ffn_affine = nn.Parameter(torch.zeros((feat_dim, feat_dim)))
            self.global_ffn_bias = nn.Parameter(torch.zeros((class_num, feat_dim)))  ## unknown use the category mean.
            self.text_affine = nn.Parameter(torch.zeros((feat_dim, feat_dim)))
            self.text_bias = nn.Parameter(torch.zeros((class_num, feat_dim)))
        self.learnable_mapping = args.mapping  ### bias | affine | all

    def update_memory_bank(self, model, target):
        # updating
        mean_prob = self.init_pred[0].float()
        value, indice = mean_prob.max(0)
        pseudo_label = indice.item()
        # print(value, indice, target)
        text_features = model.text_feat[pseudo_label]  ## 512 (not used, keep as official)
        selected_image_features_global = model.image_features_global[:1]
        current_instance_entropy = -(mean_prob * (torch.log(mean_prob + 1e-8))).sum()
        if model.image_feature_count[pseudo_label] == model.memory_size:
            ###### if the new one is low entropy, find the sample with the max entropy, and replace it with the new one
            if (current_instance_entropy < model.image_entropy_mem[pseudo_label]).sum() == 0:
                pass  ## the entropy of current test image is very large.
            else:
                _, indice = torch.sort(model.image_entropy_mem[pseudo_label])
                to_replace_indice = indice[-1]  ## with max entropy, ascending.
                model.image_feature_memory[pseudo_label][to_replace_indice] = selected_image_features_global
                model.image_prediction_mem[pseudo_label][to_replace_indice] = mean_prob[0]
                model.image_entropy_mem[pseudo_label][to_replace_indice] = current_instance_entropy
        else:
            model.image_feature_memory[pseudo_label][model.image_feature_count[pseudo_label, 0].item()] = selected_image_features_global
            model.image_prediction_mem[pseudo_label][model.image_feature_count[pseudo_label, 0].item()] = mean_prob[0]
            model.image_entropy_mem[pseudo_label][model.image_feature_count[pseudo_label, 0].item()] = current_instance_entropy
            model.image_feature_count[pseudo_label] += 1

    def get_image_pred(self, model, return_full=False, return_logit=False):
        ## prediction with dynamic memory.
        img_feat = model.image_features_global[:1]  # 1*D
        count_image_feat = model.image_feature_count.clone()
        num_class = model.image_feature_memory.shape[0]
        image_classifier = 'similarity_weighted'  ## category_center | entropy_weighted | similarity_weighted
        ### similarity_weighted achieves the best results.
        memorized_image_feat = torch.cat((model.image_feature_memory, model.fixed_global_feat_vanilla), dim=1)  ## C*(M+1)*D

        if image_classifier == 'similarity_weighted':  ## this is an instance adaptative method.
            ## calculate the cos similarity betweeen image feature and memory feature, and then weighted the memorized features according to similarity.
            ###################### 有一些memory 是空的，现在却往里面塞了一个self.global_bias， 这不合理，还要把它继续置空。
            img_feat_mappling = img_feat
            memorized_image_feat_K = memorized_image_feat
            memorized_image_feat_V = memorized_image_feat
            with torch.no_grad():
                if self.args.position == 'query':
                    img_feat_mappling = img_feat + self.global_bias.mean(0, keepdim=True)  ## N*D
                elif self.args.position == 'key':
                    memorized_image_feat_K = memorized_image_feat + self.global_bias_key.unsqueeze(1)  ## class*shot*D
                elif self.args.position == 'value':
                    memorized_image_feat_V = memorized_image_feat + self.global_bias_value.unsqueeze(1)  ## class*shot*D
                elif self.args.position == 'qkv' or self.args.position == 'all':
                    img_feat_mappling = img_feat + self.global_bias.mean(0, keepdim=True)  ## N*D
                    memorized_image_feat_K = memorized_image_feat + self.global_bias_key.unsqueeze(1)  ## class*shot*D
                    memorized_image_feat_V = memorized_image_feat + self.global_bias_value.unsqueeze(1)  ## class*shot*D
                else:
                    pass
                memorized_image_feat_K = memorized_image_feat_K / memorized_image_feat_K.norm(dim=-1, keepdim=True)
                ## some memorized_image_feat slots are empty before mapping, reseting them to empty.
                memorized_image_feat_K[memorized_image_feat.sum(-1) == 0] = 0
                memorized_image_feat_V = memorized_image_feat_V / memorized_image_feat_V.norm(dim=-1, keepdim=True)
                memorized_image_feat_V[memorized_image_feat.sum(-1) == 0] = 0
                img_feat_mappling = img_feat_mappling / img_feat_mappling.norm(dim=-1, keepdim=True)

            similarity_matrix = (img_feat_mappling * memorized_image_feat_K).sum(-1)  ## class*(shot)
            similarity_matrix = torch.exp(-self.beta * (-similarity_matrix + 1))
            ### weighting memoried features with similarity weights.
            adaptive_image_feat = (memorized_image_feat_V * similarity_matrix.unsqueeze(-1)).sum(1)
            ## torch.Size([class, dim])
            adaptive_image_feat = adaptive_image_feat / adaptive_image_feat.norm(dim=-1, keepdim=True)
            if self.args.position == 'output' or self.args.position == 'all':
                adaptive_image_feat = adaptive_image_feat + self.global_ffn_bias  ## class*D

            adaptive_image_feat = adaptive_image_feat / adaptive_image_feat.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * adaptive_image_feat @ img_feat.unsqueeze(-1)  ## class*1
            logits = logits[:, 0].unsqueeze(0)  ## 1*class
            if return_logit:
                return logits
            else:
                return logits.softmax(dim=1)
        else:
            raise NotImplementedError


class DMN(BaseMethod):
    """Zero-shot DMN with a dynamic memory bank."""

    def __init__(self, args):
        super().__init__(args)
        self.args = args

        # keep for compatibility; no longer used by the corrected DMN-ZS path
        self.mem_feats = {}
        self.mem_entropy = {}

        # official DMN-ZS core module (initialized lazily when feat_dim / num_classes are known)
        self.dmnet = None
        self._dmn_inited = False

    def test_time_tuning(self, inputs):
        pass

    @torch.no_grad()
    def _init_dmn_state(self, feat_dim, num_classes, device, dtype):
        if self._dmn_inited:
            return

        # text features
        text_feat = self.model.get_text_features().to(device=device, dtype=dtype)
        text_feat = _l2_normalize(text_feat)

        # Attach official-expected buffers/attrs onto the model (like official fixed_clip)
        self.model.text_feat = text_feat
        self.model.fixed_global_feat_vanilla = text_feat.unsqueeze(1)  # [K, 1, D]
        self.model.memory_size = int(self.args.memory_size)

        self.model.image_feature_memory = torch.zeros(num_classes, self.model.memory_size, feat_dim, device=device, dtype=dtype)
        self.model.image_prediction_mem = torch.zeros( num_classes, self.model.memory_size, num_classes, device=device, dtype=dtype)
        self.model.image_entropy_mem = torch.full((num_classes, self.model.memory_size), float("inf"), device=device, dtype=dtype)
        self.model.image_feature_count = torch.zeros(num_classes, 1, device=device, dtype=torch.long)

        # DualMem module
        self.dmnet = DualMem(
            args=self.args, beta=float(self.args.beta),
            feat_dim=feat_dim, class_num=num_classes,
            mapping=self.args.mapping).to(device)

        self._dmn_inited = True

    def test_time_adapt_eval(self, val_loader, result_dict=None):
        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, top1, top5],
            prefix='Test: ',
        )

        self.model.eval()
        end = time.time()

        text_weight = getattr(self.args, "text_weight", 1.0)
        mem_weight = getattr(self.args, "mem_weight", 0.03)

        for i, (images, target) in enumerate(val_loader):
            assert self.args.gpu is not None

            if isinstance(images, list):
                for k in range(len(images)):
                    images[k] = resize_with_CLIP(images[k], self.args.resolution).cuda(self.args.gpu, non_blocking=True)
                image = torch.cat(images, dim=0)
            else:
                images = resize_with_CLIP(images, self.args.resolution).cuda(
                    self.args.gpu, non_blocking=True
                )
                image = images
            target = target.cuda(self.args.gpu, non_blocking=True)

            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda'):
                    clip_logits, image_features = self.model(image)

            if clip_logits.dim() == 3 and clip_logits.size(1) == 1:
                clip_logits = clip_logits.squeeze(1)

            # lazy init DMN state once we know dims/classes/dtype
            num_classes = len(self.model.classnames)
            feat_dim = image_features.size(-1)
            self._init_dmn_state(
                feat_dim=feat_dim,
                num_classes=num_classes,
                device=image_features.device,
                dtype=image_features.dtype,
            )

            # set model.image_features_global for official DualMem usage
            self.model.image_features_global = image_features

            # text prob for all views
            prob_text = clip_logits.softmax(dim=1)

            # confidence selection on views
            if image_features.size(0) > 1:
                confidence_prediction, _ = select_confident_samples_dmn(prob_text, self.args.selection_p)
                init_pred = confidence_prediction.mean(0, keepdim=True)
            else:
                init_pred = prob_text.mean(0, keepdim=True)

            # dmnet.init_pred drives pseudo-label + entropy for memory update
            self.dmnet.init_pred = init_pred

            # update dynamic memory using pseudo-label from init_pred (text-based)
            self.dmnet.update_memory_bank(self.model, target)

            # current text prediction uses the first view
            logits_text = prob_text[:1]

            # dynamic memory prediction (prob)
            logits_mem = self.dmnet.get_image_pred(self.model)

            # final output (prob fusion; matches Eq.(12) style)
            output = text_weight * logits_text + mem_weight * logits_mem

            if output.dim() == 3 and output.size(1) == 1:
                output = output.squeeze(1)

            if result_dict is not None:
                softmax_output = output
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

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], output.size(0))
            top5.update(acc5[0], output.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % self.args.print_freq == 0:
                progress.display(i)

        progress.display_summary()
        return [top1.avg, top5.avg]
