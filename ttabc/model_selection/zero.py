"""
    Frustratingly Easy Test-Time Adaptation of Vision-Language Models,
    https://arxiv.org/abs/2405.18330,
    https://github.com/FarinaMatteo/zero
"""

import time
import operator
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from ttabc.utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, greedy_break
from ttabc.model_selection.base_method import BaseMethod

def select_confident_samples_zero(logits: torch.Tensor, probs: torch.Tensor, top:float, return_idx: bool=False):
    batch_entropy = -(probs * probs.log()).sum(1)
    full_idx = torch.argsort(batch_entropy, descending=False)
    filt_idx = full_idx[:int(batch_entropy.size()[0] * top)]
    if not return_idx:
        return logits[filt_idx]
    return logits[filt_idx], filt_idx, full_idx

class ZERO(BaseMethod):
    '''
        Different from 'zero_shot', ZERO marginilizes the predictions of multiple augmented views after entropy filtering.
    '''
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        
    def test_time_tuning(self, inputs):
        # ZERO does not perform any test-time tuning
        pass

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
        
        #define a softmax layer
        softmax = torch.nn.Softmax(dim=1)

        for i, (images, target) in enumerate(val_loader):
            assert self.args.gpu is not None
            if isinstance(images, list):
                for k in range(len(images)):
                    images[k] = images[k].cuda(self.args.gpu, non_blocking=True)
                image = images[0]
            else:
                images = images.cuda(self.args.gpu, non_blocking=True)
                image = images
            target = target.cuda(self.args.gpu, non_blocking=True)
            images = torch.cat(images, dim=0)

            # compute probabilities and confidence filter
            with torch.no_grad():
                logits_ori, _ = self.model(images)
                logits = logits_ori / self.model.logit_scale.exp() # unscaled logits
                probs = logits_ori.softmax(1)
                logits_filt, _, sorted_idx = select_confident_samples_zero(logits, probs, top=self.args.selection_p, return_idx=True) # retain most confident views

            # zero-out the temperature, marginalize and predict
            zero_temp = torch.finfo(logits_filt.dtype).eps
            p_bar = (logits_filt / zero_temp).softmax(1).sum(0) # marginalize
            
            # check if we have to break ties in some way
            max_counts, scalar_pred = torch.max(p_bar, dim=-1)
            ties = [scalar_pred]
            for idx in range(len(p_bar)):
                if idx == scalar_pred: continue
                if p_bar[idx] == max_counts: ties.append(idx)

            # if so, break ties greedily
            if len(ties) > 1:
                k = int(images.size(0) * self.args.selection_p) 
                sorted_logits = logits[sorted_idx]
                scalar_pred = greedy_break(ties, sorted_logits[k:], device=logits.device)
                p_bar[scalar_pred]+=1

            # need to unsqueeze for compatibility with the 'accuracy' function
            p_bar = p_bar.unsqueeze(0)
            
            if result_dict is not None:
                softmax_output = softmax(p_bar)
                
                #maximum confidence of the softmax_output and its index
                max_confidence, max_index = torch.max(softmax_output, 1)
                
                #save the max confidence, prediction, and label to the result_dict
                result_dict['max_confidence'].append(max_confidence.item())
                result_dict['prediction'].append(max_index.item())
                result_dict['label'].append(target.item())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(p_bar, target, topk=(1, 5))
            
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % self.args.print_freq == 0:
                progress.display(i)

        progress.display_summary()

        return [top1.avg, top5.avg]