"""
    Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models,
    https://arxiv.org/abs/2209.07511,
    https://azshue.github.io/TPT/
"""

import time

from PIL import Image

import torch
import torch.optim

from ttabc.utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, select_confident_samples, marginal_entropy
from ttabc.model_selection.base_method import BaseMethod

class TPT(BaseMethod):
    def __init__(self, args):
        super().__init__(args)
        self.args = args 

    def test_time_tuning(self, inputs):
        if self.args.prompt_type == 'cocoop':
            image_feature, pgen_ctx = inputs
            pgen_ctx.requires_grad = True
            self.optimizer = torch.optim.AdamW([pgen_ctx], self.args.lr)
        
        selected_idx = None
        for j in range(self.args.tta_steps):
            with torch.amp.autocast(device_type='cuda'):
                if self.args.prompt_type == 'maple':
                    output = self.model(inputs)['logits']
                elif self.args.prompt_type == 'cocoop':
                    output = self.model((image_feature, pgen_ctx))
                else:
                    output = self.model(inputs) 

                if selected_idx is not None:
                    output = output[selected_idx]
                else:
                    output, selected_idx = select_confident_samples(output, self.args.selection_p)

                loss = marginal_entropy(output)
            self.optimizer.zero_grad()
            # compute gradient and do SGD step
            self.scaler.scale(loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            self.scaler.step(self.optimizer)
            self.scaler.update()
        if self.args.prompt_type == 'cocoop':
            return pgen_ctx

        return

    def test_time_adapt_eval(self, val_loader, result_dict=None):
        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, top1, top5],
            prefix='Test: ')

        # reset model and switch to evaluate mode
        self.model.eval()
        if self.args.prompt_type != 'cocoop': # no need to reset cocoop because it's fixed
            with torch.no_grad():
                self.model.reset()
        end = time.time()
    
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

            # reset the tunable prompt to its initial state
            if self.args.prompt_type != 'cocoop': # no need to reset cocoop because it's fixed
                if self.args.tta_steps > 0:
                    if self.args.episodic:
                        with torch.no_grad():
                            self.reset()
                self.optimizer.load_state_dict(self.optim_state)

                self.test_time_tuning(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type='cuda'):
                        image_feature, pgen_ctx = self.model.gen_ctx(images, self.args.tpt)
                self.optimizer = None

                pgen_ctx = self.test_time_tuning((image_feature, pgen_ctx))

            # The actual inference goes here
            if self.args.tpt:
                if self.args.prompt_type == 'cocoop':
                    image_feature = image_feature[0].unsqueeze(0)
            
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda'):
                    if self.args.prompt_type == 'maple':
                        output = self.model(image)['logits']
                    elif self.args.prompt_type == 'cocoop':
                        output = self.model((image_feature, pgen_ctx))
                    else:
                        output = self.model(image)


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
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % self.args.print_freq == 0:
                progress.display(i)

        progress.display_summary()

        return [top1.avg, top5.avg]