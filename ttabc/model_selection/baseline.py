import time

from PIL import Image

import torch
import torch.optim


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from ttabc.utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed, ece_calculator, ECE_Loss, select_confident_samples, avg_entropy
from ttabc.model_selection.base_method import BaseMethod

import ipdb

class BASELINE(BaseMethod):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.temperature_value = {'ViT': 1.16, 'RN': 1.15} #for temperature scaling experiments 

    def test_time_tuning(self, inputs):
        if self.args.cocoop:
            image_feature, pgen_ctx = inputs
            pgen_ctx.requires_grad = True
            self.optimizer = torch.optim.AdamW([pgen_ctx], self.args.lr)

        if self.args.cocoop:
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
        if not self.args.cocoop: # no need to reset cocoop because it's fixed
            with torch.no_grad():
                self.model.reset()
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
            if not self.args.cocoop: # no need to reset cocoop because it's fixed
                if self.args.tta_steps > 0:
                    with torch.no_grad():
                        self.model.reset()
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
                if self.args.cocoop:
                    image_feature = image_feature[0].unsqueeze(0)
            
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda'):
                    if self.args.cocoop:
                        output = self.model((image_feature, pgen_ctx))
                    else:
                        output = self.model(image)

            softmax_output = softmax(output)   

            if result_dict is not None:
                #maximum confidence of the softmax_output and its index
                max_confidence, max_index = torch.max(softmax_output, 1)

                #save the max confidence, prediction, and label to the result_dict
                result_dict['max_confidence'].append(max_confidence.item())
                result_dict['prediction'].append(max_index.item())
                result_dict['label'].append(target.item())

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