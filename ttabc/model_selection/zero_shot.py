import time
import torch
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from ttabc.utils.tools import Summary, AverageMeter, ProgressMeter, accuracy
from ttabc.model_selection.base_method import BaseMethod


class ZEROSHOT(BaseMethod):
    """
    Pure Zero-shot CLIP evaluation without any test-time adaptation.
    This serves as the baseline for comparison with TTA methods.
    """
    
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        
    def test_time_tuning(self, inputs):
        """
        Zero-shot method does not perform any test-time tuning.
        """
        return
    
    def test_time_adapt_eval(self, val_loader, result_dict=None):
        """
        Evaluate the model using pure zero-shot inference.
        No adaptation, no optimization, just direct inference.
        """
        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, top1, top5],
            prefix='Zero-shot Test: ')

        # Set model to evaluation mode
        self.model.eval()
        
        # Define softmax layer for confidence calculation
        softmax = torch.nn.Softmax(dim=1)

        end = time.time()
        
        with torch.no_grad():  # No gradients needed for zero-shot evaluation
            for i, (images, target) in enumerate(val_loader):
                # Move data to GPU
                if isinstance(images, list):
                    for k in range(len(images)):
                        images[k] = images[k].cuda(self.args.gpu, non_blocking=True)
                    image = images[0]  # Use first image for inference
                else:
                    if len(images.size()) > 4:
                        # Handle ImageNet Sampler format
                        assert images.size()[0] == 1
                        images = images.squeeze(0)
                    images = images.cuda(self.args.gpu, non_blocking=True)
                    image = images
                
                target = target.cuda(self.args.gpu, non_blocking=True)
                
                # Reset model to initial state (important for consistency)
                self.model.reset()
                
                # Direct inference without any adaptation
                with torch.amp.autocast(device_type='cuda'):
                    output, _ = self.model(image)
                
                # Calculate softmax for confidence
                softmax_output = softmax(output)
                
                # Save results if result_dict is provided (for ECE calculation)
                if result_dict is not None:
                    max_confidence, max_index = torch.max(softmax_output, 1)
                    # Handle batch of samples
                    for j in range(max_confidence.size(0)):
                        result_dict['max_confidence'].append(max_confidence[j].item())
                        result_dict['prediction'].append(max_index[j].item())
                        result_dict['label'].append(target[j].item())
                    # result_dict['max_confidence'].append(max_confidence.item())
                    # result_dict['prediction'].append(max_index.item())
                    # result_dict['label'].append(target.item())
                
                # Calculate accuracy
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], image.size(0))
                top5.update(acc5[0], image.size(0))
                
                # Measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                
                # Print progress
                if (i + 1) % self.args.print_freq == 0:
                    progress.display(i)
        
        progress.display_summary()
        return [top1.avg, top5.avg]