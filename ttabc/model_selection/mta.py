"""
    On the Test-Time Zero-Shot Generalization of Vision-Language Models:
    Do We Really Need Prompt Learning?
    https://arxiv.org/abs/2405.02266
"""

import time

import torch
import torch.nn.functional as F

from ttabc.model_selection.base_method import BaseMethod
from ttabc.utils.tools import Summary, AverageMeter, ProgressMeter, accuracy


def gaussian_kernel(mu, bandwidth, datapoints):
    dist = torch.norm(datapoints - mu, dim=-1, p=2)
    return torch.exp(-dist ** 2 / (2 * bandwidth ** 2))


def solve_mta(model, inputs, args):
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda"):
            if hasattr(model, "get_text_features"):
                logits, image_features = model(inputs)
                text_features = model.get_text_features()
            else:
                logits, image_features, text_features, _, _ = model(inputs, return_features=True)

    lambda_y = args.lambda_y
    lambda_q = args.lambda_q
    max_iter = 5
    temperature = 1.0
    batch_size = image_features.shape[0]

    dist = torch.cdist(image_features, image_features)
    sorted_dist, _ = torch.sort(dist, dim=1)
    k = max(1, int(0.3 * (batch_size - 1)))
    selected_distances = sorted_dist[:, 1 : k + 1] ** 2
    mean_distance = torch.mean(selected_distances, dim=1)
    bandwidth = torch.sqrt(0.5 * mean_distance).clamp_min(torch.finfo(image_features.dtype).eps)

    affinity_matrix = (logits / temperature).softmax(1) @ (logits / temperature).softmax(1).t()

    y = torch.ones(batch_size, device=image_features.device, dtype=image_features.dtype) / batch_size
    mode = image_features[0]
    threshold = 1e-6

    for _ in range(max_iter):
        density = gaussian_kernel(mode, bandwidth, image_features)

        for _ in range(max_iter):
            old_y = y
            weighted_affinity = affinity_matrix * y.unsqueeze(0)
            y = F.softmax(
                (density + lambda_q * torch.sum(weighted_affinity, dim=1)) / lambda_y,
                dim=-1,
            )
            if torch.norm(old_y - y) < threshold:
                break

        for _ in range(max_iter):
            old_mode = mode
            density = gaussian_kernel(mode, bandwidth, image_features)
            weighted_density = density * y
            mode = torch.sum(weighted_density.unsqueeze(1) * image_features, dim=0) / torch.sum(weighted_density)
            mode = mode / mode.norm(p=2, dim=-1).clamp_min(torch.finfo(mode.dtype).eps)
            if torch.norm(old_mode - mode) < threshold:
                break

    return mode.unsqueeze(0) @ text_features.t() * model.logit_scale.exp()


class MTA(BaseMethod):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        assert self.args.prompt_type == "no_prompt", "MTA only supports prompt_type='no_prompt'"

    def test_time_tuning(self, inputs):
        pass

    def test_time_adapt_eval(self, val_loader, result_dict=None):
        batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
        top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
        top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, top1, top5],
            prefix="Test: ",
        )

        self.model.eval()
        end = time.time()

        for i, (images, target) in enumerate(val_loader):
            assert self.args.gpu is not None
            if isinstance(images, list):
                for k in range(len(images)):
                    images[k] = images[k].cuda(self.args.gpu, non_blocking=True)
                image = images[0]
                images = torch.cat(images, dim=0)
            else:
                if len(images.size()) > 4:
                    assert images.size()[0] == 1
                    images = images.squeeze(0)
                images = images.cuda(self.args.gpu, non_blocking=True)
                image = images

            target = target.cuda(self.args.gpu, non_blocking=True)

            with torch.no_grad():
                output = solve_mta(self.model, images, self.args)

            if result_dict is not None:
                softmax_output = output.softmax(dim=1)
                max_confidence, max_index = torch.max(softmax_output, 1)

                if max_confidence.numel() == 1:
                    result_dict["max_confidence"].append(max_confidence.item())
                    result_dict["prediction"].append(max_index.item())
                    result_dict["label"].append(target.item())
                else:
                    for j in range(max_confidence.size(0)):
                        result_dict["max_confidence"].append(max_confidence[j].item())
                        result_dict["prediction"].append(max_index[j].item())
                        result_dict["label"].append(target[j].item())

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % self.args.print_freq == 0:
                progress.display(i)

        progress.display_summary()
        return [top1.avg, top5.avg]