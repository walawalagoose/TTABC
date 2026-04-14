"""
    R-TPT: Improving Adversarial Robustness of Vision-Language Models
    through Test-Time Prompt Tuning
    https://arxiv.org/abs/2504.11195
"""

import time

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

from ttabc.model_selection.base_method import BaseMethod
from ttabc.utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, select_confident_samples


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def get_top_sim(sim_matrix):
    k = min(20, max(1, sim_matrix.size(-1) - 1))
    sim_matrix = sim_matrix.clone()
    sim_matrix[sim_matrix >= 1.0] = float("-inf")
    top_k_values, _ = sim_matrix.topk(k, dim=-1)
    return top_k_values.mean(dim=-1)


def entropy_avg(outputs):
    batch_entropy = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
    return batch_entropy.mean()


def _stats_like(images):
    mean = torch.tensor(CLIP_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return mean, std


def denormalize(images):
    mean, std = _stats_like(images)
    return images * std + mean


def normalize(images):
    mean, std = _stats_like(images)
    return (images - mean) / std


def _forward_features(model, images):
    image_features = model.image_encoder(images.type(model.dtype))
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = model.get_text_features()
    logit_scale = model.logit_scale.exp()
    return image_features, text_features, logit_scale


class _RawImageModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, raw_images):
        return self.model(normalize(raw_images))


def pgd_attack(model, images, labels, eps, alpha, steps):
    adv_images = images.detach().clone()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
    adv_images = adv_images.clamp(0.0, 1.0).detach()

    for _ in range(steps):
        adv_images.requires_grad_(True)
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = (images + delta).clamp(0.0, 1.0).detach()

    return adv_images


class RTPT(BaseMethod):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        assert self.args.prompt_type == "coop", "R-TPT only supports prompt_type='coop'"

        if getattr(self.args, "load_tecoa", ""):
            robust_pretrain_path = {
                "RN50-eps1": "pretrain/tecoa/rn50_eps1.pth.tar",
            }.get(self.args.load_tecoa)
            if robust_pretrain_path is None:
                raise ValueError(f"Unsupported TeCoA checkpoint: {self.args.load_tecoa}")
            robust_state_dict = torch.load(robust_pretrain_path, map_location="cpu", weights_only=False)
            self.model.image_encoder.load_state_dict(robust_state_dict["vision_encoder_state_dict"])

        self.to_pil = ToPILImage()
        self.attack_model = _RawImageModelWrapper(self.model)

    def test_time_tuning(self, inputs):
        selected_idx = None
        for _ in range(self.args.tta_steps):
            with torch.amp.autocast(device_type="cuda"):
                output = self.model(inputs)
                if selected_idx is not None:
                    output = output[selected_idx]
                else:
                    output, selected_idx = select_confident_samples(output, self.args.selection_p)
                loss = entropy_avg(output)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

    def _prepare_inputs(self, images):
        if isinstance(images, list):
            for idx in range(len(images)):
                images[idx] = images[idx].cuda(self.args.gpu, non_blocking=True)
            image = images[0]
            stacked = torch.cat(images, dim=0)
            return image, stacked

        if len(images.size()) > 4:
            assert images.size()[0] == 1
            images = images.squeeze(0)
        images = images.cuda(self.args.gpu, non_blocking=True)
        return images, images

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
        self.attack_model.eval()
        end = time.time()

        attack_eps = float(getattr(self.args, "rtpt_eps", 0.0))
        attack_steps = int(getattr(self.args, "rtpt_steps", 0))
        attack_alpha = attack_eps / 4.0

        for i, (images, target) in enumerate(val_loader):
            assert self.args.gpu is not None
            target = target.cuda(self.args.gpu, non_blocking=True)

            if attack_eps > 0.0:
                if attack_steps <= 0:
                    raise ValueError("R-TPT requires rtpt_steps > 0 when rtpt_eps > 0.")
                raw_image = denormalize(images[0].cuda(self.args.gpu, non_blocking=True))
                adv_image = pgd_attack(
                    self.attack_model,
                    raw_image,
                    target,
                    eps=attack_eps / 255.0,
                    alpha=attack_alpha / 255.0,
                    steps=attack_steps,
                )
                transform = getattr(val_loader.dataset, "transform", None)
                if transform is None:
                    raise RuntimeError("R-TPT expects the dataset to expose a callable transform.")
                adv_pil = self.to_pil(adv_image.squeeze(0).detach().cpu())
                images = transform(adv_pil)
                images = [img.unsqueeze(0) for img in images]

            image, images = self._prepare_inputs(images)

            with torch.no_grad():
                self.model.reset()
            self.optimizer.load_state_dict(self.optim_state)

            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda"):
                    _ = self.model(image)
                    clip_features, _, _ = _forward_features(self.model, images)

            self.test_time_tuning(images)

            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda"):
                    tuned_outputs = self.model(images)

            sim_matrix_images = torch.bmm(
                clip_features.unsqueeze(0),
                clip_features.unsqueeze(0).transpose(1, 2),
            )
            score = get_top_sim(sim_matrix_images)
            weight = F.softmax(score / 0.01, dim=-1).to(dtype=tuned_outputs.dtype)
            output = torch.bmm(
                weight.unsqueeze(-1).transpose(1, 2),
                tuned_outputs.unsqueeze(0),
            ).squeeze(1)

            if result_dict is not None:
                softmax_output = output.softmax(dim=1)
                max_confidence, max_index = torch.max(softmax_output, 1)
                result_dict["max_confidence"].append(max_confidence.item())
                result_dict["prediction"].append(max_index.item())
                result_dict["label"].append(target.item())

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % self.args.print_freq == 0:
                progress.display(i)

        progress.display_summary()
        return [top1.avg, top5.avg]