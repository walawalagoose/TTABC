"""
    MOON: Von Mises-Fisher Mixture Model with Dynamic Shrinkage
    for Realistic Test-Time Transduction,
    https://arxiv.org/abs/2607.15851
    https://github.com/walawalagoose/MOON
"""

import math
import time

import torch
import torch.nn.functional as F

from ttabc.model_selection.base_method import BaseMethod
from ttabc.utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, resize_with_CLIP


def _inverse_resultant_length(r, dim):
    r = r.clamp(min=0.0, max=0.999)
    return (r * dim - r.pow(3)) / (1.0 - r.square())


def _build_affinity_matrix(features, neighbors):
    size = features.size(0)
    affinity = features @ features.t()
    # Match upstream CPU top-k tie-breaking for duplicate image features.
    columns = affinity.cpu().topk(neighbors + 1, dim=-1).indices[:, 1:].to(features.device).flatten()
    rows = torch.arange(size, device=features.device).repeat_interleave(neighbors)
    return torch.sparse_coo_tensor(
        torch.stack([rows, columns]), affinity[rows, columns],
        (size, size), device=features.device,
    )


@torch.no_grad()
def solve_moon(
    query_features,
    clip_logits,
    text_features,
    alpha=1.0,
    soft_beta=False,
    lambda_y_hat=1.0,
    lambda_laplacian=1.0,
    n_neighbors=3,
    max_iter=10,
):
    """Batch-local MOON; use TTABC's CLIP priors, never ground-truth labels.

    Matches the official batch solver (hard beta by default, one assignment
    update per iteration, normalized class weights and kappa capped at 500).
    """
    query_features = F.normalize(query_features.float(), dim=-1)
    text_features = F.normalize(text_features.float(), dim=-1)
    clip_logits = clip_logits.float()
    if query_features.ndim != 2 or text_features.ndim != 2:
        raise ValueError("MOON expects image [N, D] and text [C, D] features")
    num_samples, dim = query_features.shape
    num_classes = text_features.size(0)
    if clip_logits.shape != (num_samples, num_classes) or text_features.size(1) != dim:
        raise ValueError("MOON feature and logit shapes do not match")
    if alpha < 0 or n_neighbors < 0 or max_iter < 0:
        raise ValueError("MOON alpha, n_neighbors and max_iter must be non-negative")
    y_hat = clip_logits.softmax(dim=1)
    if num_samples <= 1 or num_classes <= 1:
        return y_hat

    z = y_hat.clone()
    mu = text_features.clone()
    # Initial concentration from the weighted distance to each text anchor.
    weighted_distance = torch.zeros(num_classes, device=z.device)
    for start in range(0, num_samples, 2500):
        diff = query_features[start:start + 2500, None, :] - text_features[None, :, :]
        weighted_distance += (z[start:start + 2500] * diff.square().sum(dim=-1)).sum(dim=0)
    r = 1.0 - weighted_distance / (z.sum(dim=0) + 1e-10) / 2.0
    kappa = _inverse_resultant_length(r, dim)

    neighbors = min(int(n_neighbors), num_samples - 1)
    affinity = _build_affinity_matrix(query_features, neighbors)
    class_weights = (y_hat.mean(dim=0) * y_hat.max(dim=0).values).sqrt()
    class_weights = class_weights / class_weights.max().clamp_min(1e-12)
    alpha_k = alpha / (class_weights + 1e-12)
    prior = y_hat.pow(lambda_y_hat)

    for step in range(int(max_iter) + 1):
        log_likelihood = (query_features @ mu.t()) * kappa.unsqueeze(0)
        log_likelihood += ((dim - 1) / 2.0 * kappa.clamp_min(1e-6).log() - kappa).unsqueeze(0)
        if neighbors > 0 and lambda_laplacian != 0:
            log_likelihood += lambda_laplacian * (50.0 / (neighbors * 2.0)) * (
                torch.sparse.mm(affinity.t(), z) + torch.sparse.mm(affinity, z)
            )
        log_likelihood -= log_likelihood.max(dim=1, keepdim=True).values
        assignments = prior * torch.exp(log_likelihood / 50.0)
        mass = assignments.sum(dim=1, keepdim=True)
        z = assignments / mass.clamp_min(torch.finfo(assignments.dtype).tiny)
        # The log-space equivalent handles extreme logits without 0/0.
        underflow = mass.squeeze(1) == 0
        if underflow.any():
            z[underflow] = (
                lambda_y_hat * clip_logits[underflow].log_softmax(dim=1)
                + log_likelihood[underflow] / 50.0
            ).softmax(dim=1)
        if step == int(max_iter):
            break

        entropy = -(z * (z + 1e-12).log()).sum(dim=1)
        sample_weights = (1.0 - entropy / math.log(num_classes)).clamp(0.0, 1.0)
        weighted_z = z * sample_weights.unsqueeze(1)
        if soft_beta:
            counts = weighted_z.sum(dim=0)
        else:
            counts = torch.bincount(weighted_z.argmax(dim=1), minlength=num_classes).to(z.dtype)
        beta = counts / (alpha_k + counts + 1e-12)
        empirical_mean = torch.einsum("ij,ik->jk", weighted_z, query_features)
        empirical_mean /= weighted_z.sum(dim=0).unsqueeze(1) + 1e-10
        resultant = beta.unsqueeze(1) * empirical_mean + (1.0 - beta.unsqueeze(1)) * text_features
        mu = F.normalize(resultant, dim=-1)
        kappa = _inverse_resultant_length(resultant.norm(dim=-1), dim).clamp(max=500.0)

    return z


class MOON(BaseMethod):
    """Training-free MOON batch adaptation for zero-shot CLIP features."""

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        assert self.args.prompt_type == "no_prompt", "MOON only supports prompt_type='no_prompt'"

    def test_time_tuning(self, inputs):
        return

    def _prepare_images(self, images):
        if isinstance(images, list):
            images = images[0]
        if len(images.size()) > 4:
            assert images.size(0) == 1
            images = images.squeeze(0)
        return resize_with_CLIP(images, self.args.resolution).cuda(self.args.gpu, non_blocking=True)

    def test_time_adapt_eval(self, val_loader, result_dict=None):
        batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
        top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
        top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
        progress = ProgressMeter(len(val_loader), [batch_time, top1, top5], prefix="MOON Test: ")

        self.model.eval()
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            assert self.args.gpu is not None
            image = self._prepare_images(images)
            target = target.cuda(self.args.gpu, non_blocking=True)

            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda"):
                    clip_logits, image_features = self.model(image)
                    text_features = self.model.get_text_features()
                output = solve_moon(
                    image_features, clip_logits, text_features,
                    alpha=self.args.moon_alpha,
                    soft_beta=self.args.moon_soft_beta,
                    lambda_y_hat=self.args.moon_lambda_y_hat,
                    lambda_laplacian=self.args.moon_lambda_laplacian,
                    n_neighbors=self.args.moon_n_neighbors,
                    max_iter=self.args.moon_max_iter,
                )

            if result_dict is not None:
                max_confidence, max_index = torch.max(output, dim=1)
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
