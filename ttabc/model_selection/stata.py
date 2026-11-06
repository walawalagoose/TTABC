"""
    StatA: Realistic Test-Time Adaptation of Vision-Language Models,
    https://arxiv.org/abs/2501.03729
    https://github.com/MaxZanella/StatA
"""

import time

import torch

from ttabc.model_selection.base_method import BaseMethod
from ttabc.utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, resize_with_CLIP


def _l2_normalize(x, eps=1e-6):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def _build_affinity_matrix(query_features, n_neighbors):
    num_samples = query_features.size(0)
    if num_samples <= 1 or n_neighbors <= 0:
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long, device=query_features.device),
            torch.empty((0,), dtype=query_features.dtype, device=query_features.device),
            (num_samples, num_samples),
            device=query_features.device,
        )

    n_neighbors = min(int(n_neighbors), num_samples - 1)
    affinity = query_features @ query_features.t()
    knn_index = affinity.topk(n_neighbors + 1, dim=-1, largest=True).indices[:, 1:]

    row_indices = torch.arange(num_samples, device=query_features.device).unsqueeze(1)
    row_indices = row_indices.repeat(1, n_neighbors).flatten()
    col_indices = knn_index.flatten()
    values = affinity[row_indices, col_indices]

    return torch.sparse_coo_tensor(
        torch.stack([row_indices, col_indices]),
        values,
        (num_samples, num_samples),
        device=query_features.device,
    )


class _DiagonalGaussian:
    def __init__(self, mu, cov):
        self.mu = mu.clone()
        self.cov = cov.clone()

    def log_likelihood(self, x, chunk_size=2500):
        num_samples = x.size(0)
        num_classes = self.mu.size(0)
        likelihoods = torch.empty(
            (num_samples, num_classes),
            dtype=x.dtype,
            device=x.device,
        )

        inv_cov = 1.0 / self.cov.clamp_min(torch.finfo(x.dtype).eps)
        for start_idx in range(0, num_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, num_samples)
            diff = x[start_idx:end_idx, None, :] - self.mu[None, :, 0, :]
            likelihoods[start_idx:end_idx] = -0.5 * (diff.square() * inv_cov[None, :, :]).sum(dim=2)

        return likelihoods

    def set_mu(self, mu):
        self.mu = mu

    def set_cov(self, cov):
        self.cov = cov


def _init_cov(text_prototypes, query_features, z):
    num_samples = z.size(0)
    cov = torch.zeros(
        text_prototypes.size(-1),
        dtype=query_features.dtype,
        device=query_features.device,
    )

    chunk_size = 2500
    for start_idx in range(0, num_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, num_samples)
        query_chunk = query_features[start_idx:end_idx]
        cov += torch.einsum(
            "ij,ijk->k",
            z[start_idx:end_idx],
            (query_chunk[:, None, :] - text_prototypes[None, :, 0, :]).square(),
        )

    cov = cov / max(num_samples, 1)
    return cov.clamp_min(torch.finfo(query_features.dtype).eps)


def _update_z(likelihoods, y_hat, z, affinity, lambda_y_hat, lambda_laplacian, n_neighbors, sigma, max_iter=5):
    effective_neighbors = min(int(n_neighbors), max(z.size(0) - 1, 0))
    graph_weight = 0.0
    if effective_neighbors > 0:
        graph_weight = lambda_laplacian * (50.0 / (effective_neighbors * 2.0))

    sigma_log_sum = sigma.clamp_min(torch.finfo(sigma.dtype).eps).log().sum(dim=1)
    y_hat_term = y_hat.clamp_min(torch.finfo(y_hat.dtype).eps).pow(lambda_y_hat)

    for _ in range(max_iter):
        intermediate = likelihoods.clone()
        if graph_weight:
            intermediate += graph_weight * (torch.sparse.mm(affinity.t(), z) + torch.sparse.mm(affinity, z))
        intermediate -= 0.5 * sigma_log_sum.unsqueeze(0)
        intermediate -= intermediate.max(dim=1, keepdim=True)[0]
        intermediate = y_hat_term * torch.exp(intermediate / 50.0)
        z = intermediate / intermediate.sum(dim=1, keepdim=True).clamp_min(torch.finfo(intermediate.dtype).eps)

    return z


def _update_beta(z, alpha, soft=False):
    if soft:
        sum_z = z.sum(dim=0)
    else:
        predicted_classes = z.argmax(dim=1)
        sum_z = torch.bincount(predicted_classes, minlength=z.size(1)).to(device=z.device, dtype=z.dtype)
    return sum_z / (float(alpha) + sum_z + 1e-12)


def _update_mu(query_features, z, beta, text_prototypes):
    class_mass = z.sum(dim=0).clamp_min(torch.finfo(query_features.dtype).eps)
    mu = torch.einsum("ij,ik->jk", z, query_features) / class_mass.unsqueeze(-1)
    mu = _l2_normalize(mu.unsqueeze(1))

    beta = beta.unsqueeze(-1).unsqueeze(-1)
    mu = beta * mu + (1.0 - beta) * text_prototypes
    return _l2_normalize(mu)


def _update_cov(adapter, query_features, z, beta, text_prototypes, init_covariance):
    class_mass = z.sum(dim=0).clamp_min(torch.finfo(query_features.dtype).eps)
    cov = torch.zeros_like(adapter.cov)

    chunk_size = 2500
    for start_idx in range(0, query_features.size(0), chunk_size):
        end_idx = min(start_idx + chunk_size, query_features.size(0))
        query_chunk = query_features[start_idx:end_idx]
        diff = query_chunk[:, None, :] - adapter.mu[None, :, 0, :]
        cov += (diff.square() * z[start_idx:end_idx, :, None]).sum(dim=0)
    cov = cov / class_mass.unsqueeze(-1)

    delta_mu = (text_prototypes - adapter.mu).squeeze(1)
    anchor_cov = init_covariance + delta_mu.square()

    beta = beta.unsqueeze(-1)
    cov = beta * cov + (1.0 - beta) * anchor_cov
    return cov.clamp_min(torch.finfo(query_features.dtype).eps)


@torch.no_grad()
def solve_stata(
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
    query_features = _l2_normalize(query_features.float())
    text_features = _l2_normalize(text_features.float())
    clip_logits = clip_logits.float()

    if query_features.size(0) <= 1:
        return clip_logits.softmax(dim=1)

    y_hat = clip_logits.softmax(dim=1)
    z = y_hat.clone()

    text_prototypes = text_features.unsqueeze(1)
    cov = _init_cov(text_prototypes, query_features, z)
    cov = cov.unsqueeze(0).repeat(text_features.size(0), 1)
    init_covariance = cov.clone()

    adapter = _DiagonalGaussian(mu=text_prototypes, cov=cov)
    affinity = _build_affinity_matrix(query_features, n_neighbors)

    for step_idx in range(int(max_iter) + 1):
        likelihoods = adapter.log_likelihood(query_features)
        z = _update_z(
            likelihoods,
            y_hat,
            z,
            affinity,
            lambda_y_hat,
            lambda_laplacian,
            n_neighbors,
            adapter.cov,
        )

        if step_idx == int(max_iter):
            break

        beta = _update_beta(z, alpha=alpha, soft=soft_beta)
        mu = _update_mu(query_features, z, beta, text_prototypes)
        adapter.set_mu(mu)
        cov = _update_cov(adapter, query_features, z, beta, text_prototypes, init_covariance)
        adapter.set_cov(cov)

    return z


class StatA(BaseMethod):
    """Training-free StatA batch adaptation for zero-shot CLIP features."""

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        assert self.args.prompt_type == "no_prompt", "StatA only supports prompt_type='no_prompt'"

    def test_time_tuning(self, inputs):
        return

    def _prepare_images(self, images):
        if isinstance(images, list):
            for idx in range(len(images)):
                images[idx] = resize_with_CLIP(images[idx], self.args.resolution).cuda(
                    self.args.gpu, non_blocking=True
                )
            return images[0]

        if len(images.size()) > 4:
            assert images.size(0) == 1
            images = images.squeeze(0)
        return resize_with_CLIP(images, self.args.resolution).cuda(self.args.gpu, non_blocking=True)

    def test_time_adapt_eval(self, val_loader, result_dict=None):
        batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
        top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
        top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
        progress = ProgressMeter(len(val_loader), [batch_time, top1, top5], prefix="StatA Test: ")

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

                output = solve_stata(
                    image_features,
                    clip_logits,
                    text_features,
                    alpha=self.args.stata_alpha,
                    soft_beta=self.args.stata_soft_beta,
                    lambda_y_hat=self.args.stata_lambda_y_hat,
                    lambda_laplacian=self.args.stata_lambda_laplacian,
                    n_neighbors=self.args.stata_n_neighbors,
                    max_iter=self.args.stata_max_iter,
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
