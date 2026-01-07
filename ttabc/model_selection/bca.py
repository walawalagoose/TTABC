
"""
Bayesian Class Adaptation (BCA)

Official idea: Bayesian Test-Time Adaptation for Vision-Language Models (CVPR 2025).
This TTABC integration is training-free (no backprop) and maintains:
- class centers/prototypes mu (likelihood adaptation)
- a cluster-to-class probability matrix P(Y|mu) (prior adaptation)
"""

import time
from dataclasses import dataclass

import torch

from ttabc.model_selection.base_method import BaseMethod
from ttabc.utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, softmax_entropy


@dataclass(frozen=True)
class BCAConfig:
    thr1: float = 0.05
    init_count1: float = 20000.0
    thr2: float = 0.65
    init_count2: float = 1.0
    tem: float = 100.0


class BCAState(torch.nn.Module):
    def __init__(
        self,
        init_centers: torch.Tensor,  # (K,d), normalized
        cfg: BCAConfig,
    ):
        super().__init__()
        if init_centers.dim() != 2:
            raise ValueError(f"init_centers must be (K,d), got {tuple(init_centers.shape)}")
        K, d = init_centers.shape
        self.K = int(K)
        self.d = int(d)
        self.cfg = cfg

        # Use float32 for numerical stability (K up to 1000 in ImageNet).
        self.register_buffer("mu", init_centers.to(dtype=torch.float32).clone())
        self.register_buffer("cluster_to_class_prob", torch.eye(self.K, dtype=torch.float32, device=init_centers.device))

        self.register_buffer("c1", torch.full((self.K,), float(cfg.init_count1), dtype=torch.float32, device=init_centers.device))
        self.register_buffer("c2", torch.full((self.K,), float(cfg.init_count2), dtype=torch.float32, device=init_centers.device))

    def _predict_probs(self, features: torch.Tensor) -> torch.Tensor:
        # features: (N,d) normalized
        # P(u|x) ~ softmax(tem * <x, mu>)
        logits_um = float(self.cfg.tem) * (features @ self.mu.t())  # (N,K)
        s1 = logits_um.softmax(dim=1)  # (N,K)
        probs = s1 @ self.cluster_to_class_prob  # (N,K)
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-12)
        return probs

    @torch.no_grad()
    def step(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Online prediction + conditional updates.
        Returns:
          - probs: (N,K)
          - pred:  (N,)
        """
        if features.dim() != 2 or features.size(1) != self.d:
            raise ValueError(f"features must be (N,{self.d}), got {tuple(features.shape)}")

        probs = self._predict_probs(features)  # (N,K)
        prob_max, pred = probs.max(dim=1)  # (N,), (N,)

        for i in range(features.size(0)):
            k = int(pred[i].item())

            if prob_max[i] > float(self.cfg.thr1):
                # mu_k <- (c1*mu_k + x) / (c1+1), then normalize
                self.mu[k] = (self.c1[k] * self.mu[k] + features[i]) / (self.c1[k] + 1.0)
                self.c1[k] = self.c1[k] + 1.0
                self.mu[k] = self.mu[k] / (self.mu[k].norm() + 1e-12)

            if prob_max[i] > float(self.cfg.thr2):
                # P(Y|mu_k) <- (c2*old + probs_i) / (c2+1)
                self.cluster_to_class_prob[k] = (
                    self.c2[k] * self.cluster_to_class_prob[k] + probs[i]
                ) / (self.c2[k] + 1.0)
                self.c2[k] = self.c2[k] + 1.0
                self.cluster_to_class_prob[k] = self.cluster_to_class_prob[k] / (
                    self.cluster_to_class_prob[k].sum() + 1e-12
                )

        return probs, pred


class BCA(BaseMethod):
    """
    Training-free BCA in TTABC's method interface.

    NOTE: This method requires access to image features, so it currently supports prompt_type=no_prompt.
    """

    def __init__(self, args):
        super().__init__(args)
        self.args = args

        default_cfg = BCAConfig()
        user_cfg = getattr(args, "bca_config", None) or {}
        self.bca_config = BCAConfig(**{**default_cfg.__dict__, **user_cfg})

    def test_time_tuning(self, inputs):
        # BCA performs no parameter updates
        return None

    def _get_text_features(self) -> torch.Tensor:
        if hasattr(self.model, "get_text_features"):
            return self.model.get_text_features()
        if hasattr(self.model, "text_features"):
            return self.model.text_features
        raise AttributeError("Current CLIP wrapper does not expose text features; use prompt_type=no_prompt.")

    def test_time_adapt_eval(self, val_loader, result_dict=None):
        batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
        top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
        top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
        progress = ProgressMeter(len(val_loader), [batch_time, top1, top5], prefix="Test: ")

        self.model.eval()
        with torch.no_grad():
            self.model.reset()

        # (re-)init per dataset (classnames may change via reset_classnames())
        with torch.no_grad():
            text_features = self._get_text_features().to(torch.float32)  # (K,d), normalized
        bca_state = BCAState(text_features, self.bca_config).cuda(self.args.gpu)

        end = time.time()

        for i, (images, target) in enumerate(val_loader):
            assert self.args.gpu is not None

            if isinstance(images, list):
                for k in range(len(images)):
                    images[k] = images[k].cuda(self.args.gpu, non_blocking=True)
                if getattr(self.args, "tpt", False):
                    images_in = torch.cat(images, dim=0)  # (V,C,H,W)
                else:
                    images_in = images[0]
                meter_image = images[0]
            else:
                if len(images.size()) > 4:
                    assert images.size()[0] == 1
                    images = images.squeeze(0)
                images_in = images.cuda(self.args.gpu, non_blocking=True)
                meter_image = images_in

            target = target.cuda(self.args.gpu, non_blocking=True)

            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda"):
                    logits, feats = self.model(images_in)  # (N,K), (N,d)

            # For multi-view inference, select low-entropy views (as in official BCA)
            if feats.size(0) > 1 and getattr(self.args, "tpt", False):
                logits_um = float(bca_state.cfg.tem) * (feats.to(torch.float32) @ bca_state.mu.t())  # (V,K)
                view_entropy = softmax_entropy(logits_um)
                k = max(1, int(view_entropy.size(0) * float(getattr(self.args, "selection_p", 0.1))))
                selected_idx = torch.argsort(view_entropy, descending=False)[:k]
                feats = feats[selected_idx].mean(0, keepdim=True)

            # Online update (streaming). If batch dimension > 1, process sequentially.
            feats = feats.to(torch.float32)
            feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-12)

            if feats.size(0) == 1:
                probs, pred = bca_state.step(feats)
                out_logits = (probs + 1e-12).log()
            else:
                all_probs = []
                all_pred = []
                for j in range(feats.size(0)):
                    probs_j, pred_j = bca_state.step(feats[j : j + 1])
                    all_probs.append(probs_j)
                    all_pred.append(pred_j)
                probs = torch.cat(all_probs, dim=0)
                pred = torch.cat(all_pred, dim=0)
                out_logits = (probs + 1e-12).log()

            if result_dict is not None:
                max_confidence, max_index = probs.max(dim=1)
                if probs.size(0) == 1:
                    result_dict["max_confidence"].append(float(max_confidence.item()))
                    result_dict["prediction"].append(int(max_index.item()))
                    result_dict["label"].append(int(target.view(-1)[0].item()))
                else:
                    for j in range(probs.size(0)):
                        result_dict["max_confidence"].append(float(max_confidence[j].item()))
                        result_dict["prediction"].append(int(max_index[j].item()))
                        result_dict["label"].append(int(target.view(-1)[j].item()))

            acc1, acc5 = accuracy(out_logits, target, topk=(1, 5))
            top1.update(acc1[0], meter_image.size(0))
            top5.update(acc5[0], meter_image.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.args.print_freq == 0:
                progress.display(i)

        progress.display_summary()
        return [top1.avg, top5.avg]