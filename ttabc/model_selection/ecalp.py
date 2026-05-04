"""
    Efficient and Context-Aware Label Propagation for Zero-/Few-Shot Training-Free
    Adaptation of Vision-Language Model,
    https://arxiv.org/abs/2412.18303,
    https://github.com/yushuli-bupt/ECALP
"""

import time

import torch
import torch.nn.functional as F

from ttabc.model_selection.base_method import BaseMethod
from ttabc.utils.tools import Summary, AverageMeter, ProgressMeter, accuracy


def _combine_knns(knn_im2im, sim_im2im, knn_im2text, sim_im2text, num_classes):
    knn_im = knn_im2im + num_classes
    knn = torch.cat((knn_im, knn_im2text), dim=1)
    sim = torch.cat((sim_im2im, sim_im2text), dim=1)
    return knn, sim


def _sparse_slice(sparse_matrix, row_start, row_end):
    sparse_matrix = sparse_matrix.coalesce()
    indices = sparse_matrix.indices()
    values = sparse_matrix.values()

    mask = (indices[0] >= row_start) & (indices[0] < row_end)
    new_indices = indices[:, mask].clone()
    new_values = values[mask]
    new_indices[0] = new_indices[0] - row_start

    new_size = (row_end - row_start, sparse_matrix.size(1))
    return torch.sparse_coo_tensor(new_indices, new_values, new_size, device=values.device)


def _edge_reweighting(X, Q, anchor, k, return_full=False, force_anchor_only=False):
    if (X.shape != Q.shape) and (not force_anchor_only):
        weights = 1 / (X.view(1, -1, anchor.shape[1]).var(dim=1).mean(dim=0) + 1e-6)
        Q_weighted = Q * weights
        X_weighted = X
    else:
        weights = anchor.var(dim=0)
        weights = F.normalize(weights, p=2, dim=0)
        Q_weighted = Q * weights
        X_weighted = X

    Q_weighted = F.normalize(Q_weighted, p=2, dim=1)
    X_weighted = F.normalize(X_weighted, p=2, dim=1)
    similarity = Q_weighted @ X_weighted.t()

    if return_full:
        return similarity

    sim, knn = similarity.topk(k, largest=True, dim=1)
    return knn, sim


def _knn_to_weighted_adj_matrix(knn, sim, num_nodes):
    knn_indices = knn.flatten()
    distances = sim.flatten()
    knn_indices = knn_indices.clone()
    knn_indices[knn_indices == -1] = 0

    row_indices = torch.arange(num_nodes, device=knn.device).repeat_interleave(knn.shape[1])
    nonzero_mask = distances != 0

    return torch.sparse_coo_tensor(
        torch.stack([row_indices[nonzero_mask], knn_indices[nonzero_mask]]),
        distances[nonzero_mask],
        (num_nodes, num_nodes),
        device=sim.device,
    )


class _StreamingECALPState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sim_im2im = None
        self.knn_im2im = None
        self.sim_im2text = None
        self.knn_im2text = None
        self.sim_im2im_store = None
        self.knn_im2im_store = None
        self.Y_0 = None
        self.Y_txt = None

    def _dynamic_graph_expansion(self, features, text_features, k_text, k_image):
        num_classes = text_features.shape[0]
        num_features = features.shape[0]
        first_time = (self.sim_im2im is None) or (num_features == 2)
        k_im2im = min(k_image, num_features)

        if first_time:
            sim_full = _edge_reweighting(
                features, features, text_features, k=k_im2im, return_full=True, force_anchor_only=True
            )
            self.sim_im2im, self.knn_im2im = sim_full[:num_features, :num_features].topk(
                k_im2im, largest=True, dim=1
            )
            self.sim_im2im_store = self.sim_im2im.clone()
            self.knn_im2im_store = self.knn_im2im.clone()
        else:
            sim_full_cr = _edge_reweighting(
                features[-2:], features, text_features, k=k_im2im, return_full=True, force_anchor_only=True
            ).t()

            sim_new, knn_new = sim_full_cr.topk(k_im2im, largest=True, dim=1)
            knn_new_idx = torch.full(
                (self.knn_im2im.shape[0] - 1, 1),
                num_features - 1,
                dtype=knn_new.dtype,
                device=knn_new.device,
            )

            last_mask = self.knn_im2im_store == num_features - 2
            self.sim_im2im_store[last_mask] = -1.0

            sim_extended = torch.cat(
                [self.sim_im2im_store[:-1], sim_full_cr[:, : num_features - 2].transpose(1, 0)],
                dim=1,
            )
            knn_extended = torch.cat([self.knn_im2im_store[:-1], knn_new_idx - 1, knn_new_idx], dim=1)

            if sim_extended.shape[0] == k_im2im + 1:
                k_im_new = k_im2im + 1
            elif sim_extended.shape[0] > k_im2im:
                k_im_new = k_im2im + 2
            else:
                k_im_new = k_im2im

            self.sim_im2im_store, ht = sim_extended.topk(k_im_new, largest=True, dim=1)
            self.sim_im2im = self.sim_im2im_store[:, :k_im2im]
            self.knn_im2im_store = knn_extended.gather(1, ht)
            self.knn_im2im = self.knn_im2im_store[:, :k_im2im]

            if sim_extended.shape[0] > k_im2im:
                sim_new_more, knn_new_more = sim_full_cr.topk(k_im_new, largest=True, dim=1)
                self.sim_im2im_store = torch.cat([self.sim_im2im_store, sim_new_more], dim=0)
                self.knn_im2im_store = torch.cat([self.knn_im2im_store, knn_new_more], dim=0)
            else:
                self.sim_im2im_store = torch.cat([self.sim_im2im_store, sim_new], dim=0)
                self.knn_im2im_store = torch.cat([self.knn_im2im_store, knn_new], dim=0)

            self.sim_im2im = torch.cat([self.sim_im2im, sim_new], dim=0)
            self.knn_im2im = torch.cat([self.knn_im2im, knn_new], dim=0)

        k_im2text = min(k_text, num_classes)
        if first_time:
            sim_full_text = features.mm(text_features.t())
            self.sim_im2text, self.knn_im2text = sim_full_text[:num_features].topk(k_im2text, largest=True, dim=1)
        else:
            sim_full_text_cr = features[-1:].mm(text_features.t())
            sim_new, knn_new = sim_full_text_cr.topk(k_im2text, largest=True, dim=1)
            self.sim_im2text = torch.cat((self.sim_im2text[:-1], sim_new, sim_new), dim=0)
            self.knn_im2text = torch.cat((self.knn_im2text[:-1], knn_new, knn_new), dim=0)

        knn, sim = _combine_knns(self.knn_im2im, self.sim_im2im, self.knn_im2text, self.sim_im2text, num_classes)
        knn_text = -1 * torch.ones((num_classes, knn.shape[1]), dtype=knn.dtype, device=knn.device)
        sim_text = torch.zeros((num_classes, sim.shape[1]), dtype=sim.dtype, device=sim.device)
        knn = torch.cat((knn_text, knn), dim=0)
        sim = torch.cat((sim_text, sim), dim=0)
        return knn, sim

    def predict(self, image_features, text_features, k_text, k_image, gamma, alpha, beta, max_iter):
        if image_features.dim() == 3:
            image_features = image_features.mean(dim=1)

        num_classes = text_features.shape[0]
        if image_features.shape[0] == 2:
            self.Y_0 = None
            self.Y_txt = None

        knn, sim = self._dynamic_graph_expansion(image_features, text_features, k_text, k_image)
        mask_knn = knn < num_classes
        sim[mask_knn] = sim[mask_knn] ** gamma
        mask_im2im = knn >= num_classes
        sim[mask_im2im] = sim[mask_im2im] ** gamma

        num_nodes = image_features.shape[0] + num_classes
        W = _knn_to_weighted_adj_matrix(knn, sim, num_nodes)
        W = W.transpose(0, 1) + W

        W_sum = torch.sparse.sum(W, dim=1).to_dense()
        diag = 1 / (torch.sqrt(W_sum) + 1e-6)
        diag = diag.to(W.device)
        indices = torch.arange(W.size(0), device=W.device)
        D_rootsquare = torch.sparse_coo_tensor(
            torch.stack([indices, indices]),
            diag,
            (W.size(0), W.size(0)),
            device=W.device,
        )
        W_tilde = torch.sparse.mm(D_rootsquare, torch.sparse.mm(W, D_rootsquare))

        if self.Y_txt is None or self.Y_txt.shape[0] != num_classes:
            self.Y_txt = torch.eye(num_classes, dtype=torch.float32, device=W.device)
            self.Y_0 = torch.zeros((W_tilde.size(0) - num_classes, num_classes), dtype=torch.float32, device=W.device)
            Y_hat = torch.cat((self.Y_txt, self.Y_0), dim=0)
        else:
            self.Y_0 = torch.cat(
                (self.Y_0[:-1], torch.zeros((2, num_classes), dtype=torch.float32, device=W.device)),
                dim=0,
            )
            Y_hat = torch.cat((self.Y_txt, self.Y_0), dim=0)

        W_tilde_0 = _sparse_slice(W_tilde.coalesce(), num_classes, W_tilde.size(0))
        for _ in range(max_iter):
            Y_t = alpha * torch.sparse.mm(W_tilde_0, Y_hat) + (1 - alpha) * self.Y_0
            Y_hat = torch.cat((self.Y_txt, Y_t), dim=0)

        pred = Y_hat[-1, :num_classes]
        propagated = Y_hat[num_classes:, :num_classes]
        top1_values, top1_indices = torch.topk(propagated, 1, dim=1)
        next_Y_0 = torch.zeros_like(propagated)
        next_Y_0.scatter_(1, top1_indices, top1_values * beta)
        self.Y_0 = next_Y_0

        return (pred * 10000).unsqueeze(0).softmax(dim=1)


class ECALP(BaseMethod):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        assert self.args.prompt_type == "no_prompt", "ECALP only supports prompt_type='no_prompt'"
        self.lp_state = _StreamingECALPState()
        self.stream_features = None

    def test_time_tuning(self, inputs):
        return

    def _reset_stream(self):
        self.lp_state.reset()
        self.stream_features = None

    def _prepare_images(self, images):
        if isinstance(images, list):
            for idx in range(len(images)):
                images[idx] = images[idx].cuda(self.args.gpu, non_blocking=True)
            return images[0]

        if len(images.size()) > 4:
            assert images.size()[0] == 1
            images = images.squeeze(0)
        return images.cuda(self.args.gpu, non_blocking=True)

    @torch.no_grad()
    def _predict_single(self, image_feature, text_features):
        if self.stream_features is None:
            features = torch.cat([image_feature, image_feature], dim=0)
        else:
            features = torch.cat([self.stream_features, image_feature, image_feature], dim=0)

        output = self.lp_state.predict(
            image_features=features,
            text_features=text_features,
            k_text=self.args.ecalp_k_text,
            k_image=self.args.ecalp_k_image,
            gamma=self.args.ecalp_gamma,
            alpha=self.args.ecalp_alpha,
            beta=self.args.ecalp_beta,
            max_iter=self.args.ecalp_num_iterations,
        )

        if self.stream_features is None:
            self.stream_features = image_feature
        else:
            self.stream_features = torch.cat([self.stream_features, image_feature], dim=0)

        return output

    def test_time_adapt_eval(self, val_loader, result_dict=None):
        batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
        top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
        top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
        progress = ProgressMeter(len(val_loader), [batch_time, top1, top5], prefix="ECALP Test: ")

        self.model.eval()
        self._reset_stream()
        end = time.time()

        for i, (images, target) in enumerate(val_loader):
            image = self._prepare_images(images)
            target = target.cuda(self.args.gpu, non_blocking=True)

            with torch.amp.autocast(device_type="cuda"):
                _, image_features = self.model(image)
                text_features = self.model.get_text_features()

            image_features = image_features.float()
            text_features = text_features.float()

            outputs = []
            for sample_idx in range(image_features.size(0)):
                output = self._predict_single(
                    image_features[sample_idx : sample_idx + 1],
                    text_features,
                )
                outputs.append(output)
            output = torch.cat(outputs, dim=0)

            if result_dict is not None:
                # probs = output.softmax(dim=1)
                # max_confidence, max_index = torch.max(probs, 1)
                max_confidence, max_index = torch.max(output, 1)
                for j in range(max_confidence.size(0)):
                    result_dict["max_confidence"].append(max_confidence[j].item())
                    result_dict["prediction"].append(max_index[j].item())
                    result_dict["label"].append(target[j].item())

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], output.size(0))
            top5.update(acc5[0], output.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % self.args.print_freq == 0:
                progress.display(i)

        progress.display_summary()
        return [top1.avg, top5.avg]