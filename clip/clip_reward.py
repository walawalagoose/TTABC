"""
CLIP Reward module for RLCF.
Adapted from the official RLCF implementation to match TTABC style.
"""

import os

import torch
import torch.nn as nn

import clip

DOWNLOAD_ROOT = os.path.expanduser("~/.cache/clip")

CONFIDENCES = {
    "ViT-L/14@336px": 10,
    "ViT-L/14": 5,
    "RN50x64": 3,
    "ViT-B/16": 1,
}


def get_reward_model(device, args):
    if getattr(args, "multiple_reward_models", False):
        reward_model = CLIPRewardsMultiple(
            device,
            arch=["ViT-L/14@336px", "RN50x64", "ViT-L/14"],
            classification=True,
            amplify_rewards=args.reward_amplify,
            sample_k=args.sample_k,
            reward_process=args.reward_process,
            process_batch=args.process_batch,
            weighted_scores=getattr(args, "weighted_scores", True),
        )
    else:
        reward_model = CLIPRewards(
            device,
            arch=args.reward_arch,
            classification=True,
            amplify_rewards=args.reward_amplify,
            sample_k=args.sample_k,
            reward_process=args.reward_process,
            process_batch=args.process_batch,
        )

    return reward_model


class BaseRewards(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()
    def extract_image_features(self, images):
        raise NotImplementedError

    @torch.no_grad()
    def extract_text_features(self, captions=None, tokenized_cap=None):
        raise NotImplementedError

    @torch.no_grad()
    def set_class_features(self, classnames=None, tokenized_classes=None):
        self.class_features = self.extract_text_features(captions=classnames, tokenized_cap=tokenized_classes)

    @torch.no_grad()
    def set_image_features(self, images):
        self.image_features = self.extract_image_features(images)


class CLIPRewards(BaseRewards):
    def __init__(
        self,
        device,
        arch="ViT-B/16",
        clipscore_weight=2.5,
        classification=True,
        amplify_rewards=False,
        sample_k=5,
        reward_process=True,
        process_batch=False,
        default_resolutions=224,
    ) -> None:
        super().__init__()
        self.default_resolutions = default_resolutions
        self.clip_model, self.embed_dim, self.preprocess = clip.load(
            arch, device=device, download_root=DOWNLOAD_ROOT, jit=False
        )
        self.resolutions = self.clip_model.visual.input_resolution
        self.clipscore_weight = clipscore_weight
        self.device = device
        self.classification = classification
        self.class_features = None
        self.image_features = None
        self.amplify_rewards = amplify_rewards
        self.sample_k = sample_k
        self.reward_process = reward_process
        self.process_batch = process_batch
        self.clip_model.eval()

        print(
            "\n CLIPRewards model created: \n"
            "\t visual backbone: {}, resolutions: {}, amplify_rewards: {}, sample_k: {}, \n"
            "\t reward_process: {}, process_batch: {}\n".format(
                arch, self.resolutions, amplify_rewards, sample_k, reward_process, process_batch
            )
        )

    @torch.no_grad()
    def CLIPScore(
        self,
        class_index,
        images=None,
        image_features=None,
        captions=None,
        tokenized_cap=None,
        text_features=None,
        pairwise=True,
    ):
        text_features = self.class_features[class_index]
        image_features = torch.repeat_interleave(self.image_features, self.sample_k, dim=0)

        if pairwise:
            similarity = self.clipscore_weight * text_features @ image_features.t()
        else:
            similarity = self.clipscore_weight * torch.sum(text_features * image_features, dim=-1)

        scores = torch.maximum(similarity, torch.zeros_like(similarity)).squeeze()
        return scores

    @torch.no_grad()
    def extract_image_features(self, images):
        if self.resolutions != self.default_resolutions:
            images = nn.functional.interpolate(images, size=self.resolutions, mode="bicubic", align_corners=True)
        image_features = self.clip_model.encode_image(images).float()
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    @torch.no_grad()
    def extract_text_features(self, captions=None, tokenized_cap=None):
        if captions is not None:
            caption_tokens = clip.tokenize(captions, truncate=True).to(self.device)
            text_features = self.clip_model.encode_text(caption_tokens).float()
        if tokenized_cap is not None:
            text_features = self.clip_model.encode_text(tokenized_cap).float()

        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    @torch.no_grad()
    def rewards_post_process(self, clip_score):
        if clip_score.shape[-1] > 1 and self.reward_process:
            mean = torch.mean(clip_score, dim=-1, keepdim=True)
            if self.amplify_rewards:
                std = torch.std(clip_score, dim=-1, keepdim=True) + 1e-5
            else:
                std = 1.0
            clip_score = (clip_score - mean) / std

        return clip_score.flatten()


class CLIPRewardsMultiple(BaseRewards):
    def __init__(
        self,
        device,
        arch=None,
        clipscore_weight=2.5,
        classification=True,
        amplify_rewards=False,
        sample_k=5,
        reward_process=True,
        process_batch=True,
        weighted_scores=True,
        default_resolutions=224,
    ) -> None:
        super().__init__()
        arch = arch or ["ViT-B/16", "RN50x64", "ViT-L/14"]
        clip_models = []
        self.preprocess = []
        self.resolutions = []
        weights = []
        self.default_resolutions = default_resolutions
        for ar in arch:
            clip_model, embed_dim, preprocess = clip.load(ar, device=device, download_root=DOWNLOAD_ROOT, jit=False)
            clip_models.append(clip_model)
            self.preprocess.append(preprocess)
            self.resolutions.append(clip_model.visual.input_resolution)
            weights.append(CONFIDENCES[ar])
        self.clip_models = nn.ModuleList(clip_models)
        self.n_model = len(self.clip_models)
        self.weights = [round(x / sum(weights), 2) for x in weights]

        self.clipscore_weight = clipscore_weight
        self.device = device
        self.classification = classification
        self.class_features = None
        self.image_features = None
        self.amplify_rewards = amplify_rewards
        self.sample_k = sample_k
        self.reward_process = reward_process
        self.process_batch = process_batch
        self.weighted_scores = weighted_scores

        self.clip_models.eval()

        print(
            "\n CLIPRewardsMultiple model created: \n"
            "\t visual backbone: {}, resolutions: {}, weighted_scores / weights: [ {} / {} ] \n"
            "\t amplify_rewards: {}, sample_k: {}, reward_process: {}, process_batch: {}\n".format(
                arch,
                self.resolutions,
                weighted_scores,
                self.weights,
                amplify_rewards,
                sample_k,
                reward_process,
                process_batch,
            )
        )

    @torch.no_grad()
    def CLIPScore(
        self,
        class_index,
        images=None,
        image_features=None,
        captions=None,
        tokenized_cap=None,
        text_features=None,
        pairwise=True,
    ):
        all_scores = []
        for i in range(self.n_model):
            text_features = self.class_features[i][class_index]
            image_features = torch.repeat_interleave(self.image_features[i], self.sample_k, dim=0)

            if pairwise:
                similarity = self.clipscore_weight * text_features @ image_features.t()
                raise NotImplementedError
            else:
                similarity = self.clipscore_weight * torch.sum(text_features * image_features, dim=-1)

            scores = torch.maximum(similarity, torch.zeros_like(similarity)).squeeze()
            all_scores.append(scores)

        scores = torch.stack(all_scores, dim=0)
        if self.weighted_scores:
            weights = torch.tensor(self.weights, device=scores.device, dtype=scores.dtype).unsqueeze(1)
            final_scores = torch.sum(weights * scores, dim=0)
        else:
            final_scores = torch.mean(scores, dim=0)

        return final_scores

    @torch.no_grad()
    def extract_image_features(self, images):
        image_features = []
        for i in range(self.n_model):
            if self.resolutions[i] != self.default_resolutions:
                tmp_images = nn.functional.interpolate(
                    images, size=self.resolutions[i], mode="bicubic", align_corners=True
                )
            else:
                tmp_images = images
            image_feat = self.clip_models[i].encode_image(tmp_images).float()
            image_feat = image_feat / image_feat.norm(dim=1, keepdim=True)
            image_features.append(image_feat)

        return image_features

    @torch.no_grad()
    def extract_text_features(self, captions=None, tokenized_cap=None):
        text_features = []
        for i in range(self.n_model):
            if captions is not None:
                caption_tokens = clip.tokenize(captions, truncate=True).to(self.device)
                text_feat = self.clip_models[i].encode_text(caption_tokens).float()

            if tokenized_cap is not None:
                text_feat = self.clip_models[i].encode_text(tokenized_cap).float()

            text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)
            text_features.append(text_feat)

        return text_features

    @torch.no_grad()
    def rewards_post_process(self, clip_score):
        if clip_score.shape[-1] > 1 and self.reward_process:
            mean = torch.mean(clip_score, dim=-1, keepdim=True)
            if self.amplify_rewards:
                std = torch.std(clip_score, dim=-1, keepdim=True) + 1e-5
            else:
                std = 1.0
            clip_score = (clip_score - mean) / std

        return clip_score.flatten()
