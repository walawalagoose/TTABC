import json
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from clip import tokenize

from data.cls_to_names import *
from data.cifar_prompts import cifar10_classes, cifar100_classes, cifar_templates
from data.imagenet_prompts import imagenet_classes, imagenet_templates
from data.imagenet_variants import (
    imagenet_a_mask,
    imagenet_r_mask,
    imagenet_v_mask,
)


def get_classnames(dataset: str) -> List[str]:
    """Get class names for a dataset identifier."""
    if dataset == 'I':
        return imagenet_classes
    if dataset == 'A':
        return [imagenet_classes[i] for i in imagenet_a_mask]
    if dataset == 'R':
        classnames = []
        for i, m in enumerate(imagenet_r_mask):
            if m:
                classnames.append(imagenet_classes[i])
        return classnames
    if dataset == 'V':
        return [imagenet_classes[i] for i in imagenet_v_mask]
    if dataset == 'K':
        return imagenet_classes[:200]  # ImageNet-Sketch uses first 200 classes
    if dataset in ['cifar10', 'cifar10c']:
        return cifar10_classes
    if dataset in ['cifar100', 'cifar100c']:
        return cifar100_classes

    try:
        return eval(f"{dataset.lower()}_classes")
    except Exception as exc:
        raise ValueError(f"Unknown dataset: {dataset}") from exc


def resolve_templates(dataset: str) -> List[str]:
    if 'cifar' in dataset:
        return cifar_templates
    return imagenet_templates


def make_descriptor_sentence(descriptor: str) -> str:
    """Convert a descriptor into a proper sentence fragment."""
    if descriptor.startswith(('a', 'an')):
        return f"which is {descriptor}"
    if descriptor.startswith(('has', 'often', 'typically', 'may', 'can')):
        return f"which {descriptor}"
    if descriptor.startswith('used'):
        return f"which is {descriptor}"
    return f"which has {descriptor}"


def build_prompts(
    classname: str,
    concepts: Optional[Iterable[str]] = None,
    templates: Optional[Iterable[str]] = None,
    x_templates: bool = False,
) -> List[str]:
    if not concepts:
        if templates:
            return [template.format(classname) for template in templates]
        return [f"a photo of a {classname}"]

    descriptor_prompts = [
        f"{classname}, {make_descriptor_sentence(concept)}" for concept in concepts
    ]
    if not x_templates:
        return descriptor_prompts

    if not templates:
        raise ValueError("templates must be provided when x_templates=True")
    return [
        template.format(f"{classname}, {make_descriptor_sentence(concept)}")
        for concept in concepts
        for template in templates
    ]


def build_basic_prompts(classname: str, templates: Optional[Iterable[str]] = None) -> List[str]:
    return build_prompts(classname, concepts=None, templates=templates)


def build_concept_prompts(
    classname: str,
    concepts: Iterable[str],
    templates: Optional[Iterable[str]] = None,
    x_templates: bool = False,
) -> List[str]:
    return build_prompts(classname, concepts=concepts, templates=templates, x_templates=x_templates)


def load_concepts(concepts_json: str) -> Dict[str, List[str]]:
    with open(concepts_json, 'r') as f:
        concepts_dict = json.load(f)
    return concepts_dict


def normalize_prompt_setting(prompt_setting: Optional[Dict]) -> Optional[Dict]:
    if prompt_setting is None:
        return None
    if not isinstance(prompt_setting, dict):
        raise TypeError("prompt_setting must be a dict or None")
    setting = dict(prompt_setting)
    setting.setdefault("templates", False)
    setting.setdefault("pooling", "mean")
    return setting


def build_prompt_sets(
    classnames: List[str],
    prompt_setting: Optional[Dict],
    dataset: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Dict[str, List[str]]]], Optional[Dict]]:
    """Build per-class prompt sets based on prompt_setting.

    Returns:
        prompt_sets: {classname: {all_prompts, concept_prompts, template_prompts}}
        meta: {x_templates, macro_pooling, pooling}
    """
    setting = normalize_prompt_setting(prompt_setting)
    if setting is None:
        return None, None

    templates_flag = bool(setting.get("templates"))
    concepts_json = setting.get("concepts_json")
    concepts_dict = setting.get("concepts_dict")
    pooling = setting.get("pooling", "mean")
    macro_pooling = pooling == "macro"

    if concepts_dict is None and concepts_json:
        concepts_dict = load_concepts(concepts_json)

    templates_list = setting.get("templates_list")
    if templates_flag and templates_list is None:
        templates_list = resolve_templates(dataset) if dataset else imagenet_templates

    x_templates = templates_flag and concepts_dict is not None

    prompt_sets = {}
    for classname in classnames:
        classname_clean = classname.replace("_", " ")
        concept_prompts: List[str] = []
        template_prompts: List[str] = []

        if concepts_dict is None:
            all_prompts = build_prompts(
                classname_clean,
                concepts=None,
                templates=templates_list if templates_flag else None,
            )
        else:
            concepts = concepts_dict.get(classname_clean, [])
            if len(concepts) == 0:
                all_prompts = build_prompts(
                    classname_clean,
                    concepts=None,
                    templates=templates_list if templates_flag else None,
                )
            else:
                if x_templates:
                    concept_prompts = [
                        template.format(
                            f"{classname_clean}, {make_descriptor_sentence(concept)}"
                        )
                        for concept in concepts
                        for template in templates_list
                    ]
                    template_prompts = [
                        template.format(classname_clean) for template in templates_list
                    ]
                    all_prompts = concept_prompts + template_prompts
                else:
                    all_prompts = build_prompts(
                        classname_clean,
                        concepts=concepts,
                        templates=templates_list,
                        x_templates=False,
                    )

        prompt_sets[classname] = {
            "all_prompts": all_prompts,
            "concept_prompts": concept_prompts,
            "template_prompts": template_prompts,
        }

    meta = {
        "x_templates": x_templates,
        "macro_pooling": macro_pooling,
        "pooling": pooling,
    }
    return prompt_sets, meta


def build_text_features(
    clip_model,
    classnames: List[str],
    prompt_setting: Optional[Dict],
    dataset: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Build averaged text features for classnames based on prompt_setting.

    Returns:
        pre_text_features: unnormalized (K, d) or None
        text_features: normalized (K, d) or None
    """
    prompt_sets, meta = build_prompt_sets(classnames, prompt_setting, dataset=dataset)
    if prompt_sets is None:
        return None, None

    pre_list = []
    text_list = []
    for classname in classnames:
        prompts = prompt_sets[classname]["all_prompts"]
        concept_prompts = prompt_sets[classname]["concept_prompts"]
        template_prompts = prompt_sets[classname]["template_prompts"]

        if (
            meta["macro_pooling"]
            and meta["x_templates"]
            and concept_prompts
            and template_prompts
        ):
            concept_tokens = torch.cat([tokenize(p) for p in concept_prompts])
            template_tokens = torch.cat([tokenize(p) for p in template_prompts])
            if device is not None:
                concept_tokens = concept_tokens.to(device)
                template_tokens = template_tokens.to(device)
            concept_pre = clip_model.encode_text(concept_tokens)
            template_pre = clip_model.encode_text(template_tokens)
            concept_text = concept_pre / concept_pre.norm(dim=-1, keepdim=True)
            template_text = template_pre / template_pre.norm(dim=-1, keepdim=True)

            num_concepts = len(concept_prompts)
            num_templates = len(template_prompts)
            pre_text = (concept_pre.mean(dim=0) * num_concepts + template_pre.mean(dim=0) * num_templates) / (
                num_concepts + num_templates
            )
            text_feature = (concept_text.mean(dim=0) * num_concepts + template_text.mean(dim=0) * num_templates) / (
                num_concepts + num_templates
            )
        else:
            tokens = torch.cat([tokenize(p) for p in prompts])
            if device is not None:
                tokens = tokens.to(device)
            pre = clip_model.encode_text(tokens)
            text = pre / pre.norm(dim=-1, keepdim=True)
            pre_text = pre.mean(dim=0)
            text_feature = text.mean(dim=0)

        pre_list.append(pre_text)
        text_list.append(text_feature)

    pre_text_features = torch.stack(pre_list, dim=0)
    text_features = torch.stack(text_list, dim=0)
    return pre_text_features, text_features


def build_ctx_init(
    clip_model,
    classnames: List[str],
    prompt_setting: Optional[Dict],
    n_ctx: int,
    dataset: Optional[str] = None,
    device: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """Build a ctx initialization tensor (n_ctx, dim) from prompt_setting."""
    prompt_sets, _ = build_prompt_sets(classnames, prompt_setting, dataset=dataset)
    if prompt_sets is None:
        return None

    ctx_list = []
    for classname in classnames:
        prompts = prompt_sets[classname]["all_prompts"]
        tokens = torch.cat([tokenize(p) for p in prompts])
        if device is not None:
            tokens = tokens.to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokens).type(clip_model.dtype)
        ctx_vectors = embedding[:, 1 : 1 + n_ctx, :]
        ctx_list.append(ctx_vectors.mean(dim=0))

    return torch.stack(ctx_list, dim=0).mean(dim=0)
