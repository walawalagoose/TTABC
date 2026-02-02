#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import sys
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from clip.clip import load as load_clip
from clip.clip import tokenize
from data.prompt_utils import (
    build_basic_prompts,
    build_concept_prompts,
    get_classnames,
    load_concepts,
    make_descriptor_sentence,
    resolve_templates,
)

DOWNLOAD_ROOT=os.path.expanduser('~/.cache/clip')

def _encode_prompts(model, prompts, device, average=True):
    text_tokens = tokenize(prompts).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    if average:
        return text_features.mean(dim=0).cpu()
    return text_features.cpu()


def generate_prototypes(
    model,
    classnames: List[str],
    device: str,
    templates: Optional[List[str]] = None,
    concepts_dict: Optional[Dict[str, List[str]]] = None,
    x_templates: bool = False,
    macro_pooling: bool = False,
    average: bool = True,
):
    prototypes = {}
    prompt_dict = {}

    with torch.no_grad():
        for classname in tqdm(classnames, desc="Generating prototypes"):
            classname_clean = classname.replace("_", " ")
            if concepts_dict is None:
                prompts = build_basic_prompts(classname_clean, templates if templates else None)
                prompt_dict[classname] = prompts
                prototypes[classname] = _encode_prompts(model, prompts, device, average=average)
                continue

            concepts = concepts_dict.get(classname_clean, [])
            if len(concepts) == 0:
                prompts = build_basic_prompts(classname_clean, templates if templates else None)
                prompt_dict[classname] = prompts
                prototypes[classname] = _encode_prompts(model, prompts, device, average=average)
                continue

            if x_templates and macro_pooling and average:
                concept_prompts = [
                    template.format(
                        f"{classname_clean}, {make_descriptor_sentence(concept)}"
                    )
                    for concept in concepts
                    for template in templates
                ]
                template_prompts = [template.format(classname_clean) for template in templates]
                concept_embed = _encode_prompts(model, concept_prompts, device, average=True)
                template_embed = _encode_prompts(model, template_prompts, device, average=True)
                num_concepts = len(concepts)
                num_templates = len(templates)
                prototypes[classname] = (
                    concept_embed * num_concepts + template_embed * num_templates
                ) / (num_concepts + num_templates)
                prompt_dict[classname] = concept_prompts + template_prompts
                continue

            prompts = build_concept_prompts(
                classname_clean,
                concepts,
                templates=templates,
                x_templates=x_templates,
            )
            prompt_dict[classname] = prompts
            prototypes[classname] = _encode_prompts(model, prompts, device, average=average)

    return prototypes, prompt_dict


def main():
    parser = argparse.ArgumentParser(description='Generate prototypes for TPS')
    parser.add_argument('--arch', default='ViT-B/16', choices=['RN50', 'ViT-B/16', 'ViT-L/14'], help='CLIP architecture')
    parser.add_argument('--datasets', default='I', help='Datasets to generate prototypes for, separated by /')
    parser.add_argument('--output_dir', default='./.caches/prototypes', help='Output directory')
    parser.add_argument('--templates', action='store_true', help='Use template prompts')
    parser.add_argument('--concepts_json', type=str, default=None, help='Path to GPT concepts JSON')
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'macro', 'none'],
                        help='Prompt pooling: mean (default), macro (concept+template only), none (keep per-prompt embeds)')
    parser.add_argument('--save_prompts', action='store_true', help='Save prompt lists to JSON')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.arch} CLIP model on {device}...")
    model, embed_dim, _ = load_clip(args.arch, device=device, download_root=DOWNLOAD_ROOT)
    model.eval()
    print(f"Successfully loaded {args.arch} (embed_dim: {embed_dim})")

    concepts_dict = load_concepts(args.concepts_json) if args.concepts_json else None

    datasets = args.datasets.split('/')
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset}")
        print(f"{'='*60}")
        try:
            classnames = get_classnames(dataset)
            print(f"Number of classes: {len(classnames)}")

            use_templates = bool(args.templates)
            templates = resolve_templates(dataset) if use_templates else None
            x_templates = bool(concepts_dict) and use_templates
            macro_pooling = args.pooling == 'macro'
            average = args.pooling != 'none'
            if macro_pooling and not x_templates:
                print("[WARN] macro pooling only applies to concept+template prompts; falling back to mean pooling.")
                macro_pooling = False

            prototypes, prompt_dict = generate_prototypes(
                model,
                classnames,
                device,
                templates=templates,
                concepts_dict=concepts_dict,
                x_templates=x_templates,
                macro_pooling=macro_pooling,
                average=average,
            )

            suffix = ""
            if args.concepts_json:
                suffix = "_gpt4_templates" if x_templates else "_gpt4"
                if macro_pooling and x_templates:
                    suffix += "_macro"
            elif use_templates:
                suffix = "_templates"
            if args.pooling == 'none':
                suffix += "_raw"

            filename = f"{dataset}_{args.arch.replace('/', '-')}{suffix}_prototypes.pkl"
            filepath = os.path.join(args.output_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(prototypes, f)
            print(f"✓ Saved {len(prototypes)} prototypes to: {filepath}")

            if args.save_prompts:
                prompt_path = os.path.join(
                    args.output_dir,
                    f"{dataset}_{args.arch.replace('/', '-')}{suffix}_prompts.json",
                )
                with open(prompt_path, 'w') as f:
                    json.dump(prompt_dict, f, indent=2)
                print(f"✓ Saved prompts to: {prompt_path}")

            with open(filepath, 'rb') as f:
                loaded = pickle.load(f)
            print(f"✓ Verified: loaded {len(loaded)} prototypes")
        except Exception as e:
            print(f"✗ Error processing {dataset}: {e}")
            continue

    print(f"\n{'='*60}")
    print("Prototype generation complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()