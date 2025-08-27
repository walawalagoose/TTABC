# TTABC: Test-Time Adaptation Benchmark of CLIP

This repository provides the official implementation of our paper [[Paper Link](https://doi.org/10.48550/arXiv.2506.02671)]:

> NB!!    
> Authors: Xiao Chen, Jiazhen Huang

The implementation is built upon [TPT](https://github.com/azshue/TPT).

Experiment results are shown in [Link](https://docs.qq.com/sheet/DZUp4dFRZZER0VVdy?tab=jci45g).

## Available algorithms
The [currently available algorithms](https://github.com/LINs-lab/ttab/tree/main/ttab/model_adaptation) are:

- Batch Normalization Test-time Adaptation (BN_Adapt, [Schneider et al., 2020](https://arxiv.org/abs/2006.16971))
- Source Hypothesis Transfer (SHOT, [Liang et al., 2020](https://arxiv.org/abs/2002.08546))
- Test-time Training (TTT, [Sun et al., 2020](https://arxiv.org/abs/1909.13231))
- Test-time Entropy Minimization (TENT, [Wang et al., 2021](https://arxiv.org/abs/2006.10726))
- Test-time Template Adjuster (T3A, [Iwasawa & Matsuo, 2021](https://proceedings.neurips.cc/paper/2021/hash/1415fe9fea0fa1e45dddcff5682239a0-Abstract.html))
- Marginal Entropy Minimization (MEMO, [Zhang et al., 2022](https://arxiv.org/abs/2110.09506))
- Non-i.i.d.Test-time Adaptation (NOTE, [Gong et al., 2022](https://arxiv.org/abs/2208.05117))
- Continual Test-time Adaptation (CoTTA, [Wang et al., 2022](https://arxiv.org/abs/2203.13591))
- Conjugate Pseudo-Labels (Conjugate PL, [Goyal et al., 2022](https://arxiv.org/abs/2207.09640))
- Efficient Anti-forgetting Test-time Adaptation (EATA, [Niu et al., 2022](https://arxiv.org/abs/2204.02610))
- Sharpness-aware Entropy Minimization (SAR, [Niu et al., 2023](https://arxiv.org/abs/2302.12400))

Send us a PR to add your algorithm! Our implementations use ResNets ([He et al., 2015](https://arxiv.org/abs/1512.03385)) and ViTs ([Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)) pretrained by ERM or self-supervised rotation prediction task ([Gidaris et al., 2018](https://arxiv.org/abs/1803.07728)).

## Available datasets

Our evaluation focuses on 

1) fine-grained classification: ImageNet, Flower102, OxfordPets, SUN397, DTD, Food101, StanfordCars, Aircraft, UCF101, EuroSAT, Caltech101

2) natural distribution shift: ImageNet-V2, ImageNet-A, ImageNet-R, ImageNet-Sketch

Prepare the datasets based on the following link [TPT](https://github.com/azshue/TPT).
The download links of currently available datasets are:

> Download link:
> + [Imagenet-A](https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar)
> + [ImageNet-V2](https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main)
> + [Imagenet-R](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar)
> + [Imagenet-Sketch](https://www.kaggle.com/datasets/wanghaohan/imagenetsketch)
> + [CIFAR-10-C](https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1)
> + [CIFAR-100-C](https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1)
> + [ImageNet-C](https://zenodo.org/records/2235448)

Send us a PR to add your dataset! Any custom image dataset with folder structure `dataset/domain/class/image.xyz` is readily usable.

## Installation
```bash

# Clone this repo
git clone https://github.com/Cevaaa/TTABC.git
cd N-TPT

# Create a conda enviroment
conda create -n ntpt python=3.10 -y
conda activate ntpt

# Install PyTorch. Below is a sample command to do this, but you should check the following link
# to find installation instructions that are specific to your compute platform:
# https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 # UPDATE ME!

pip install -r requirements.txt

```

## Running Experiments

In each of the .sh files, change the `{data_root}` accordingly. Additionally, you can change the CLIP architecture by modifying the `{arch}` parameter to either `RN50` or `ViT-B/16`. By changing `{run_type}`, you can select a method, such as `tpt`, `ctpt`, `otpt`, or other supported methods.

An example to run TPT:

```bash

bash ./test_ds_example.sh R # test on imgenet-r
bash ./test_ds_example.sh I/A/V/R/K # test on imgenet, imgenet-a, imgenet-v, imgenet-r and imgenet-k
# NOTE: some methods can only be tested on one dataset

```

The command line argument {dataset} can be specified as follows: ‘I’, ‘DTD’, ‘Flower102’, ‘Food101’, ‘Cars’, ‘SUN397’, ‘Aircraft’, ‘Pets’, ‘Caltech101’, ‘UCF101’, or ‘eurosat’ for fine-grained classification datasets, and ‘V2’, ‘A’, ‘R’, or ‘K’ for datasets with natural distribution shifts.

## Adding a new method

First, create a `newMethod.py` file under `./methods`. In this file, create the class `newMethod` and make it inherit from `methods.base_method.BaseMethod`. Make sure that the functions `test_time_tuning` and `test_time_adapt_eval` have been implemented.

Then, add the dictionary key for `newMethod` and its corresponding return value in `./methods/__init__.py`.

Finally, add a `choices` option to `--run_type` in `./main.py` and implement the corresponding operation in `main_worker`.

## Acknowledgement
This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.2022-0-00184, Development and Study of AI Technologies to Inexpensively Conform to Evolving Policy on Ethics), and Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. 2022-0-00951, Development of Uncertainty-Aware Agents Learning by Asking Questions).

Also, we thank the authors of the [CoOp/CoCoOp](https://github.com/KaiyangZhou/CoOp) and [TPT](https://github.com/azshue/TPT) for their open-source contributions and their assistance with the data preparation.

## Citation
If you find our work useful in your research, please cite:
```
@article{chen2025small,
  title={Small Aid, Big Leap: Efficient Test-Time Adaptation for Vision-Language Models with AdaptNet},
  author={Chen, Xiao and Huang, Jiazhen and Jiang, Qinting and Huang, Fanding and Fu, Xianghua and Jiang, Jingyan and Wang, Zhi},
  journal={arXiv preprint arXiv:2506.02671},
  year={2025}
}
```

## Contact
If you have any questions, please feel free to email chen-x25@mails.tsinghua.edu.cn
