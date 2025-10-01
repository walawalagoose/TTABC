# TTABC: Test-Time Adaptation Benchmark of CLIP

This is an open source Test-Time Adaptation Benchmark of CLIP repository based on PyTorch. It is joint work by Jiazhen Huang, Xiao Chen and Xu Jiang.

The implementation is built upon [TPT](https://github.com/azshue/TPT). Experiment results are shown in [Link](https://docs.qq.com/sheet/DZUp4dFRZZER0VVdy?tab=jci45g).

## Available algorithms
The [currently available algorithms](https://github.com/Cevaaa/TTABC_basic/tree/main/ttabc/model_selection) are:

- Learning Transferable Visual Models From Natural Language Supervision (zero-shot, [Alec Radford et al., 2021](https://arxiv.org/abs/2103.00020))
- Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models (TPT, [Manli Shu et al., 2022](https://arxiv.org/abs/2209.07511))
- C-TPT: Calibrated Test-Time Prompt Tuning for Vision-Language Models via Text Feature Dispersion (C-TPT, [Hee Suk Yoon et al., 2024](https://arxiv.org/abs/2403.14119))
- Efficient Test-Time Adaptation of Vision-Language Models (TDA, [Adilbek Karmanov et al., 2024](https://arxiv.org/abs/2403.18293))
- BoostAdapter: Improving Vision-Language Test-Time Adaptation via Regional Bootstrapping (BoostAdapter, [Taolin Zhang et al., 2024](https://arxiv.org/abs/2410.15430))
- O-TPT: Orthogonality Constraints for Calibrating Test-time Prompt Tuning in Vision-Language Models (O-TPT, [Ashshak Sharifdeen et al., 2025](https://arxiv.org/abs/2503.12096))

Send us a PR to add your algorithm!

## Available datasets

Our evaluation focuses on 

1) fine-grained classification: ImageNet, Flower102, OxfordPets, SUN397, DTD, Food101, StanfordCars, Aircraft, UCF101, EuroSAT, Caltech101

2) natural distribution shift: ImageNet-V2, ImageNet-A, ImageNet-R, ImageNet-Sketch

Prepare the datasets based on the following link [TPT](https://github.com/azshue/TPT).
The download links of currently available datasets are:

<details>
<summary>Download link</summary>

+ [Imagenet-A](https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar)
+ [ImageNet-V2](https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main)
+ [Imagenet-R](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar)
+ [Imagenet-Sketch](https://www.kaggle.com/datasets/wanghaohan/imagenetsketch)
+ [CIFAR-10-C](https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1)
+ [CIFAR-100-C](https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1)
+ [ImageNet-C](https://zenodo.org/records/2235448)
+ [Flower102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz)
+ [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz)
+ [OxfordPets](https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz)
+ [StanfordCars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
+ [UCF101](https://drive.google.com/file/d/10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O/view?usp=sharing)
+ [Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz)
+ [Food101](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz)
+ [SUN397](http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz)
+ [Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz)
+ [EuroSAT](http://madm.dfki.de/files/sentinel/EuroSAT.zip)

</details>

<details>
<summary>Dataset path</summary>

> /path/to/your/data/root  
> ├── Aircraft  
> ├── Caltech101  
> ├── CIFAR-10  
> ├── CIFAR-100  
> ├── CIFAR-100-C  
> ├── CIFAR-10-C  
> ├── DTD  
> ├── EuroSAT  
> ├── Flower102  
> ├── Food101  
> ├── ImageNet  
> ├── imagenet-a  
> ├── imagenet-c  
> ├── imagenet-r  
> ├── ImageNet-Sketch  
> ├── imagenetv2-matched-frequency-format-val  
> ├── OxfordPets  
> ├── StanfordCars  
> ├── SUN397  
> ├── UCF101

</details>

## Installation
```bash

# Clone this repo
git clone https://github.com/Cevaaa/TTABC_basic.git
cd TTABC_basic

# Create a conda enviroment
conda create -n ttabc python=3.10 -y
conda activate ttabc

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

```

The command line argument `{dataset}` can be specified as follows: `I`, `DTD`, `Flower102`, `Food101`, `StanfordCars`, `SUN397`, `Aircraft`, ‘OxfordPets’, `Caltech101`, `UCF101`, or `EuroSAT` for fine-grained classification datasets, and `V2`, `A`, `R`, or `K` for datasets with natural distribution shifts.

## Adding a new method

First, create a `newMethod.py` file under `./ttabc/model_selection`. In this file, create the class `newMethod` and make it inherit from `./ttabc/model_selection/base_method.BaseMethod`. Make sure that the functions `test_time_tuning` and `test_time_adapt_eval` have been implemented.

Then, add the dictionary key for `newMethod` and its corresponding return value in `./ttabc/model_selection/__init__.py`.

Finally, add a `choices` option to `--run_type` in `./main.py` and implement the corresponding operation in `main_worker`.

## Acknowledgement

We thank the authors of the [CoOp/CoCoOp](https://github.com/KaiyangZhou/CoOp) and [TPT](https://github.com/azshue/TPT) for their open-source contributions and their assistance with the data preparation.

<!-- ## Citation
If you find our work useful in your research, please cite:
```
@article{chen2025small,
  title={Small Aid, Big Leap: Efficient Test-Time Adaptation for Vision-Language Models with AdaptNet},
  author={Chen, Xiao and Huang, Jiazhen and Jiang, Qinting and Huang, Fanding and Fu, Xianghua and Jiang, Jingyan and Wang, Zhi},
  journal={arXiv preprint arXiv:2506.02671},
  year={2025}
}
``` -->

## Contact
If you have any questions, please feel free to email chen-x25@mails.tsinghua.edu.cn.
