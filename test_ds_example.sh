#!/bin/bash

data_root='/path/to/your/data/root'
testsets=$1
arch=RN50 # ViT-B/16
run_type=tpt # tpt ctpt otpt ntpt
down_sample_ratio=0.001
log_dir='./logs/'
gpu=1

python ./main.py ${data_root} --test_sets ${testsets} \
-a ${arch} --gpu ${gpu} \
--tpt --run_type ${run_type} --I_augmix \
--down_sample_ratio ${down_sample_ratio} \
--log_dir ${log_dir} \