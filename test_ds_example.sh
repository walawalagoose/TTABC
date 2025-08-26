#!/bin/bash

data_root='/path/to/your/data/root'
testsets=$1
arch=RN50 # ViT-B/16
bs=64
ctx_init=a_photo_of_a
run_type=tpt # tpt ctpt otpt ntpt
down_sample_ratio=0.001
lambda_term=2.0
log_dir='./logs/'
is_ece=False
gpu=1

python ./main.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu ${gpu} \
--tpt --ctx_init ${ctx_init} --run_type ${run_type} --I_augmix \
--down_sample_ratio ${down_sample_ratio} \
--lambda_term ${lambda_term} \
--log_dir ${log_dir} --is_ece ${is_ece} \