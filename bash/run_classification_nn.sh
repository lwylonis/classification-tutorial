#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/run_classification_nn.py \
--n_batch 128 \
--dataset cifar10 \
--checkpoint_path nn_cifar10_t1 \
--output_path nn_cifar10_t1/evaluation_results \
--device cpu \
