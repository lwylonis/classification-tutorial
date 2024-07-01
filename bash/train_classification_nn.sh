#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_classification_nn.py \
--n_batch 128 \
--dataset cifar10 \
--n_epoch 50 \
--learning_rate 3e-2 \
--learning_rate_decay 5e-1 \
--learning_rate_period 10 \
--checkpoint_path nn_cifar10_t1 \
--device cpu \
