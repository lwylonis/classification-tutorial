#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_classification_cnn.py \
--n_batch 24 \
--dataset cifar10 \
--encoder_type vggnet11 \
--n_epoch 50 \
--learning_rate 1e-3 \
--learning_rate_decay 3e-1 \
--learning_rate_period 10 \
--checkpoint_path cnn_mnist_cifar10_t1 \
--device cpu \
