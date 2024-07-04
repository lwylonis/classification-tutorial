#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/run_classification_cnn.py \
--n_batch 24 \
--dataset cifar10 \
--encoder_type vggnet11 \
--checkpoint_path cnn_mnist_cifar10_t1 \
--output_path cnn_mnist_cifar10_t1/evaluation_results \
--device cpu \
