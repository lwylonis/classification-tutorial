#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_classification_nn.py \
--n_batch 24 # TODO: Fill in hyperparameter \
--dataset mnist # TODO: Fill in hyperparameter \
--n_epoch 50 # TODO: Fill in hyperparameter \
--learning_rate 1e-3 # TODO: Fill in hyperparameter \
--learning_rate_decay 0.5 # TODO: Fill in hyperparameter \
--learning_rate_period 10 # TODO: Fill in hyperparameter \
--checkpoint_path trained_classification # TODO: Fill in hyperparameter \
--device gpu # TODO: Fill in hyperparameter \
