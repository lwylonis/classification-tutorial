#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_classification_cnn.py \
--n_batch # TODO: Fill in hyperparameter \
--dataset # TODO: Fill in hyperparameter \
--encoder_type # TODO: Fill in hyperparameter \
--n_epoch # TODO: Fill in hyperparameter \
--learning_rate # TODO: Fill in hyperparameter \
--learning_rate_decay # TODO: Fill in hyperparameter \
--learning_rate_period # TODO: Fill in hyperparameter \
--checkpoint_path # TODO: Fill in hyperparameter \
--device # TODO: Fill in hyperparameter \
