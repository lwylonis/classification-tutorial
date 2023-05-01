#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/run_classification_nn.py \
--n_batch # TODO: Fill in hyperparameter \
--dataset # TODO: Fill in hyperparameter \
--checkpoint_path # TODO: Fill in hyperparameter \
--output_path # TODO: Fill in hyperparameter \
--device # TODO: Fill in hyperparameter \
