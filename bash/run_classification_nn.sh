#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/run_classification_nn.py \
--n_batch 24 # TODO: Fill in hyperparameter \
--dataset mnist # TODO: Fill in hyperparameter \
--checkpoint_path trained_classification # TODO: Fill in hyperparameter \
--output_path trained_classification/evaluation_results # TODO: Fill in hyperparameter \
--device gpu # TODO: Fill in hyperparameter \
