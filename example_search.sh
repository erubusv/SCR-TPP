#!/bin/bash
# Example hyperparameter search commands

# Quick search for HNSTPP (default parameters)
python workspace/train/search_hyperparams.py --model hnstpp --epochs 5

# Quick search for Transformer (default parameters)
python workspace/train/search_hyperparams.py --model transformer --epochs 5

# Custom search ranges for HNSTPP
python workspace/train/search_hyperparams.py \
  --model hnstpp \
  --epochs 10 \
  --lrs 1e-4,1e-3,1e-2 \
  --min_lrs 1e-5,1e-4 \
  --lambda_ortho 0,0.001,0.01 \
  --lambda_sparse 0,0.001,0.01 \
  --lambda_inter 0,0.001,0.01

# Results are saved to: ./search_results/{model}_search_results.csv
