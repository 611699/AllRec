#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision=no \
  --dynamo_backend=no \
  main.py \
  --model=MMPGR \
  --dataset=AmazonReviews2014 \
  --category=Toys_and_Games \
  --tokenizer_datasets_mode=slide_window \
  --log_dir=Toys_and_Games
