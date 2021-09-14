#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python run_classifier.py \
    --data_dir "./data/IMDB/" \
    --model_name_or_path bert-base-uncased \
    --output_dir "./output/tmp" \
    --do_train \
    --do_eval \
    --loss_weight=1.0
