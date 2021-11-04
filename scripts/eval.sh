#!/bin/bash

TASK="IMDB"
# MODEL_PATH="bert-base-uncased"
MODEL_PATH="./output/tmp/checkpoint-10000/"
MODEL_PATH="./output/tmp/"

CUDA_VISIBLE_DEVICES=$1 python run_classifier.py \
    --data_dir "./data/${TASK}/" \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir "./output/tmp" \
    --num_train_steps 10000 \
    --learning_rate 2e-05 \
    --warmup_steps 1000 \
    --loss_weight 1.0 \
    --tsa_mode 'linear' \
    --softmax_temp 0.85 \
    --do_eval \
    --eval_steps 100 \
    --save_steps 1000

    