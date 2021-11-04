#!/bin/bash
TASK="IMDB"

LR=2e-05
WARMUP_STEPS=1000
TRAIN_STEPS=10000
LOSS_WEIGHT=1.0
TEMP=0.85
EVAL_STEPS=500
SAVE_STEPS=500

MODEL_NAME=uda-lr${LR}-step${TRAIN_STEPS}-wu${WARMUP_STEPS}-lw${LOSS_WEIGHT}-t{TEMP}

CUDA_VISIBLE_DEVICES=$1 python run_classifier.py \
    --data_dir "./data/${TASK}/" \
    --model_name_or_path bert-base-uncased \
    --output_dir "./output/${MODEL_NAME}" \
    --num_train_steps $TRAIN_STEPS \
    --learning_rate $LR \
    --warmup_steps $WARMUP_STEPS \
    --loss_weight $LOSS_WEIGHT \
    --tsa_mode 'linear' \
    --softmax_temp $TEMP \
    --do_train \
    --do_eval \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS
    