# UDA Re-implementation

Please refer to original paper [Unsupervised Data Augmentation for Consistency Training (NIPS 2020)](https://arxiv.org/pdf/1904.12848.pdf) and code [google-research/uda](https://github.com/google-research/uda).

## Run code
```
bash train.sh 0
```

```
TASK="IMDB"

CUDA_VISIBLE_DEVICES=$1 python run_classifier.py \
    --data_dir "./data/${TASK}/" \
    --model_name_or_path bert-base-uncased \
    --output_dir "./output/tmp" \
    --num_train_steps 10000 \
    --learning_rate 2e-05 \
    --warmup_steps 1000 \
    --loss_weight 1.0 \
    --softmax_temp 0.85 \
    --tsa_mode 'linear' \
    --do_train \
    --do_eval \
    --eval_steps 100 \
    --save_steps 1000
```

## Reproducing results

### IMDB

Model | # labeled samples (train) | # unlabeled samples (train) | # test samples | Error rate (from paper) | Error rate (re-implementation)
---|---|---|---|---|---
BERT | 20 | 69972 | 25000 | 6.50 | TBA
UDA | 20 | 69972 | 25000 | 4.20 | TBA