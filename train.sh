CUDA_VISIBLE_DEVICES=$1 python utils/train.py \
    --data_dir "./data/IMDB/" \
    --model_name_or_path bert-base-uncased \
    --output_dir "./output/tmp" \
    --do_train