accelerate launch --config_file ./config.yaml run.py \
    --mode train \
    --data_dir ./data/raw_data/ \
    --model sfsdsa \
    --epochs 2 \
    --batch_size 8 \
    --lr 1e-3 \
    --regularizer 0.15 \
    --ckpt_dir ./checkpoints \

