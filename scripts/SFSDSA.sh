accelerate launch --config_file ./config.yaml train.py \
    --data_dir data/raw_data/ \
    --model sfsdsa \
    --epochs 222 \
    --batch_size 32 \
    --lr 1e-3 \
    --regularizer 0.15 \
    --stddev 0.3 \
    --ckpt_dir checkpoints \
