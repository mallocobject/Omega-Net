accelerate launch --config_file ./config.yaml train.py \
    --data_dir dataset/ \
    --model temdnet \
    --epochs 222 \
    --batch_size 128 \
    --lr 1e-3 \
    --lr_decay 0.98 \
    --lr_step 10 \
    --stddev 0.01 \
    --ckpt_dir checkpoints \
