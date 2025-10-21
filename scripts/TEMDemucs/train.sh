accelerate launch --config_file ./config.yaml run.py \
    --mode train \
    --data_dir ./data/raw_data/ \
    --model temdemucs \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-3 \
    --lr_decay 0.98 \
    --lr_step 1 \
    --ckpt_dir ./checkpoints \
