accelerate launch --config_file ./config.yaml run.py \
    --mode train \
    --data_dir ./data/raw_data/ \
    --model temdemucs \
    --epochs 100 \
    --batch_size 64 \
    --lr 3e-4 \
    --lr_decay 0.99 \
    --lr_step 1 \
    --ckpt_dir ./checkpoints \
