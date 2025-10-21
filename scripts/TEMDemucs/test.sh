accelerate launch --config_file ./config.yaml run.py \
    --mode test \
    --data_dir ./data/raw_data/ \
    --model temdemucs \
    --batch_size 32 \
    --load_checkpoint ./checkpoints/temdemucs_best.pth \