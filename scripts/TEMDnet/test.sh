accelerate launch --config_file ./config.yaml run.py \
    --mode test \
    --data_dir ./data/raw_data/ \
    --model temdnet \
    --batch_size 32 \
    --stddev 0.05 \
    --load_checkpoint ./checkpoints/temdnet_best.pth \