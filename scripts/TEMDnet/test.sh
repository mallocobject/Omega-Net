accelerate launch --config_file ./config.yaml run.py \
    --mode test \
    --data_dir ./data/raw_data/ \
    --model temdnet \
    --stddev 0.01 \
    --load_checkpoint ./checkpoints/temdnet_best.pth \