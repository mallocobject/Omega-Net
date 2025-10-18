accelerate launch --config_file ./config.yaml run.py \
    --mode test \
    --data_dir ./data/raw_data/ \
    --model temsgnet \
    --batch_size 32 \
    --time_steps 200 \
    --load_checkpoint ./checkpoints/temsgnet_best.pth \
