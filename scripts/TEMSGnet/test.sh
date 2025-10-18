accelerate launch --config_file ./config.yaml run.py \
    --mode test \
    --data_dir ./data/raw_data/ \
    --model temsgnet \
    --batch_size 64 \
    --time_steps 1000 \
    --load_checkpoint ./checkpoints/temsgnet_best.pth \
    --start_step 10 \
