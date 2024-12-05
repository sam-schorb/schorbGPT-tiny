#!/bin/bash
# run.sh

# H100-specific settings
export NUM_GPUS=8
export BATCH_SIZE=32  # Per GPU, will be multiplied by 8 effectively
export NUM_WORKERS=8  # Per GPU data loader workers

echo "Starting distributed training on $NUM_GPUS H100 GPUs..."
echo "Global batch size will be: $((BATCH_SIZE * NUM_GPUS))"

# Launch distributed training with environment variables
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train.py \
    --batch_size=$BATCH_SIZE \
    --num_workers=$NUM_WORKERS