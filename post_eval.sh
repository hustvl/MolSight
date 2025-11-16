#!/bin/bash
#export CUDA_VISIBLE_DEVICES=2,3

NUM_GPUS_PER_NODE=4

BATCH_SIZE=32

LOAD_PATH=runs/stereo_grpo_10_2/ckpt_model/epoch_1.pth

set -x

torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    post_train.py \
    --data_path data \
    --valid_file stereo/val.csv \
    --test_file real/CLEF.csv,real/UOB.csv,real/USPTO.csv,real/staker.csv,real/acs.csv,synthetic/indigo.csv,synthetic/chemdraw.csv \
    --vocab_file vocab/vocab_chars.json \
    --formats grpo \
    --n_coord_bins 128 \
    --input_size 512 \
    --encoder efficientvit \
    --load_path $LOAD_PATH \
    --batch_size $BATCH_SIZE \
    --print_freq 100 \
    --do_valid \
    --lora \
    --num_workers 4 \
    --use_qknorm --use_swiglu --use_rmsnorm \
    --exp_name debug \
    --resume \
    --load_model_only 2>&1
