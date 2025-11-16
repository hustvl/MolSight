#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0,1,2,3

NUM_GPUS_PER_NODE=4

BATCH_SIZE=8

LOAD_PATH=runs/pubchem_uspto_smiles_edges/ckpt_model/epoch_9.pth

EXP_NAME=stereo_grpo_10_2
LOG_FILE=runs/$EXP_NAME/run_$(date +%F_%H-%M-%S).log

mkdir -p "$(dirname "$LOG_FILE")" || { echo "unable to create log dir" >&2; exit 1; }
touch "$LOG_FILE" || { echo "unable to create log file" >&2; exit 1; }
exec > >(tee -a "$LOG_FILE") 2>&1

set -x

torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    post_train.py \
    --data_path data \
    --coords_file aux_file \
    --valid_file stereo/val.csv \
    --test_file real/CLEF.csv,real/UOB.csv,real/USPTO.csv,real/staker.csv,real/acs.csv,synthetic/indigo.csv,synthetic/chemdraw.csv \
    --vocab_file vocab/vocab_chars.json \
    --formats grpo \
    --augment \
    --n_coord_bins 128 \
    --input_size 512 \
    --encoder efficientvit \
    --encoder_lr 0 \
    --decoder_lr 1e-4 \
    --predictor_lr 0 \
    --epochs 2 \
    --batch_size $BATCH_SIZE \
    --accum_freq 2 \
    --warmup 0.02 \
    --print_freq 100 \
    --do_train --do_valid --do_test \
    --lora \
    --num_workers 4 \
    --train_datasets stereo \
    --use_qknorm --use_swiglu --use_rmsnorm \
    --exp_name $EXP_NAME \
    --resume \
    --load_path $LOAD_PATH \
    --weight_decay 0 \
    --load_model_only \
    --smiles_only \
    --label_smoothing 0.1
