#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0,1,2,3

NUM_GPUS_PER_NODE=4

BATCH_SIZE=32

LOAD_PATH=runs/pubchem_uspto_smiles_edges/ckpt_model/epoch_9.pth

EXP_NAME=pubchem_coords_mle_wd
LOG_FILE=runs/$EXP_NAME/run_$(date +%F_%H-%M-%S).log

mkdir -p "$(dirname "$LOG_FILE")" || { echo "unable to create log dir" >&2; exit 1; }
touch "$LOG_FILE" || { echo "unable to create log file" >&2; exit 1; }
exec > >(tee -a "$LOG_FILE") 2>&1

set -x

torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    train_loc_predictor.py \
    --data_path data \
    --coords_file aux_file \
    --valid_file real/USPTO.csv \
    --test_file real/CLEF.csv,real/UOB.csv,real/USPTO.csv,real/staker.csv,real/acs.csv,synthetic/indigo.csv,synthetic/chemdraw.csv \
    --vocab_file vocab/vocab_chars.json \
    --formats char,edges,coords \
    --dynamic_indigo \
    --augment \
    --mol_augment \
    --include_condensed \
    --n_coord_bins 128 \
    --input_size 512 \
    --encoder efficientvit \
    --encoder_lr 0 \
    --decoder_lr 0 \
    --predictor_lr 4e-5 \
    --epochs 2 \
    --batch_size $BATCH_SIZE \
    --accum_freq 2 \
    --warmup 0.02 \
    --print_freq 100 \
    --do_train --do_valid --do_test \
    --num_workers 4 \
    --train_datasets pubchem \
    --use_qknorm --use_swiglu --use_rmsnorm \
    --amp \
    --exp_name $EXP_NAME \
    --resume \
    --load_path $LOAD_PATH \
    --weight_decay 1e-4 \
    --load_model_only \
    #--regression \
    #--label_smoothing 0.1
