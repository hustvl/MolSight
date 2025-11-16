#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0,1,2,3

NUM_GPUS_PER_NODE=4

BATCH_SIZE=16

LOAD_PATH=runs/pubchem_uspto_smiles_edges_aug_30/ckpt_model/epoch_0.pth

set -x

torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    train.py \
    --data_path data \
    --coords_file aux_file \
    --valid_file real/USPTO.csv \
    --test_file real/CLEF.csv,real/UOB.csv,real/USPTO.csv,real/staker.csv,real/acs.csv,synthetic/indigo.csv,synthetic/chemdraw.csv \
    --vocab_file vocab/vocab_chars.json \
    --formats char,edges \
    --dynamic_indigo \
    --augment \
    --mol_augment \
    --include_condensed \
    --n_coord_bins 128 \
    --input_size 512 \
    --encoder efficientvit \
    --encoder_lr 4e-5 \
    --decoder_lr 4e-4 \
    --predictor_lr 4e-4 \
    --epochs 30 \
    --batch_size $BATCH_SIZE \
    --accum_freq 4 \
    --warmup 0.02 \
    --print_freq 100 \
    --do_train --do_valid --do_test \
    --num_workers 4 \
    --train_datasets pubchem,uspto \
    --use_qknorm --use_swiglu --use_rmsnorm \
    --amp \
    --exp_name pubchem_uspto_smiles_edges_aug_30 \
    --resume \
    --load_path $LOAD_PATH \
    --weight_decay 1e-4 \
    --label_smoothing 0.1 2>&1
    #--load_model_only
