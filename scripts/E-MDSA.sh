#!/usr/bin/env bash

source $1
gpu_id=0
export PYTHONPATH="."
cmd="python code/run_augment_test_volatility_single_tracin_zero_shot.py \
        --num_epochs $num_epochs \
        --batch_size $batch_size\
        --lr $lr \
        --label_size $label_size \
        --embedding_size $embedding_size \
        --vocab_size $vocab_size \
        --max_length $max_length \
        --n_heads $n_heads \
        --depth $depth \
        --seed $RANDOM \
        --lr_warmup $lr_warmup \
        --gradient_clipping $gradient_clipping \
        --alpha $alpha \
        --task $task \
        --idx $idx \
        --cuda
        --train_pkl $train_pkl \
        --dev_pkl $dev_pkl \
        --test_pkl $test_pkl \
        --model $model
        --kl $kl"
        
echo $cmd
CUDA_VISIBLE_DEVICES=$gpu_id $cmd

#       --lr=0.0002 \
#        --paras_times=1 \
#         --sents_times=1 \

#         --train_pkl='train/train_opening.pkl' \
#         --dev_pkl='dev/dev_opening.pkl' \
#         --test_pkl='test/test_opening.pkl' \
        
#         --train_pkl='train/train_qa.pkl' \
#         --dev_pkl='dev/dev_qa.pkl' \
#         --test_pkl='test/test_qa.pkl' \
        
#         --train_pkl='train/train_opening_qa.pkl' \
#         --dev_pkl='dev/dev_opening_qa.pkl' \
#         --test_pkl='test/test_opening_qa.pkl' \
        
#         --train_pkl='train/train_opening_yang.pkl' \
#         --dev_pkl='dev/dev_opening_yang.pkl' \
#         --test_pkl='test/test_opening_yang.pkl' \
        
#         --train_pkl='train/train_qa_yang.pkl' \
#         --dev_pkl='dev/dev_qa_yang.pkl' \
#         --test_pkl='test/test_qa_yang.pkl' \
        
#         --train_pkl='train/train_opening_qa_yang.pkl' \
#         --dev_pkl='dev/dev_opening_qa_yang.pkl' \
#         --test_pkl='test/test_opening_qa_yang.pkl' \
