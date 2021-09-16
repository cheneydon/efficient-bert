#!/bin/bash

NUM_GPU=4
GPU_DEVICES='0,1,2,3'

VOCAB_PATH='./pretrained_ckpt/bert-base-uncased-vocab.txt'
WIKI_DIR='./dataset/pretrain_data/wikipedia_nomask'
BOOK_DIR='./dataset/pretrain_data/bookcorpus_nomask'
CONCATE_DATA_DIR='./dataset/pretrain_data/wiki_book_nomask'

STUDENT_MODEL='supernet'
TEACHER_PRETRAIN_PATH='./pretrained_ckpt/bert-base-uncased-pytorch_model.bin'

PRETRAIN_LR=1e-4
PRETRAIN_TRAIN_RATIO=1
PRETRAIN_EPOCHS=10
PRETRAIN_BS=64
PRETRAIN_EXP_PATH='./exp/pretrain/supernet/'


# Stage 1, 2 (Wikipedia only)
bash dist_pretrain.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --student_model $STUDENT_MODEL --train_ratio $PRETRAIN_TRAIN_RATIO --total_epochs $PRETRAIN_EPOCHS --batch_size $PRETRAIN_BS --lr $PRETRAIN_LR --teacher_pretrain_path $TEACHER_PRETRAIN_PATH --vocab_path $VOCAB_PATH --wiki_dir $WIKI_DIR --exp_dir $PRETRAIN_EXP_PATH

# Stage 3 (Wikipedia + BooksCorpus)
bash dist_pretrain.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --student_model $STUDENT_MODEL --train_ratio $PRETRAIN_TRAIN_RATIO --total_epochs $PRETRAIN_EPOCHS --batch_size $PRETRAIN_BS --lr $PRETRAIN_LR --teacher_pretrain_path $TEACHER_PRETRAIN_PATH --vocab_path $VOCAB_PATH --wiki_dir $WIKI_DIR --book_dir $BOOK_DIR --concate_data_dir $CONCATE_DATA_DIR --exp_dir $PRETRAIN_EXP_PATH
