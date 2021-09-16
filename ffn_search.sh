#!/bin/bash

NUM_GPU=2
GPU_DEVICES='2,3'

EXP_DIR='./exp/ffn_search/'
VOCAB_PATH='./pretrained_ckpt/bert-base-uncased-vocab.txt'
GLUE_DIR='./dataset/glue'

PRETRAIN_DIR1='./dataset/pretrain_data/wikipedia_nomask'
PRETRAIN_DIR2='./dataset/pretrain_data/wiki_book_nomask'
WIKI_DIR='./dataset/pretrain_data/wikipedia_nomask'
BOOK_DIR='./dataset/pretrain_data/bookcorpus_nomask'

TEACHER_PRETRAIN_PATH='./pretrained_ckpt/bert-base-uncased-pytorch_model.bin'
TEACHER_DOWNSTREAM_PATH1='./exp/train/bert_base/mnli/best_model.bin'
STUDENT_PRETRAIN_PATH1='./exp/pretrain/supernet/stage1_2/ckpt_ep10.bin'  # Pre-trained supernet checkpoint for search stage 1, 2 (Wikipedia only)
STUDENT_PRETRAIN_PATH2='./exp/pretrain/supernet/stage3/ckpt_ep10.bin'  # Pre-trained supernet checkpoint for search stage 3 (entirely pre-trained w/o weight sharing, Wikipedia + BooksCorpus)
STUDENT_PRETRAIN_PATH3='./exp/ffn_search/pretrain/pretrain_ckpt_ep10.bin'  # Pre-trained supernet checkpoint for search stage 3 (with weight sharing, Wikipedia + BooksCorpus)
STUDENT_DOWNSTREAM_PATH='./exp/ffn_search/downstream/downstream_ckpt_ep10.bin'  # Fine-tuned supernet checkpoint for search stage 3 (with weight sharing)


# Search stage 1
bash dist_ffn_search.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --exp_dir $EXP_DIR --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR --pretrain_dir1 $PRETRAIN_DIR1 --pretrain_dir2 $PRETRAIN_DIR2 --wiki_dir $WIKI_DIR --book_dir $BOOK_DIR --teacher_pretrain_path $TEACHER_PRETRAIN_PATH --teacher_downstream_path1 $TEACHER_DOWNSTREAM_PATH1 --student_pretrain_path1 $STUDENT_PRETRAIN_PATH1 --student_pretrain_path2 $STUDENT_PRETRAIN_PATH2 --student_pretrain_path3 $STUDENT_PRETRAIN_PATH3 --student_downstream_path $STUDENT_DOWNSTREAM_PATH


# Search stage 2
bash dist_ffn_search.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --stage2 --exp_dir $EXP_DIR --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR --pretrain_dir1 $PRETRAIN_DIR1 --pretrain_dir2 $PRETRAIN_DIR2 --wiki_dir $WIKI_DIR --book_dir $BOOK_DIR --teacher_pretrain_path $TEACHER_PRETRAIN_PATH --teacher_downstream_path1 $TEACHER_DOWNSTREAM_PATH1 --student_pretrain_path1 $STUDENT_PRETRAIN_PATH1 --student_pretrain_path2 $STUDENT_PRETRAIN_PATH2 --student_pretrain_path3 $STUDENT_PRETRAIN_PATH3 --student_downstream_path $STUDENT_DOWNSTREAM_PATH


# Search stage 3
# Pre-train with weight sharing (modify the sub-dir name into $STUDENT_PRETRAIN_PATH3 after training)
bash dist_ffn_search.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --stage3 --exp_dir $EXP_DIR --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR --pretrain_dir1 $PRETRAIN_DIR1 --pretrain_dir2 $PRETRAIN_DIR2 --wiki_dir $WIKI_DIR --book_dir $BOOK_DIR --teacher_pretrain_path $TEACHER_PRETRAIN_PATH --teacher_downstream_path1 $TEACHER_DOWNSTREAM_PATH1 --student_pretrain_path2 $STUDENT_PRETRAIN_PATH2

# Fine-tune with weight sharing (modify the sub-dir name into $STUDENT_DOWNSTREAM_PATH after training)
bash dist_ffn_search.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --stage3 --exp_dir $EXP_DIR --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR --pretrain_dir1 $PRETRAIN_DIR1 --pretrain_dir2 $PRETRAIN_DIR2 --wiki_dir $WIKI_DIR --book_dir $BOOK_DIR --teacher_pretrain_path $TEACHER_PRETRAIN_PATH --teacher_downstream_path1 $TEACHER_DOWNSTREAM_PATH1 --student_pretrain_path2 $STUDENT_PRETRAIN_PATH2 --student_pretrain_path3 $STUDENT_PRETRAIN_PATH3

# Architecture search
bash dist_ffn_search.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --stage3 --exp_dir $EXP_DIR --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR --pretrain_dir1 $PRETRAIN_DIR1 --pretrain_dir2 $PRETRAIN_DIR2 --wiki_dir $WIKI_DIR --book_dir $BOOK_DIR --teacher_pretrain_path $TEACHER_PRETRAIN_PATH --teacher_downstream_path1 $TEACHER_DOWNSTREAM_PATH1 --student_pretrain_path2 $STUDENT_PRETRAIN_PATH2 --student_pretrain_path3 $STUDENT_PRETRAIN_PATH3 --student_downstream_path $STUDENT_DOWNSTREAM_PATH
