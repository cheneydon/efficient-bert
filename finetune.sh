#!/bin/bash

NUM_GPU=4
GPU_DEVICES='0,1,2,3'

VOCAB_PATH='./pretrained_ckpt/bert-base-uncased-vocab.txt'
GLUE_DIR='./dataset/glue'
SQUAD_DIR='./dataset/squad'

STUDENT_PRETRAIN_PATH='./pretrained_ckpt/2nd_General_TinyBERT_4L_312D/pytorch_model.bin'
TEACHER_PRETRAIN_PATH_MNLI='./exp/train/bert_base/mnli/best_model.bin'
TEACHER_PRETRAIN_PATH_QQP='./exp/train/bert_base/qqp/best_model.bin'
TEACHER_PRETRAIN_PATH_QNLI='./exp/train/bert_base/qnli/best_model.bin'
TEACHER_PRETRAIN_PATH_SST2='./exp/train/bert_base/sst-2/best_model.bin'
TEACHER_PRETRAIN_PATH_COLA='./exp/train/bert_base/cola/best_model.bin'
TEACHER_PRETRAIN_PATH_STSB='./exp/train/bert_base/sts-b/best_model.bin'
TEACHER_PRETRAIN_PATH_MRPC='./exp/train/bert_base/mrpc/best_model.bin'
TEACHER_PRETRAIN_PATH_RTE='./exp/train/bert_base/rte/best_model.bin'
TEACHER_PRETRAIN_PATH_WNLI='./exp/train/bert_base/wnli/best_model.bin'
TEACHER_PRETRAIN_PATH_SQUAD1='./exp/train/bert_base/squad1.1/best_model.bin'
TEACHER_PRETRAIN_PATH_SQUAD2='./exp/train/bert_base/squad2.0/best_model.bin'

GLUE_SEQ_LEN=128
GLUE_BS=8
GLUE_EPOCHS=10
GLUE_EPOCHS_COLA=50
GLUE_LR=5e-5

SQUAD_SEQ_LEN=128
SQUAD_QLEN=64
SQUAD_TRUNCATE_STRIDE=32
SQUAD_BS=8
SQUAD_EPOCHS=10
SQUAD_LR=1e-4

HIDDEN_RATIO=1
PRED_RATIO=1

STUDENT_MODEL='tiny_bert'
EXP_DIR='./exp/downstream/tiny_bert/'


# GLUE
bash dist_finetune.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --student_model $STUDENT_MODEL --task mnli      --val_freq 50  --lr $GLUE_LR  --batch_size $GLUE_BS  --total_epochs $GLUE_EPOCHS      --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR  --max_seq_len $GLUE_SEQ_LEN  --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH_MNLI --student_pretrain_path $STUDENT_PRETRAIN_PATH --exp_dir $EXP_DIR
bash dist_finetune.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --student_model $STUDENT_MODEL --task qnli      --val_freq 50  --lr $GLUE_LR  --batch_size $GLUE_BS  --total_epochs $GLUE_EPOCHS      --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR  --max_seq_len $GLUE_SEQ_LEN  --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH_QNLI --student_pretrain_path $STUDENT_PRETRAIN_PATH --exp_dir $EXP_DIR
bash dist_finetune.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --student_model $STUDENT_MODEL --task sst-2     --val_freq 50  --lr $GLUE_LR  --batch_size $GLUE_BS  --total_epochs $GLUE_EPOCHS      --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR  --max_seq_len $GLUE_SEQ_LEN  --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH_SST2 --student_pretrain_path $STUDENT_PRETRAIN_PATH --exp_dir $EXP_DIR
bash dist_finetune.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --student_model $STUDENT_MODEL --task cola      --val_freq 50  --lr $GLUE_LR  --batch_size $GLUE_BS  --total_epochs $GLUE_EPOCHS_COLA --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR  --max_seq_len $GLUE_SEQ_LEN  --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH_COLA --student_pretrain_path $STUDENT_PRETRAIN_PATH --exp_dir $EXP_DIR
bash dist_finetune.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --student_model $STUDENT_MODEL --task sts-b     --val_freq 50  --lr $GLUE_LR  --batch_size $GLUE_BS  --total_epochs $GLUE_EPOCHS      --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR  --max_seq_len $GLUE_SEQ_LEN  --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH_STSB --student_pretrain_path $STUDENT_PRETRAIN_PATH --exp_dir $EXP_DIR
bash dist_finetune.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --student_model $STUDENT_MODEL --task mrpc      --val_freq 50  --lr $GLUE_LR  --batch_size $GLUE_BS  --total_epochs $GLUE_EPOCHS      --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR  --max_seq_len $GLUE_SEQ_LEN  --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH_MRPC --student_pretrain_path $STUDENT_PRETRAIN_PATH --exp_dir $EXP_DIR
bash dist_finetune.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --student_model $STUDENT_MODEL --task rte       --val_freq 50  --lr $GLUE_LR  --batch_size $GLUE_BS  --total_epochs $GLUE_EPOCHS      --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR  --max_seq_len $GLUE_SEQ_LEN  --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH_RTE  --student_pretrain_path $STUDENT_PRETRAIN_PATH --exp_dir $EXP_DIR
bash dist_finetune.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --student_model $STUDENT_MODEL --task wnli      --val_freq 50  --lr $GLUE_LR  --batch_size $GLUE_BS  --total_epochs $GLUE_EPOCHS      --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR  --max_seq_len $GLUE_SEQ_LEN  --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH_WNLI --student_pretrain_path $STUDENT_PRETRAIN_PATH --exp_dir $EXP_DIR
bash dist_finetune.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --student_model $STUDENT_MODEL --task qqp       --val_freq 250 --lr $GLUE_LR  --batch_size $GLUE_BS  --total_epochs $GLUE_EPOCHS      --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR  --max_seq_len $GLUE_SEQ_LEN  --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH_QQP  --student_pretrain_path $STUDENT_PRETRAIN_PATH --exp_dir $EXP_DIR
bash dist_finetune.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --student_model $STUDENT_MODEL --task squad1.1  --val_freq 500 --lr $SQUAD_LR --batch_size $SQUAD_BS --total_epochs $SQUAD_EPOCHS     --vocab_path $VOCAB_PATH --data_dir $SQUAD_DIR --max_seq_len $SQUAD_SEQ_LEN --max_query_len $SQUAD_QLEN --trunc_stride $SQUAD_TRUNCATE_STRIDE --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH_SQUAD1 --student_pretrain_path $STUDENT_PRETRAIN_PATH --exp_dir $EXP_DIR
bash dist_finetune.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --student_model $STUDENT_MODEL --task squad2.0  --val_freq 500 --lr $SQUAD_LR --batch_size $SQUAD_BS --total_epochs $SQUAD_EPOCHS     --vocab_path $VOCAB_PATH --data_dir $SQUAD_DIR --max_seq_len $SQUAD_SEQ_LEN --max_query_len $SQUAD_QLEN --trunc_stride $SQUAD_TRUNCATE_STRIDE --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH_SQUAD2 --student_pretrain_path $STUDENT_PRETRAIN_PATH --exp_dir $EXP_DIR

# SQuAD
bash dist_finetune.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --official_pretrain --student_model $STUDENT_MODEL --task squad2.0  --val_freq 500 --lr $SQUAD_LR --batch_size $SQUAD_BS --total_epochs $SQUAD_EPOCHS     --vocab_path $VOCAB_PATH --data_dir $SQUAD_DIR --max_seq_len $SQUAD_SEQ_LEN --max_query_len $SQUAD_QLEN --trunc_stride $SQUAD_TRUNCATE_STRIDE --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH_SQUAD2 --student_pretrain_path $STUDENT_PRETRAIN_PATH --exp_dir $EXP_DIR
bash dist_finetune.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --official_pretrain --student_model $STUDENT_MODEL --task squad1.1  --val_freq 500 --lr $SQUAD_LR --batch_size $SQUAD_BS --total_epochs $SQUAD_EPOCHS     --vocab_path $VOCAB_PATH --data_dir $SQUAD_DIR --max_seq_len $SQUAD_SEQ_LEN --max_query_len $SQUAD_QLEN --trunc_stride $SQUAD_TRUNCATE_STRIDE --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH_SQUAD1 --student_pretrain_path $STUDENT_PRETRAIN_PATH --exp_dir $EXP_DIR
