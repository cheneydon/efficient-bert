#!/bin/bash

GPU_DEVICES='2'
DATA_DIR='./dataset/glue'
VOCAB_DIR='./pretrained_ckpt/bert-base-uncased-vocab.txt'

MODEL_NAME='auto_bert'
CKPT_PATH='./exp/downstream/auto_bert/'
SAVE_DIR='test_results/auto_bert/'
FFN_EXPR="tanh(linear4_2(gelu(linear4_2(x,wb1)),wb2)) elu(linear2_1(gelu(linear2_1(x,wb1)),wb2)) sigmoid(linear1_2(relu(linear1_2(x,wb1)),wb2)) linear2_2(gelu(linear2_2(max(x,tanh(x)),wb1)),wb2) linear1_2(relu(mul(linear1_2(x,wb1),x)),wb2) leaky_relu(linear2_3(gelu(linear2_3(x,wb1)),wb2))"
EXP_DIR=$CKPT_PATH'/test_logs/'


python test.py --task mnli    --gpu_devices $GPU_DEVICES --lowercase --model_name $MODEL_NAME --data_dir $DATA_DIR --vocab_path $VOCAB_DIR --result_dir $SAVE_DIR --exp_dir $EXP_DIR --ffn_expr $FFN_EXPR --resume_path $CKPT_PATH'/mnli/best_model.bin'
python test.py --task mnli-mm --gpu_devices $GPU_DEVICES --lowercase --model_name $MODEL_NAME --data_dir $DATA_DIR --vocab_path $VOCAB_DIR --result_dir $SAVE_DIR --exp_dir $EXP_DIR --ffn_expr $FFN_EXPR --resume_path $CKPT_PATH'/mnli/best_model.bin'
python test.py --task ax      --gpu_devices $GPU_DEVICES --lowercase --model_name $MODEL_NAME --data_dir $DATA_DIR --vocab_path $VOCAB_DIR --result_dir $SAVE_DIR --exp_dir $EXP_DIR --ffn_expr $FFN_EXPR --resume_path $CKPT_PATH'/mnli/best_model.bin'
python test.py --task qnli    --gpu_devices $GPU_DEVICES --lowercase --model_name $MODEL_NAME --data_dir $DATA_DIR --vocab_path $VOCAB_DIR --result_dir $SAVE_DIR --exp_dir $EXP_DIR --ffn_expr $FFN_EXPR --resume_path $CKPT_PATH'/qnli/best_model.bin'
python test.py --task sst-2   --gpu_devices $GPU_DEVICES --lowercase --model_name $MODEL_NAME --data_dir $DATA_DIR --vocab_path $VOCAB_DIR --result_dir $SAVE_DIR --exp_dir $EXP_DIR --ffn_expr $FFN_EXPR --resume_path $CKPT_PATH'/sst-2/best_model.bin'
python test.py --task cola    --gpu_devices $GPU_DEVICES --lowercase --model_name $MODEL_NAME --data_dir $DATA_DIR --vocab_path $VOCAB_DIR --result_dir $SAVE_DIR --exp_dir $EXP_DIR --ffn_expr $FFN_EXPR --resume_path $CKPT_PATH'/cola/best_model.bin'
python test.py --task sts-b   --gpu_devices $GPU_DEVICES --lowercase --model_name $MODEL_NAME --data_dir $DATA_DIR --vocab_path $VOCAB_DIR --result_dir $SAVE_DIR --exp_dir $EXP_DIR --ffn_expr $FFN_EXPR --resume_path $CKPT_PATH'/sts-b/best_model.bin'
python test.py --task mrpc    --gpu_devices $GPU_DEVICES --lowercase --model_name $MODEL_NAME --data_dir $DATA_DIR --vocab_path $VOCAB_DIR --result_dir $SAVE_DIR --exp_dir $EXP_DIR --ffn_expr $FFN_EXPR --resume_path $CKPT_PATH'/mrpc/best_model.bin'
python test.py --task rte     --gpu_devices $GPU_DEVICES --lowercase --model_name $MODEL_NAME --data_dir $DATA_DIR --vocab_path $VOCAB_DIR --result_dir $SAVE_DIR --exp_dir $EXP_DIR --ffn_expr $FFN_EXPR --resume_path $CKPT_PATH'/rte/best_model.bin'
python test.py --task wnli    --gpu_devices $GPU_DEVICES --lowercase --model_name $MODEL_NAME --data_dir $DATA_DIR --vocab_path $VOCAB_DIR --result_dir $SAVE_DIR --exp_dir $EXP_DIR --ffn_expr $FFN_EXPR --resume_path $CKPT_PATH'/wnli/best_model.bin'
python test.py --task qqp     --gpu_devices $GPU_DEVICES --lowercase --model_name $MODEL_NAME --data_dir $DATA_DIR --vocab_path $VOCAB_DIR --result_dir $SAVE_DIR --exp_dir $EXP_DIR --ffn_expr $FFN_EXPR --resume_path $CKPT_PATH'/qqp/best_model.bin'
