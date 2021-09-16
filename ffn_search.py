import os
import re
import json
import numpy as np
import torch
import torch.nn as nn
import logging
import argparse
import time
import datetime
import random
from deap import gp
from pathlib import Path
from apex.parallel import DistributedDataParallel as DDP
from datasets import glue_train_tasks
from models import select_config, select_model
from tokenizers import select_tokenizer
from metrics import compute_glue_metrics, all_glue_select_metrics
from utils import AverageMeter, register_custom_ops, register_custom_ops2, SearchPhase, \
    set_seeds, setup_logger, reduce_tensor, calc_params, soft_cross_entropy, load_pretrain_state_dict, \
    load_multi_task_state_dict, create_optimizer, create_scheduler, create_split_dataset, create_dataset, \
    create_pretrain_dataset, create_multi_task_dataset, save_checkpoint, get_entire_linear_idx, get_entire_params

parser = argparse.ArgumentParser()
parser.add_argument('--distributed', action='store_true', help='distributed mode')
parser.add_argument('--gpu_devices', default='0,1,2,3', type=str, help='available gpu devices')
parser.add_argument('--seed', default=42, type=int, help='seed')

parser.add_argument('--lowercase', action='store_true', help='whether to do lowercase')
parser.add_argument('--stage2', action='store_true', help='whether search for stage 2')
parser.add_argument('--stage3', action='store_true', help='whether search for stage 3')
parser.add_argument('--sce_temp', default=1, type=float, help='temperature for soft cross entropy loss')
parser.add_argument('--hidden_ratio', default=1, type=float, help='ratio for hidden loss')
parser.add_argument('--pred_ratio', default=1, type=float, help='ratio for prediction loss')
parser.add_argument('--teacher_model1', default='bert_base', type=str, help='teacher model name for stage 1 and 2')
parser.add_argument('--teacher_model2', default='mt_bert_base', type=str, help='teacher model name for stage 3')
parser.add_argument('--student_model1', default='auto_bert', type=str, help='student model name for stage 1 and 2')
parser.add_argument('--student_model2', default='mt_auto_bert', type=str, help='student model name for stage 3')

parser.add_argument('--task', default='mnli', type=str, help='task name')
parser.add_argument('--data_dir', default='', type=str, help='task dataset directory')
parser.add_argument('--vocab_path', default='', type=str, help='path to pretrained vocabulary file')
parser.add_argument('--merge_path', default='', type=str, help='path to pretrained merge file (for roberta)')
parser.add_argument('--max_seq_len', default=128, type=int, help='max length of input sequences')
parser.add_argument('--max_query_len', default=64, type=int, help='max length of input questions (for squad) or question-answer pairs (for multi-choice tasks)')
parser.add_argument('--trunc_stride', default=32, type=int, help='context truncate stride (for squad)')
parser.add_argument('--n_best_size', default=20, type=int, help='total number of top-n best predictions to generate (for squad)')
parser.add_argument('--max_answer_len', default=30, type=int, help='maximum length of an answer that can be generated (for squad)')
parser.add_argument('--null_score_diff_threshold', default=0, type=float, help='if null_score - best_non_null is greater than the threshold predict null (for squad)')

parser.add_argument('--min_height', default=3, type=int, help='min height of GP tree')
parser.add_argument('--max_height', default=7, type=int, help='max height of GP tree')
parser.add_argument('--min_params', default=15, type=float, help='min params to search')
parser.add_argument('--max_params', default=16, type=float, help='max params to search')
parser.add_argument('--n_init_samples', default=80, type=int, help='num of initial samples to train action space')
parser.add_argument('--train_interval', default=10, type=int, help='num sample interval to train action space')
parser.add_argument('--n_total_samples', default=1000000, type=int, help='num of total samples to search')

parser.add_argument('--train_ratio1', default=0.02, type=float, help='pretrain train ratio for stage 1 and 2')
parser.add_argument('--train_ratio2', default=0.1, type=float, help='downstream train ratio for stage 1 and 2')
parser.add_argument('--train_ratio3', default=1, type=float, help='pretrain train ratio for stage 3')
parser.add_argument('--train_ratio4', default=0.9, type=float, help='downstream train ratio for stage 3')
parser.add_argument('--val_ratio1', default=0, type=float, help='pretrain val ratio for stage 1 and 2')
parser.add_argument('--val_ratio2', default=0.01, type=float, help='downstream val ratio for stage 1 and 2')
parser.add_argument('--val_ratio3', default=0, type=float, help='pretrain val ratio for stage 3')
parser.add_argument('--val_ratio4', default=0.1, type=float, help='downstream val ratio for stage 3')
parser.add_argument('--start_epoch1', default=1, type=int, help='pretrain start epoch for stage 1 and 2 (default is 1)')
parser.add_argument('--start_epoch2', default=1, type=int, help='downstream start epoch for stage 1 and 2 (default is 1)')
parser.add_argument('--start_epoch3', default=1, type=int, help='pretrain start epoch for stage 3 (default is 1)')
parser.add_argument('--start_epoch4', default=1, type=int, help='downstream start epoch for stage 3 (default is 1)')
parser.add_argument('--total_epochs1', default=1, type=int, help='total pretrain epochs for stage 1 and 2')
parser.add_argument('--total_epochs2', default=3, type=int, help='total downstream epochs for stage 1 and 2')
parser.add_argument('--total_epochs3', default=10, type=int, help='total pretrain epochs for stage 3')
parser.add_argument('--total_epochs4', default=10, type=int, help='total downstream epochs for stage 3')
parser.add_argument('--batch_size1', default=64, type=int, help='pretrain batch size for stage 1 and 2')
parser.add_argument('--batch_size2', default=64, type=int, help='downstream batch size for stage 1 and 2')
parser.add_argument('--batch_size3', default=64, type=int, help='pretrain batch size for stage 3')
parser.add_argument('--batch_size4', default=64, type=int, help='downstream batch size for stage 3')
parser.add_argument('--lr1', default=1e-4, type=float, help='initial pretrain learning rate for stage 1 and 2')
parser.add_argument('--lr2', default=4e-4, type=float, help='initial downstream learning rate for stage 1 and 2')
parser.add_argument('--lr3', default=1e-4, type=float, help='initial pretrain learning rate for stage 3')
parser.add_argument('--lr4', default=4e-4, type=float, help='initial downstream learning rate for stage 3')
parser.add_argument('--optim_type', default='adamw', type=str, help='optimizer type')
parser.add_argument('--sched_type', default='step', type=str, help='lr scheduler type')
parser.add_argument('--warmup_proportion', default=0.1, type=float, help='proportion of warmup steps')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay')
parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max gradient norm')
parser.add_argument('--loss_disp_freq', default=50, type=int, help='loss display frequency')
parser.add_argument('--ind_disp_freq', default=10, type=int, help='top individuals display frequency')
parser.add_argument('--val_freq1', default=1000000, type=int, help='validate frequency for stage 1 and 2')
parser.add_argument('--val_freq2', default=500, type=int, help='validate frequency for stage 3')
parser.add_argument('--save_freq', default=1, type=int, help='checkpoint save frequency for stage 3')
parser.add_argument('--ckpt_keep_num', default=1, type=int, help='max number of checkpoint files to keep for stage 3')

parser.add_argument('--teacher_pretrain_path', default='', type=str, help='path to pretrained teacher state dict for all stages')
parser.add_argument('--teacher_downstream_path1', default='', type=str, help='path to finetuned downstream teacher state dict for stage 1 and 2')
parser.add_argument('--teacher_downstream_path2', default='', type=str, help='path to finetuned downstream teacher state dict for stage 3')
parser.add_argument('--student_pretrain_path1', default='', type=str, help='path to pretrained student state dict for stage 1 and 2')
parser.add_argument('--student_pretrain_path2', default='', type=str, help='path to pretrained student state dict for stage 3 (entire)')
parser.add_argument('--student_pretrain_path3', default='', type=str, help='path to pretrained student state dict for stage 3 (slice)')
parser.add_argument('--student_downstream_path', default='', type=str, help='path to finetuned downstream student state dict for stage 3')
parser.add_argument('--pretrain_dir1', default='', type=Path, help='directory to train dataset for stage 1 and 2')
parser.add_argument('--pretrain_dir2', default='', type=Path, help='directory to train dataset for stage 3')
parser.add_argument('--wiki_dir', default='', type=Path, help='directory to wikipedia dataset')
parser.add_argument('--book_dir', default='', type=Path, help='directory to bookcorpus dataset')
parser.add_argument('--cache_dir', default='./cache', type=str, help='cache directory to save processed dataset')
parser.add_argument('--fixed_mat_dir', default='./fixed_mat', type=str, help='fixed matrix directory')
parser.add_argument('--exp_dir', default='./exp/tmp/', type=str, help='experiment directory')
parser.add_argument('--local_rank', default=0, type=int, help='DDP local rank')
parser.add_argument('--world_size', default=1, type=int, help='DDP world size')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices


init_expr = [
    'linear1_1(gelu(linear1_1(x,wb1)),wb2)', 'linear1_2(gelu(linear1_2(x,wb1)),wb2)', 'linear1_3(gelu(linear1_3(x,wb1)),wb2)', 'linear1_4(gelu(linear1_4(x,wb1)),wb2)',
    'linear2_1(gelu(linear2_1(x,wb1)),wb2)', 'linear2_2(gelu(linear2_2(x,wb1)),wb2)', 'linear2_3(gelu(linear2_3(x,wb1)),wb2)', 'linear2_4(gelu(linear2_4(x,wb1)),wb2)',
    'linear3_1(gelu(linear3_1(x,wb1)),wb2)', 'linear3_2(gelu(linear3_2(x,wb1)),wb2)', 'linear3_3(gelu(linear3_3(x,wb1)),wb2)', 'linear3_4(gelu(linear3_4(x,wb1)),wb2)',
    'linear4_1(gelu(linear4_1(x,wb1)),wb2)', 'linear4_2(gelu(linear4_2(x,wb1)),wb2)', 'linear4_3(gelu(linear4_3(x,wb1)),wb2)', 'linear4_4(gelu(linear4_4(x,wb1)),wb2)',
]

if args.stage2:
    init_expr = [
        'tanh(linear4_2(gelu(linear4_2(x,wb1)),wb2)), linear2_1(gelu(linear2_1(x,wb1)),wb2), sigmoid(linear1_4(gelu(linear1_4(x,wb1)),wb2)), linear4_4(gelu(linear4_4(x,wb1)),wb2), linear1_3(gelu(linear1_3(x,wb1)),wb2), leaky_relu(linear2_4(gelu(linear2_4(x,wb1)),wb2))',
        'linear4_2(gelu(linear4_2(x,wb1)),wb2), linear2_1(gelu(linear2_1(x,wb1)),wb2), linear1_4(gelu(linear1_4(x,wb1)),wb2), linear4_4(gelu(linear4_4(x,wb1)),wb2), linear1_3(gelu(linear1_3(x,wb1)),wb2), linear2_4(gelu(linear2_4(x,wb1)),wb2)',
    ]

if args.stage3:
    init_expr = [
        'tanh(linear4_2(gelu(linear4_2(x,wb1)),wb2)), elu(linear2_1(gelu(linear2_1(x,wb1)),wb2)), sigmoid(linear1_4(relu(linear1_4(x,wb1)),wb2)), linear4_4(gelu(linear4_4(max(x,tanh(x)),wb1)),wb2), linear1_3(relu(mul(linear1_3(x,wb1),x)),wb2), leaky_relu(linear2_4(gelu(linear2_4(x,wb1)),wb2))',
        'tanh(linear1_1(gelu(linear1_1(x,wb1)),wb2)), elu(linear1_1(gelu(linear1_1(x,wb1)),wb2)), sigmoid(linear1_1(relu(linear1_1(x,wb1)),wb2)), linear1_1(gelu(linear1_1(max(x,tanh(x)),wb1)),wb2), linear1_1(relu(mul(linear1_1(x,wb1),x)),wb2), leaky_relu(linear1_1(gelu(linear1_1(x,wb1)),wb2))',
        'tanh(linear1_2(gelu(linear1_2(x,wb1)),wb2)), elu(linear1_2(gelu(linear1_2(x,wb1)),wb2)), sigmoid(linear1_2(relu(linear1_2(x,wb1)),wb2)), linear1_2(gelu(linear1_2(max(x,tanh(x)),wb1)),wb2), linear1_2(relu(mul(linear1_2(x,wb1),x)),wb2), leaky_relu(linear1_2(gelu(linear1_2(x,wb1)),wb2))',
        'tanh(linear1_3(gelu(linear1_3(x,wb1)),wb2)), elu(linear1_3(gelu(linear1_3(x,wb1)),wb2)), sigmoid(linear1_3(relu(linear1_3(x,wb1)),wb2)), linear1_3(gelu(linear1_3(max(x,tanh(x)),wb1)),wb2), linear1_3(relu(mul(linear1_3(x,wb1),x)),wb2), leaky_relu(linear1_3(gelu(linear1_3(x,wb1)),wb2))',
        'tanh(linear1_4(gelu(linear1_4(x,wb1)),wb2)), elu(linear1_4(gelu(linear1_4(x,wb1)),wb2)), sigmoid(linear1_4(relu(linear1_4(x,wb1)),wb2)), linear1_4(gelu(linear1_4(max(x,tanh(x)),wb1)),wb2), linear1_4(relu(mul(linear1_4(x,wb1),x)),wb2), leaky_relu(linear1_4(gelu(linear1_4(x,wb1)),wb2))',
        'tanh(linear2_1(gelu(linear2_1(x,wb1)),wb2)), elu(linear2_1(gelu(linear2_1(x,wb1)),wb2)), sigmoid(linear2_1(relu(linear2_1(x,wb1)),wb2)), linear2_1(gelu(linear2_1(max(x,tanh(x)),wb1)),wb2), linear2_1(relu(mul(linear2_1(x,wb1),x)),wb2), leaky_relu(linear2_1(gelu(linear2_1(x,wb1)),wb2))',
        'tanh(linear2_2(gelu(linear2_2(x,wb1)),wb2)), elu(linear2_2(gelu(linear2_2(x,wb1)),wb2)), sigmoid(linear2_2(relu(linear2_2(x,wb1)),wb2)), linear2_2(gelu(linear2_2(max(x,tanh(x)),wb1)),wb2), linear2_2(relu(mul(linear2_2(x,wb1),x)),wb2), leaky_relu(linear2_2(gelu(linear2_2(x,wb1)),wb2))',
        'tanh(linear2_3(gelu(linear2_3(x,wb1)),wb2)), elu(linear2_3(gelu(linear2_3(x,wb1)),wb2)), sigmoid(linear2_3(relu(linear2_3(x,wb1)),wb2)), linear2_3(gelu(linear2_3(max(x,tanh(x)),wb1)),wb2), linear2_3(relu(mul(linear2_3(x,wb1),x)),wb2), leaky_relu(linear2_3(gelu(linear2_3(x,wb1)),wb2))',
        'tanh(linear2_4(gelu(linear2_4(x,wb1)),wb2)), elu(linear2_4(gelu(linear2_4(x,wb1)),wb2)), sigmoid(linear2_4(relu(linear2_4(x,wb1)),wb2)), linear2_4(gelu(linear2_4(max(x,tanh(x)),wb1)),wb2), linear2_4(relu(mul(linear2_4(x,wb1),x)),wb2), leaky_relu(linear2_4(gelu(linear2_4(x,wb1)),wb2))',
        'tanh(linear3_1(gelu(linear3_1(x,wb1)),wb2)), elu(linear3_1(gelu(linear3_1(x,wb1)),wb2)), sigmoid(linear3_1(relu(linear3_1(x,wb1)),wb2)), linear3_1(gelu(linear3_1(max(x,tanh(x)),wb1)),wb2), linear3_1(relu(mul(linear3_1(x,wb1),x)),wb2), leaky_relu(linear3_1(gelu(linear3_1(x,wb1)),wb2))',
        'tanh(linear3_2(gelu(linear3_2(x,wb1)),wb2)), elu(linear3_2(gelu(linear3_2(x,wb1)),wb2)), sigmoid(linear3_2(relu(linear3_2(x,wb1)),wb2)), linear3_2(gelu(linear3_2(max(x,tanh(x)),wb1)),wb2), linear3_2(relu(mul(linear3_2(x,wb1),x)),wb2), leaky_relu(linear3_2(gelu(linear3_2(x,wb1)),wb2))',
        'tanh(linear3_3(gelu(linear3_3(x,wb1)),wb2)), elu(linear3_3(gelu(linear3_3(x,wb1)),wb2)), sigmoid(linear3_3(relu(linear3_3(x,wb1)),wb2)), linear3_3(gelu(linear3_3(max(x,tanh(x)),wb1)),wb2), linear3_3(relu(mul(linear3_3(x,wb1),x)),wb2), leaky_relu(linear3_3(gelu(linear3_3(x,wb1)),wb2))',
        'tanh(linear3_4(gelu(linear3_4(x,wb1)),wb2)), elu(linear3_4(gelu(linear3_4(x,wb1)),wb2)), sigmoid(linear3_4(relu(linear3_4(x,wb1)),wb2)), linear3_4(gelu(linear3_4(max(x,tanh(x)),wb1)),wb2), linear3_4(relu(mul(linear3_4(x,wb1),x)),wb2), leaky_relu(linear3_4(gelu(linear3_4(x,wb1)),wb2))',
        'tanh(linear4_1(gelu(linear4_1(x,wb1)),wb2)), elu(linear4_1(gelu(linear4_1(x,wb1)),wb2)), sigmoid(linear4_1(relu(linear4_1(x,wb1)),wb2)), linear4_1(gelu(linear4_1(max(x,tanh(x)),wb1)),wb2), linear4_1(relu(mul(linear4_1(x,wb1),x)),wb2), leaky_relu(linear4_1(gelu(linear4_1(x,wb1)),wb2))',
        'tanh(linear4_2(gelu(linear4_2(x,wb1)),wb2)), elu(linear4_2(gelu(linear4_2(x,wb1)),wb2)), sigmoid(linear4_2(relu(linear4_2(x,wb1)),wb2)), linear4_2(gelu(linear4_2(max(x,tanh(x)),wb1)),wb2), linear4_2(relu(mul(linear4_2(x,wb1),x)),wb2), leaky_relu(linear4_2(gelu(linear4_2(x,wb1)),wb2))',
        'tanh(linear4_3(gelu(linear4_3(x,wb1)),wb2)), elu(linear4_3(gelu(linear4_3(x,wb1)),wb2)), sigmoid(linear4_3(relu(linear4_3(x,wb1)),wb2)), linear4_3(gelu(linear4_3(max(x,tanh(x)),wb1)),wb2), linear4_3(relu(mul(linear4_3(x,wb1),x)),wb2), leaky_relu(linear4_3(gelu(linear4_3(x,wb1)),wb2))',
        'tanh(linear4_4(gelu(linear4_4(x,wb1)),wb2)), elu(linear4_4(gelu(linear4_4(x,wb1)),wb2)), sigmoid(linear4_4(relu(linear4_4(x,wb1)),wb2)), linear4_4(gelu(linear4_4(max(x,tanh(x)),wb1)),wb2), linear4_4(relu(mul(linear4_4(x,wb1),x)),wb2), leaky_relu(linear4_4(gelu(linear4_4(x,wb1)),wb2))',
    ]

if args.stage3 and not args.teacher_downstream_path2:
    args.teacher_downstream_path2 = [
        './exp/train/bert_base/mnli/best_model.bin',
        './exp/train/bert_base/qqp/best_model.bin',
        './exp/train/bert_base/qnli/best_model.bin',
        './exp/train/bert_base/sst-2/best_model.bin',
        './exp/train/bert_base/cola/best_model.bin',
        './exp/train/bert_base/sts-b/best_model.bin',
        './exp/train/bert_base/mrpc/best_model.bin',
        './exp/train/bert_base/rte/best_model.bin',
    ]


# Train function for stage 1 and 2
def train(entire_ffn_func, entire_linear_idx):
    ffn_arch = [entire_ffn_func, entire_linear_idx]

    # Load model weights for pretraining
    ckpt_path = args.teacher_pretrain_path
    if ckpt_path:
        if os.path.exists(ckpt_path):
            load_pretrain_state_dict(teacher_model_name, teacher_model, ckpt_path, use_gpu)
            if args.local_rank == 0:
                logging.info('Loaded teacher pretrained state dict from \'{}\''.format(ckpt_path))
        else:
            if args.local_rank == 0:
                logging.info('Teacher pretrained state dict is not found in \'{}\''.format(ckpt_path))

    ckpt_path = args.student_pretrain_path1
    if ckpt_path:
        if os.path.exists(ckpt_path):
            load_pretrain_state_dict(student_model_name, student_model, ckpt_path, use_gpu, is_finetune=True)
            if args.local_rank == 0:
                logging.info('Loaded student pretrained state dict from \'{}\''.format(ckpt_path))
        else:
            if args.local_rank == 0:
                logging.info('Student pretrained state dict is not found in \'{}\''.format(ckpt_path))

    # Create pretrain optimizer and scheduler
    optimizer = create_optimizer(student_model, args.optim_type, args.lr1, args.weight_decay, args.momentum)
    scheduler = create_scheduler(optimizer, args.sched_type, pretrain_num_sched_steps, pretrain_num_warmup_steps)

    # Start pretraining
    if args.local_rank == 0:
        logging.info('==> Start pretraining')
    for epoch in range(args.start_epoch1, args.total_epochs1 + 1):
        pretrain_train_loader = pretrain_train_loaders[epoch - 1]
        if args.distributed:
            pretrain_train_loader.sampler.set_epoch(epoch)

        pretrain_val_loader, best_result = None, None
        is_valid = train_epoch(
            teacher_model, student_model, epoch, optimizer, scheduler, pretrain_train_loader, pretrain_val_loader, best_result,
            is_pretrain=True, ffn_arch=ffn_arch)
        if not is_valid:
            return 0

    # ------------------------------------------------------------------------------------------------------ #

    # Load model weights for downstream training
    ckpt_path = args.teacher_downstream_path1
    if ckpt_path:
        if os.path.exists(ckpt_path):
            load_pretrain_state_dict(teacher_model_name, teacher_model, ckpt_path, use_gpu, is_finetune=True)
            if args.local_rank == 0:
                logging.info('Loaded teacher finetuned state dict from \'{}\''.format(ckpt_path))
        else:
            if args.local_rank == 0:
                logging.info('Teacher finetuned state dict is not found in \'{}\''.format(ckpt_path))

    # Create downstream optimizer and scheduler
    optimizer = create_optimizer(student_model, args.optim_type, args.lr2, args.weight_decay, args.momentum)
    scheduler = create_scheduler(optimizer, args.sched_type, downstream_num_sched_steps, downstream_num_warmup_steps)

    # Start downstream training
    if args.local_rank == 0:
        logging.info('==> Start downstream training')
    best_results = [0, None, True]
    for epoch in range(args.start_epoch2, args.total_epochs2 + 1):
        if args.distributed:
            downstream_train_loader.sampler.set_epoch(epoch)
        best_results = train_epoch(
            teacher_model, student_model, epoch, optimizer, scheduler, downstream_train_loader, downstream_val_loader,
            best_results, ffn_arch=ffn_arch)
        best_fitness, best_idx, is_valid = best_results

        if not is_valid:
            return 0
        if best_idx is not None and args.local_rank == 0:
            logging.info('Best fitness {} found in epoch {} step {}'.format(best_fitness, best_idx[0], best_idx[1]))

    fitness = best_results[0]
    return fitness


# Train function for stage 3
def train2():
    ckpt_path = args.student_downstream_path
    if ckpt_path:
        if os.path.exists(ckpt_path):
            load_pretrain_state_dict(student_model_name, student_model, ckpt_path, use_gpu, is_finetune=True)
            if args.local_rank == 0:
                logging.info('Loaded student finetuned state dict from \'{}\''.format(ckpt_path))
        else:
            if args.local_rank == 0:
                logging.info('Student finetuned state dict is not found in \'{}\''.format(ckpt_path))
    else:
        ckpt_path = args.student_pretrain_path3
        if ckpt_path:
            if os.path.exists(ckpt_path):
                load_pretrain_state_dict(student_model_name, student_model, ckpt_path, use_gpu, is_finetune=True)
                if args.local_rank == 0:
                    logging.info('Loaded student pretrained state dict from \'{}\''.format(ckpt_path))
            else:
                if args.local_rank == 0:
                    logging.info('Student pretrained state dict is not found in \'{}\''.format(ckpt_path))
        else:
            # Load model weights for pretraining
            ckpt_path = args.teacher_pretrain_path
            if ckpt_path:
                if os.path.exists(ckpt_path):
                    load_pretrain_state_dict(teacher_model_name, teacher_model, ckpt_path, use_gpu)
                    if args.local_rank == 0:
                        logging.info('Loaded teacher pretrained state dict from \'{}\''.format(ckpt_path))
                else:
                    if args.local_rank == 0:
                        logging.info('Teacher pretrained state dict is not found in \'{}\''.format(ckpt_path))

            ckpt_path = args.student_pretrain_path2
            if ckpt_path:
                if os.path.exists(ckpt_path):
                    load_pretrain_state_dict(student_model_name, student_model, ckpt_path, use_gpu, is_finetune=True)
                    if args.local_rank == 0:
                        logging.info('Loaded student pretrained state dict from \'{}\''.format(ckpt_path))
                else:
                    if args.local_rank == 0:
                        logging.info('Student pretrained state dict is not found in \'{}\''.format(ckpt_path))

            # Create pretrain optimizer and scheduler
            optimizer = create_optimizer(student_model, args.optim_type, args.lr3, args.weight_decay, args.momentum)
            scheduler = create_scheduler(optimizer, args.sched_type, pretrain_num_sched_steps, pretrain_num_warmup_steps)

            # Start pretraining
            if args.local_rank == 0:
                logging.info('==> Start pretraining')
            for epoch in range(args.start_epoch3, args.total_epochs3 + 1):
                pretrain_train_loader = pretrain_train_loaders[epoch - 1]
                if args.distributed:
                    pretrain_train_loader.sampler.set_epoch(epoch)

                pretrain_val_loader, best_result = None, None
                train_epoch(
                    teacher_model, student_model, epoch, optimizer, scheduler, pretrain_train_loader, pretrain_val_loader,
                    best_result, is_pretrain=True)

                if epoch % args.save_freq == 0 or epoch == args.total_epochs3:
                    if args.local_rank == 0:
                        state = {'state_dict': student_model.state_dict(),
                                 'epoch': epoch}
                        ckpt_name = 'pretrain_ckpt_ep' + str(epoch) + '.bin'
                        save_checkpoint(state, args.exp_dir, ckpt_name, args.ckpt_keep_num)
                        logging.info('Supernet pretrained state dict has been saved to \'{}\''.format(os.path.join(args.exp_dir, ckpt_name)))
                if args.local_rank == 0:
                    logging.info('-' * 50)

            exit(0)
        # ------------------------------------------------------------------------------------------------------ #

        # Create downstream optimizer and scheduler
        optimizer = create_optimizer(student_model, args.optim_type, args.lr4, args.weight_decay, args.momentum)
        scheduler = create_scheduler(optimizer, args.sched_type, downstream_num_sched_steps, downstream_num_warmup_steps)

        # Start downstream training
        if args.local_rank == 0:
            logging.info('==> Start downstream training')
        best_results = [0, None, True]
        for epoch in range(args.start_epoch4, args.total_epochs4 + 1):
            best_results = train_epoch(
                teacher_model, student_model, epoch, optimizer, scheduler, downstream_train_loader, downstream_val_loader,
                best_results)

            if epoch % args.save_freq == 0 or epoch == args.total_epochs4:
                if args.local_rank == 0:
                    state = {'state_dict': student_model.state_dict(),
                             'epoch': epoch}
                    ckpt_name = 'downstream_ckpt_ep' + str(epoch) + '.bin'
                    save_checkpoint(state, args.exp_dir, ckpt_name, args.ckpt_keep_num)
                    logging.info('Supernet finetuned state dict has been saved to \'{}\''.format(os.path.join(args.exp_dir, ckpt_name)))
            if args.local_rank == 0:
                logging.info('-' * 50)

        exit(0)


def train_epoch(teacher_model, student_model, epoch, optimizer, scheduler, train_loader, val_loader, best_results,
                is_pretrain=False, ffn_arch=None):
    teacher_model.eval()
    student_model.train()

    if not is_pretrain:
        best_fitness, best_idx, _ = best_results
    losses, train_time, data_time = [AverageMeter() for _ in range(3)]
    attn_losses, ffn_losses, pred_losses = [AverageMeter() for _ in range(3)]
    st_time = time.time()

    def _update_losses(all_losses, loss, data_size):
        if args.distributed:
            loss = reduce_tensor(loss.detach(), args.world_size)
        all_losses.update(loss.item(), data_size)

    for batch_idx, data in enumerate(train_loader):
        data_time.update(time.time() - st_time)
        if use_gpu:
            data = [data_.cuda() for data_ in data]

        if args.stage3:
            # Sample search model for each step that has proper parameters
            max_times = 1e4
            while max_times > 0:
                all_cand_expr = np.array([expr.split(', ') for expr in init_expr])
                entire_ind = [gp.PrimitiveTree.from_string(random.choice(all_cand_expr[:, i]), pset)
                              for i in range(num_student_layers)]
                entire_ffn_func = [gp.compile(ind, pset) for ind in entire_ind]
                entire_linear_idx = get_entire_linear_idx(entire_ind)
                ffn_arch = [entire_ffn_func, entire_linear_idx]

                entire_params = get_entire_params(param_list, entire_ind)
                if args.min_params <= entire_params <= args.max_params:
                    break
                max_times -= 1

            if is_pretrain:
                task, task_id = 'mnli', 0
                with torch.no_grad():
                    teacher_outputs = teacher_model(task_id, *data[:-1])
                student_outputs = student_model(task_id, *data[:-1], *ffn_arch)
            else:
                task_ids, token_ids, segment_ids, position_ids, attn_mask, labels = data
                task_id = task_ids[0].item()
                task = glue_train_tasks[task_id]

                load_multi_task_state_dict(teacher_model, args.teacher_downstream_path2[task_id], task_id)  # Load corresponding teacher ckpt of current task
                with torch.no_grad():
                    teacher_outputs = teacher_model(task_id, token_ids, segment_ids, position_ids, attn_mask)
                student_outputs = student_model(task_id, token_ids, segment_ids, position_ids, attn_mask, *ffn_arch)
        else:
            task = args.task
            with torch.no_grad():
                teacher_outputs = teacher_model(*data[:-1])
            student_outputs = student_model(*data[:-1], *ffn_arch)

        # Check whether ffn architecture is valid
        if student_outputs[0] is None:
            if args.local_rank == 0:
                logging.info('Invalid individual, runtime error is caused')
            if is_pretrain:
                return False
            return best_fitness, best_idx, False

        loss, attn_loss, ffn_loss, pred_loss = calc_distil_losses(
            teacher_outputs, student_outputs, data[-1], task, is_pretrain)

        # Check whether loss is nan
        check_loss = loss.detach()
        if args.distributed:
            check_loss = reduce_tensor(check_loss, args.world_size)
        if torch.isnan(check_loss):
            if args.local_rank == 0:
                logging.info('Invalid individual, loss is nan')
            if is_pretrain:
                return False
            return best_fitness, best_idx, False

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        _update_losses(losses, loss, data[0].size(0))
        _update_losses(attn_losses, attn_loss, data[0].size(0))
        _update_losses(ffn_losses, ffn_loss, data[0].size(0))
        if not is_pretrain:
            _update_losses(pred_losses, pred_loss, data[0].size(0))

        if use_gpu:
            torch.cuda.synchronize()
        train_time.update(time.time() - st_time)

        if args.local_rank == 0 and (batch_idx == 0 or (batch_idx + 1) % args.loss_disp_freq == 0 or batch_idx + 1 == len(train_loader)):
            lr = scheduler.get_lr()[0]
            if is_pretrain:
                if args.stage3:
                    total_epochs = args.total_epochs3
                    logging.info('Epoch: [{}/{}][{}/{}]\t'
                                 'LR: {:.2e}\t'
                                 'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                                 'Attn and ffn loss: {attn_loss.val:.4f} {ffn_loss.val:.4f} '
                                 '({attn_loss.avg:.4f} {ffn_loss.avg:.4f})\t'
                                 'Params: {params:.2f}M\t'
                                 'Arch: {arch}\t'
                                 'Train time: {train_time.val:.4f}s ({train_time.avg:.4f}s)\t'
                                 'Load data time: {data_time.val:.4f}s ({data_time.avg:.4f}s)'
                                 .format(epoch, total_epochs, batch_idx + 1, len(train_loader), lr,
                                         loss=losses, attn_loss=attn_losses, ffn_loss=ffn_losses,
                                         params=entire_params, arch=entire_linear_idx,
                                         train_time=train_time, data_time=data_time))
                else:
                    total_epochs = args.total_epochs1
                    logging.info('Epoch: [{}/{}][{}/{}]\t'
                                 'LR: {:.2e}\t'
                                 'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                                 'Attn and ffn loss: {attn_loss.val:.4f} {ffn_loss.val:.4f} '
                                 '({attn_loss.avg:.4f} {ffn_loss.avg:.4f})\t'
                                 'Train time: {train_time.val:.4f}s ({train_time.avg:.4f}s)\t'
                                 'Load data time: {data_time.val:.4f}s ({data_time.avg:.4f}s)'
                                 .format(epoch, total_epochs, batch_idx + 1, len(train_loader), lr,
                                         loss=losses, attn_loss=attn_losses, ffn_loss=ffn_losses,
                                         train_time=train_time, data_time=data_time))
            else:
                if args.stage3:
                    total_epochs = args.total_epochs4
                    logging.info('Epoch: [{}/{}][{}/{}]\t'
                                 'LR: {:.2e}\t'
                                 'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                                 'Attn, ffn and pred loss: {attn_loss.val:.4f} {ffn_loss.val:.4f} {pred_loss.val:.4f} '
                                 '({attn_loss.avg:.4f} {ffn_loss.avg:.4f} {pred_loss.avg:.4f})\t'
                                 'Params: {params:.2f}M\t'
                                 'Arch: {arch}\t'
                                 'Task: {task}\t'
                                 'Train time: {train_time.val:.4f}s ({train_time.avg:.4f}s)\t'
                                 'Load data time: {data_time.val:.4f}s ({data_time.avg:.4f}s)'
                                 .format(epoch, total_epochs, batch_idx + 1, len(train_loader), lr,
                                         loss=losses, attn_loss=attn_losses, ffn_loss=ffn_losses, pred_loss=pred_losses,
                                         params=entire_params, arch=entire_linear_idx, task=task, train_time=train_time,
                                         data_time=data_time))
                else:
                    total_epochs = args.total_epochs2
                    logging.info('Epoch: [{}/{}][{}/{}]\t'
                                 'LR: {:.2e}\t'
                                 'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                                 'Attn, ffn and pred loss: {attn_loss.val:.4f} {ffn_loss.val:.4f} {pred_loss.val:.4f} '
                                 '({attn_loss.avg:.4f} {ffn_loss.avg:.4f} {pred_loss.avg:.4f})\t'
                                 'Train time: {train_time.val:.4f}s ({train_time.avg:.4f}s)\t'
                                 'Load data time: {data_time.val:.4f}s ({data_time.avg:.4f}s)'
                                 .format(epoch, total_epochs, batch_idx + 1, len(train_loader), lr,
                                         loss=losses, attn_loss=attn_losses, ffn_loss=ffn_losses, pred_loss=pred_losses,
                                         train_time=train_time, data_time=data_time))

        if args.stage3:
            val_freq = args.val_freq2
        else:
            val_freq = args.val_freq1

        if not is_pretrain and ((batch_idx + 1) % val_freq == 0 or batch_idx + 1 == len(train_loader)):
            if args.stage3:
                # Evaluate the arch searched in stage 2 on dev dataset
                all_cand_expr = np.array([expr.split(', ') for expr in init_expr])
                entire_ind = [gp.PrimitiveTree.from_string(all_cand_expr[0, i], pset)
                              for i in range(num_student_layers)]
                entire_ffn_func = [gp.compile(ind, pset) for ind in entire_ind]
                entire_linear_idx = get_entire_linear_idx(entire_ind)
                ffn_arch = [entire_ffn_func, entire_linear_idx]

                fitness, all_task_fitness = validate2(ffn_arch, use_dev=True)
                is_best = fitness > best_fitness
                if is_best:
                    best_fitness = fitness
                    best_idx = [epoch, batch_idx + 1]
                    if args.local_rank == 0:
                        logging.info('Total fitness: {}\tAll tasks: {}'.format(fitness, all_task_fitness))
            else:
                fitness = validate(student_model, val_loader, ffn_arch)
                is_best = fitness > best_fitness
                if is_best:
                    best_fitness = fitness
                    best_idx = [epoch, batch_idx + 1]
                    if args.local_rank == 0:
                        logging.info('Best fitness found: {}'.format(best_fitness))

            student_model.train()
        st_time = time.time()

    if is_pretrain:
        return True
    return best_fitness, best_idx, True


# Validate function for stage 1 and 2
def validate(model, data_loader, ffn_arch):
    model.eval()
    all_fitness = AverageMeter()
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            if use_gpu:
                data = [data_.cuda() for data_ in data]
            outputs = model(*data[:-1], *ffn_arch)

            preds, labels = outputs[0].detach().cpu().numpy(), data[-1].detach().cpu().numpy()
            preds = np.squeeze(preds) if args.task == 'sts-b' else np.argmax(preds, axis=1)
            metrics = compute_glue_metrics(args.task, preds, labels)
            fitness = metrics[all_glue_select_metrics[args.task]]

            if args.distributed:
                fitness = reduce_tensor(torch.tensor(fitness, dtype=torch.float64).cuda(), args.world_size)
            all_fitness.update(fitness.item(), data[0].size(0))

        return all_fitness.avg


# Validate function for stage 3
def validate2(ffn_arch, use_dev=False, is_search=False):
    student_model.eval()
    all_fitness = AverageMeter()
    all_task_fitness = {}
    if use_dev:
        data_loader = downstream_dev_loader
    else:
        data_loader = downstream_val_loader

    with torch.no_grad():
        for task, data_loader in zip(glue_train_tasks, data_loader):
            if is_search and task != 'mnli':  # Only use MNLI to search
                continue
            if use_dev and task == 'qqp':  # Do not evaluate QQP when training the supernet with multiple tasks
                continue
            all_task_fitness[task] = AverageMeter()

            for batch_idx, data in enumerate(data_loader):
                if use_gpu:
                    data = [data_.cuda() for data_ in data]
                task_ids, token_ids, segment_ids, position_ids, attn_mask, labels = data
                outputs = student_model(task_ids[0].item(), token_ids, segment_ids, position_ids, attn_mask, *ffn_arch)

                if task != 'sts-b':
                    labels = labels.long()
                preds, labels = outputs[0].detach().cpu().numpy(), labels.detach().cpu().numpy()
                preds = np.squeeze(preds) if task == 'sts-b' else np.argmax(preds, axis=1)
                metrics = compute_glue_metrics(task, preds, labels)
                fitness = metrics[all_glue_select_metrics[task]]

                if args.distributed:
                    fitness = reduce_tensor(torch.tensor(fitness, dtype=torch.float64).cuda(), args.world_size)
                all_task_fitness[task].update(fitness.item(), data[0].size(0))
                all_fitness.update(fitness.item(), data[0].size(0))
            all_task_fitness[task] = all_task_fitness[task].avg

        if use_dev:
            return all_fitness.avg, all_task_fitness
        return all_fitness.avg


def calc_distil_losses(teacher_outputs, student_outputs, labels, task, is_pretrain):
    teacher_pred_logits, teacher_attn_outputs, teacher_ffn_outputs = teacher_outputs
    student_pred_logits, student_attn_outputs, student_ffn_outputs = student_outputs[:3]

    def _replace_attn_mask(attn_output):
        replace_values = torch.zeros_like(attn_output)
        if use_gpu:
            replace_values = replace_values.cuda()
        attn_output = torch.where(attn_output <= -1e2, replace_values, attn_output)
        return attn_output

    mse_loss = nn.MSELoss()
    attn_loss, ffn_loss = 0, 0
    ffn_loss += mse_loss(teacher_ffn_outputs[0], student_ffn_outputs[0])
    for layer_id in range(num_student_layers):
        teacher_layer_id = layer_id * teacher_interval + 1
        attn_loss += mse_loss(_replace_attn_mask(teacher_attn_outputs[teacher_layer_id]),
                              _replace_attn_mask(student_attn_outputs[layer_id]))
        ffn_loss += mse_loss(teacher_ffn_outputs[teacher_layer_id + 1], student_ffn_outputs[layer_id + 1])

    hidden_loss = attn_loss + ffn_loss
    if is_pretrain:
        return hidden_loss, attn_loss, ffn_loss, None

    pred_loss = mse_loss(student_pred_logits, labels) if task == 'sts-b' else \
        soft_cross_entropy(student_pred_logits, teacher_pred_logits, args.sce_temp)
    total_loss = args.hidden_ratio * hidden_loss + args.pred_ratio * pred_loss
    return total_loss, attn_loss, ffn_loss, pred_loss


def create_pretrain_dataset_loaders():
    if args.stage3:
        start_epoch, total_epochs = args.start_epoch3, args.total_epochs3
        batch_size, train_ratio, val_ratio = args.batch_size3, args.train_ratio3, args.val_ratio3
        pretrain_data_dir, concate_data_dir = all_train_dir, args.pretrain_dir2

        all_num_data_epochs = []
        total_pretrain_examples = 0
        for train_dir in all_train_dir:
            num_epoch_examples = []
            num_data_epochs = len([file for file in os.listdir(train_dir) if re.match(r'epoch_\d+_metrics.json', file) is not None])
            all_num_data_epochs.append(num_data_epochs)
            for i in range(num_data_epochs):
                metrics_file = train_dir / 'epoch_{}_metrics.json'.format(i)
                if metrics_file.is_file():
                    metrics = json.loads(metrics_file.read_text())
                    num_epoch_examples.append(metrics['num_training_examples'])
            for epoch in range(total_epochs):
                total_pretrain_examples += int(num_epoch_examples[epoch % len(num_epoch_examples)] * train_ratio)

        for data_epoch in all_num_data_epochs:
            assert data_epoch == all_num_data_epochs[0]
        num_data_epochs = all_num_data_epochs[0]
    else:
        start_epoch, total_epochs = args.start_epoch1, args.total_epochs1
        batch_size, train_ratio, val_ratio = args.batch_size1, args.train_ratio1, args.val_ratio1
        pretrain_data_dir, concate_data_dir = args.pretrain_dir1, None

        num_epoch_examples = []
        total_pretrain_examples = 0
        num_data_epochs = len([file for file in os.listdir(pretrain_data_dir) if re.match(r'epoch_\d+_metrics.json', file) is not None])
        for i in range(num_data_epochs):
            metrics_file = pretrain_data_dir / 'epoch_{}_metrics.json'.format(i)
            if metrics_file.is_file():
                metrics = json.loads(metrics_file.read_text())
                num_epoch_examples.append(metrics['num_training_examples'])
        for epoch in range(total_epochs):
            total_pretrain_examples += int(num_epoch_examples[epoch % len(num_epoch_examples)] * train_ratio)

    pretrain_train_loaders = []
    for epoch in range(start_epoch, total_epochs + 1):
        _, cur_pretrain_train_loader = create_pretrain_dataset(
            pretrain_data_dir, epoch - 1, tokenizer, num_data_epochs, args.local_rank, batch_size, use_gpu,
            args.distributed, 'train', train_ratio, val_ratio, concate_data_dir)
        pretrain_train_loaders.append(cur_pretrain_train_loader)

    return pretrain_train_loaders, total_pretrain_examples


def create_downstream_dataset_loaders():
    if args.stage3:
        _, _, _, downstream_train_loader = create_multi_task_dataset(
            args.student_model2, glue_train_tasks, args.data_dir, tokenizer, args.max_seq_len, args.max_query_len,
            args.trunc_stride, args.batch_size4, args.train_ratio4, args.val_ratio4, use_gpu, args.distributed,
            'train', args.local_rank, args.cache_dir)
        _, _, _, downstream_val_loader = create_multi_task_dataset(
            args.student_model2, glue_train_tasks, args.data_dir, tokenizer, args.max_seq_len, args.max_query_len,
            args.trunc_stride, args.batch_size4, args.train_ratio4, args.val_ratio4, use_gpu, args.distributed,
            'val', args.local_rank, args.cache_dir)
        _, _, _, downstream_dev_loader = create_multi_task_dataset(
            args.student_model2, glue_train_tasks, args.data_dir, tokenizer, args.max_seq_len, args.max_query_len,
            args.trunc_stride, args.batch_size4, args.train_ratio4, args.val_ratio4, use_gpu, args.distributed,
            'dev', args.local_rank, args.cache_dir)
        return downstream_train_loader, downstream_val_loader, downstream_dev_loader
    else:
        _, _, _, downstream_train_loader = create_split_dataset(
            args.student_model1, args.task, args.data_dir, tokenizer, args.max_seq_len, args.max_query_len,
            args.trunc_stride, args.batch_size2, args.train_ratio2, args.val_ratio2, use_gpu, args.distributed,
            'train', args.local_rank, args.cache_dir)
        _, _, _, downstream_val_loader = create_split_dataset(
            args.student_model1, args.task, args.data_dir, tokenizer, args.max_seq_len, args.max_query_len,
            args.trunc_stride, args.batch_size2, args.train_ratio2, args.val_ratio2, use_gpu, args.distributed,
            'val', args.local_rank, args.cache_dir)
        return downstream_train_loader, downstream_val_loader


if __name__ == '__main__':
    use_gpu = False
    if args.gpu_devices and torch.cuda.is_available():
        use_gpu = True
    args.exp_dir = os.path.join(args.exp_dir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    setup_logger(args.exp_dir)

    if args.local_rank == 0:
        logging.info(args)
        if use_gpu:
            logging.info('Currently using GPU: {}'.format(args.gpu_devices))
        else:
            logging.info('Currently using CPU')
    set_seeds(args.seed, use_gpu)

    if use_gpu and args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        logging.info('Training in distributed mode (process {}/{})'.format(args.local_rank + 1, args.world_size))

    # Load model and tokenizer
    if args.stage3:
        teacher_model_name, student_model_name = args.teacher_model2, args.student_model2
    else:
        teacher_model_name, student_model_name = args.teacher_model1, args.student_model1

    teacher_config = select_config(teacher_model_name, args.lowercase)
    student_config = select_config(student_model_name, args.lowercase)
    teacher_model = select_model(teacher_model_name, args.lowercase, args.task, return_hid=True)
    student_model = select_model(student_model_name, args.lowercase, args.task, return_hid=True)
    tokenizer = select_tokenizer(
        teacher_model_name, args.lowercase, args.task, args.vocab_path, args.max_seq_len, args.max_query_len, args.merge_path)
    teacher_interval = teacher_config.num_layers // student_config.num_layers
    num_student_layers = student_config.num_layers
    param_list = student_config.param_list
    if args.local_rank == 0:
        logging.info('Teacher model size: {:.2f}M'.format(calc_params(teacher_model) / 1e6))
        logging.info('Student model size: {:.2f}M'.format(calc_params(student_model) / 1e6))

    if use_gpu:
        teacher_model, student_model = teacher_model.cuda(), student_model.cuda()
        if args.distributed:
            teacher_model = DDP(teacher_model, delay_allreduce=True)
            student_model = DDP(student_model, delay_allreduce=True)
        else:
            teacher_model = nn.DataParallel(teacher_model)
            student_model = nn.DataParallel(student_model)

    # Create pretrain and downstream datasets
    if args.stage3:
        all_train_dir = [args.wiki_dir, args.book_dir]
        if not args.student_pretrain_path3:
            pretrain_train_loaders, total_pretrain_examples = create_pretrain_dataset_loaders()
            pretrain_num_sched_steps = total_pretrain_examples // (args.batch_size3 * args.world_size)
            pretrain_num_warmup_steps = int(pretrain_num_sched_steps * args.warmup_proportion)
        downstream_train_loader, downstream_val_loader, downstream_dev_loader = create_downstream_dataset_loaders()
        downstream_num_sched_steps = len(downstream_train_loader) * args.total_epochs4
        downstream_num_warmup_steps = int(downstream_num_sched_steps * args.warmup_proportion)
    else:
        pretrain_train_loaders, total_pretrain_examples = create_pretrain_dataset_loaders()
        pretrain_num_sched_steps = total_pretrain_examples // (args.batch_size1 * args.world_size)
        pretrain_num_warmup_steps = int(pretrain_num_sched_steps * args.warmup_proportion)
        downstream_train_loader, downstream_val_loader = create_downstream_dataset_loaders()
        downstream_num_sched_steps = len(downstream_train_loader) * args.total_epochs2
        downstream_num_warmup_steps = int(downstream_num_sched_steps * args.warmup_proportion)

    pset = register_custom_ops()
    if args.stage2:
        pset = register_custom_ops2()

    train_val_function = train
    if args.stage3:
        train_val_function = [train2, validate2]

    # Create fixed parameters to check if two expressions are the same based on the output
    if not os.path.exists(args.fixed_mat_dir):
        time.sleep(5)
        if args.local_rank == 0:
            os.makedirs(args.fixed_mat_dir, exist_ok=True)
        hidden_size = student_config.hidden_size
        x = torch.randn(10, args.max_seq_len, hidden_size)
        if args.local_rank == 0:
            torch.save(x, os.path.join(args.fixed_mat_dir, 'fixed_x.bin'))
        for layer_id in range(1, num_student_layers + 1):
            for l_idx in student_config.expansion_ratio_map.keys():
                w = torch.randn(2, hidden_size, hidden_size)
                b = torch.randn(2, hidden_size)
                wb = [[w_, b_] for w_, b_ in zip(w, b)]
                if args.local_rank == 0:
                    torch.save(wb, os.path.join(args.fixed_mat_dir, 'fixed_wb' + str(layer_id) + '_' + l_idx + '.bin'))
        if args.local_rank == 0:
            logging.info('Fixed params have been saved to \'{}\''.format(args.fixed_mat_dir))
        time.sleep(5)

    st_time = time.time()
    search_phase = SearchPhase(
        init_expr, pset, param_list, num_student_layers, args.min_height, args.max_height,
        args.min_params, args.max_params, args.fixed_mat_dir, args.n_init_samples, args.train_interval,
        args.n_total_samples, is_stage2=args.stage2, is_stage3=args.stage3)
    search_phase.run(train_val_function, args.exp_dir, args.ind_disp_freq, local_rank=args.local_rank)

    if args.local_rank == 0:
        elapsed = round(time.time() - st_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        logging.info('Finished, total training time (h:m:s): {}'.format(elapsed))
