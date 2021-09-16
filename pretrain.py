import os
import re
import torch
import torch.nn as nn
import numpy as np
import logging
import argparse
import time
import json
import datetime
from pathlib import Path
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from deap import gp
from models import select_config, select_single_model, nas_bert_models
from tokenizers import select_basic_tokenizer
from utils import AverageMeter, register_custom_ops, register_custom_ops3, set_seeds, setup_logger, get_entire_linear_idx, \
    get_entire_params, calc_params, reduce_tensor, save_checkpoint, load_pretrain_state_dict, load_resume_state_dict, \
    load_multi_task_state_dict, load_supernet_state_dict_6_540_to_12_360, load_supernet_state_dict_6_540_to_6_360, \
    create_optimizer, create_scheduler, create_pretrain_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--distributed', action='store_true', help='distributed mode')
parser.add_argument('--fp16', action='store_true', help='mixed precision training mode')
parser.add_argument('--opt_level', default='O1', type=str, help='fp16 optimization level')
parser.add_argument('--gpu_devices', default='4,5,6,7', type=str, help='available gpu devices')
parser.add_argument('--seed', default=42, type=int, help='seed')

parser.add_argument('--lowercase', action='store_true', help='whether to do lowercase')
parser.add_argument('--temperature', default=1, type=float, help='temperature for soft cross entropy loss')
parser.add_argument('--ffn_expr', default=[], nargs='+', help='feed-forward network expression')
parser.add_argument('--teacher_model', default='bert_base', type=str, help='teacher model name')
parser.add_argument('--student_model', default='tiny_bert', type=str, help='student model name')
parser.add_argument('--vocab_path', default='', type=str, help='path to pretrained vocabulary file')
parser.add_argument('--merge_path', default='', type=str, help='path to pretrained merge file (for roberta)')
parser.add_argument('--train_ratio', default=1, type=float, help='ratio of train dataset')
parser.add_argument('--val_ratio', default=0, type=float, help='ratio of val dataset')

parser.add_argument('--start_epoch', default=1, type=int, help='start epoch (default is 1)')
parser.add_argument('--total_epochs', default=3, type=int, help='total epochs')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--optim_type', default='adamw', type=str, help='optimizer type')
parser.add_argument('--sched_type', default='step', type=str, help='lr scheduler type')
parser.add_argument('--warmup_proportion', default=0.1, type=float, help='proportion of warmup steps')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay')
parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max gradient norm')
parser.add_argument('--disp_freq', default=50, type=int, help='display frequency')
parser.add_argument('--save_freq', default=1, type=int, help='checkpoint save frequency')
parser.add_argument('--ckpt_keep_num', default=1, type=int, help='max number of checkpoint files to keep')

parser.add_argument('--teacher_pretrain_path', default='', type=str,help='path to pretrained state dict of teacher model')
parser.add_argument('--student_pretrain_path', default='', type=str, help='path to pretrained student state dict')
parser.add_argument('--student_multi_task_pretrain_path', default='', type=str, help='path to multi-task pretrained student state dict')
parser.add_argument('--student_pretrain_path_6_540_to_6_360', default='', type=str, help='path to pretrained student state dict from L6-H540 to L6-H360')
parser.add_argument('--student_pretrain_path_6_540_to_12_360', default='', type=str, help='path to pretrained student state dict from L6-H540 to L12-H360')
parser.add_argument('--student_resume_path', default='', type=str, help='path to resume checkpoint of student model')
parser.add_argument('--wiki_dir', default='', type=Path, help='directory to wikipedia dataset')
parser.add_argument('--book_dir', default='', type=Path, help='directory to bookcorpus dataset')
parser.add_argument('--concate_data_dir', default='', type=Path, help='directory to concatenated dataset')
parser.add_argument('--exp_dir', default='./exp/tmp/', type=str, help='experiment directory')
parser.add_argument('--local_rank', default=0, type=int, help='DDP local rank')
parser.add_argument('--world_size', default=1, type=int, help='DDP world size')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices

if args.book_dir and args.concate_data_dir:
    args.all_train_dir = [args.wiki_dir, args.book_dir]
else:
    args.all_train_dir = [args.wiki_dir]


def main():
    args.exp_dir = os.path.join(args.exp_dir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    setup_logger(args.exp_dir)
    if args.local_rank == 0:
        logging.info(args)

    use_gpu = False
    if args.gpu_devices and torch.cuda.is_available():
        use_gpu = True
    if use_gpu and args.local_rank == 0:
        logging.info('Currently using GPU: {}'.format(args.gpu_devices))
    elif not use_gpu and args.local_rank == 0:
        logging.info('Currently using CPU')
    set_seeds(args.seed, use_gpu)

    if use_gpu and args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        logging.info('Training in distributed mode (process {}/{})'.format(args.local_rank + 1, args.world_size))

    # Register custom operators for auto bert
    if args.student_model in nas_bert_models:
        if args.student_model == 'auto_tiny_bert':
            pset = register_custom_ops3()
        else:
            pset = register_custom_ops()
        if args.student_model == 'auto_bert_12':
            args.ffn_expr = np.reshape([[x, x] for x in args.ffn_expr], -1).tolist()
        entire_ffn_func = [gp.compile(expr, pset) for expr in args.ffn_expr]
        entire_linear_idx = get_entire_linear_idx(args.ffn_expr)
        args.ffn_arch = [entire_ffn_func, entire_linear_idx]

    # Load model and tokenizer
    teacher_config = select_config(args.teacher_model, args.lowercase)
    student_config = select_config(args.student_model, args.lowercase)
    teacher_model = select_single_model(args.teacher_model, args.lowercase)
    student_model = select_single_model(args.student_model, args.lowercase)
    args.tokenizer = select_basic_tokenizer(args.teacher_model, args.lowercase, args.vocab_path, args.merge_path)
    args.teacher_interval = teacher_config.num_layers // student_config.num_layers
    args.num_student_layers = student_config.num_layers
    args.student_config = student_config

    if use_gpu:
        teacher_model, student_model = teacher_model.cuda(), student_model.cuda()
    if args.local_rank == 0:
        logging.info('Teacher model size: {:.2f}M'.format(calc_params(teacher_model) / 1e6))
        logging.info('Student model size: {:.2f}M'.format(calc_params(student_model) / 1e6))
        if args.student_model in nas_bert_models:
            logging.info('Student sub model size: {:.2f}M'.format(get_entire_params(student_config.param_list, args.ffn_expr)))
        logging.info('Student model config: {}'.format(args.student_config.__dict__))

    # Count total training examples
    all_num_data_epochs = []
    total_examples = 0
    for train_dir in args.all_train_dir:
        num_epoch_examples = []
        num_data_epochs = len([file for file in os.listdir(train_dir) if re.match(r'epoch_\d+_metrics.json', file) is not None])
        all_num_data_epochs.append(num_data_epochs)
        for i in range(num_data_epochs):
            metrics_file = train_dir / 'epoch_{}_metrics.json'.format(i)
            if metrics_file.is_file():
                metrics = json.loads(metrics_file.read_text())
                num_epoch_examples.append(metrics['num_training_examples'])
        for epoch in range(args.total_epochs):
            total_examples += int(num_epoch_examples[epoch % len(num_epoch_examples)] * args.train_ratio)

    for data_epoch in all_num_data_epochs:
        assert data_epoch == all_num_data_epochs[0]
    args.num_data_epochs = all_num_data_epochs[0]

    # Create optimizer and scheduler
    num_sched_steps = total_examples // (args.batch_size * args.world_size)
    num_warmup_steps = int(num_sched_steps * args.warmup_proportion)
    optimizer = create_optimizer(student_model, args.optim_type, args.lr, args.weight_decay, args.momentum)
    scheduler = create_scheduler(optimizer, args.sched_type, num_sched_steps, num_warmup_steps)

    # Enable fp16/distributed training
    if use_gpu:
        if args.fp16:
            amp.register_half_function(torch, 'einsum')
            student_model, optimizer = amp.initialize(student_model, optimizer, opt_level=args.opt_level)
            if args.local_rank == 0:
                logging.info('Using fp16 training mode')
        if args.distributed:
            teacher_model = DDP(teacher_model, delay_allreduce=True)
            student_model = DDP(student_model, delay_allreduce=True)
        else:
            teacher_model = nn.DataParallel(teacher_model)
            student_model = nn.DataParallel(student_model)

    # Load model weights
    ckpt_path = args.teacher_pretrain_path
    if ckpt_path:
        if os.path.exists(ckpt_path):
            load_pretrain_state_dict(args.teacher_model, teacher_model, ckpt_path, use_gpu)
            if args.local_rank == 0:
                logging.info('Loaded teacher pretrained state dict from \'{}\''.format(ckpt_path))
        else:
            if args.local_rank == 0:
                logging.info('Teacher pretrained state dict is not found in \'{}\''.format(ckpt_path))

    ckpt_path = args.student_pretrain_path
    if ckpt_path:
        if os.path.exists(ckpt_path):
            load_pretrain_state_dict(args.student_model, student_model, ckpt_path, use_gpu, is_finetune=True)
            if args.local_rank == 0:
                logging.info('Loaded student pretrained state dict from \'{}\''.format(ckpt_path))
        else:
            if args.local_rank == 0:
                logging.info('Student pretrained state dict is not found in \'{}\''.format(ckpt_path))

    ckpt_path = args.student_multi_task_pretrain_path
    if ckpt_path:
        if os.path.exists(ckpt_path):
            task_id = 0
            load_multi_task_state_dict(student_model, ckpt_path, task_id, is_finetune=True, load_pred=False)
            if args.local_rank == 0:
                logging.info('Loaded student multi-task pretrained state dict from \'{}\''.format(ckpt_path))
        else:
            if args.local_rank == 0:
                logging.info('Student multi-task pretrained state dict is not found in \'{}\''.format(ckpt_path))

    ckpt_path = args.student_pretrain_path_6_540_to_6_360
    if ckpt_path:
        if os.path.exists(ckpt_path):
            load_supernet_state_dict_6_540_to_6_360(student_model, ckpt_path)
            if args.local_rank == 0:
                logging.info('Loaded student pretrained state dict 6-540 to 6-360 from \'{}\''.format(ckpt_path))
        else:
            if args.local_rank == 0:
                logging.info('Student pretrained state dict 6-540 to 6-360 is not found in \'{}\''.format(ckpt_path))

    ckpt_path = args.student_pretrain_path_6_540_to_12_360
    if ckpt_path:
        if os.path.exists(ckpt_path):
            load_supernet_state_dict_6_540_to_12_360(student_model, ckpt_path)
            if args.local_rank == 0:
                logging.info('Loaded student pretrained state dict 6-540 to 12-360 from \'{}\''.format(ckpt_path))
        else:
            if args.local_rank == 0:
                logging.info('Student pretrained state dict 6-540 to 12-360 is not found in \'{}\''.format(ckpt_path))

    ckpt_path = args.student_resume_path
    if ckpt_path:
        if os.path.exists(ckpt_path):
            checkpoint = load_resume_state_dict(student_model, ckpt_path, optimizer, scheduler)
            args.start_epoch = checkpoint['epoch'] + 1
            if args.local_rank == 0:
                logging.info('Loaded student resume checkpoint from \'{}\''.format(ckpt_path))
                logging.info('Start epoch: {}'.format(args.start_epoch))
        else:
            if args.local_rank == 0:
                logging.info('Student resume checkpoint is not found in \'{}\''.format(ckpt_path))

    try:
        train(teacher_model, student_model, optimizer, scheduler, use_gpu)
    except KeyboardInterrupt:
        print('Keyboard interrupt (process {}/{})'.format(args.local_rank + 1, args.world_size))


def train(teacher_model, student_model, optimizer, scheduler, use_gpu):
    st_time = time.time()
    if args.local_rank == 0:
        logging.info('==> Start training')

    for epoch in range(args.start_epoch, args.total_epochs + 1):
        train_dataset, train_loader = create_pretrain_dataset(
            args.all_train_dir, epoch - 1, args.tokenizer, args.num_data_epochs, args.local_rank, args.batch_size,
            use_gpu, args.distributed, 'train', args.train_ratio, args.val_ratio, args.concate_data_dir)

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        train_epoch(teacher_model, student_model, epoch, optimizer, scheduler, train_loader, use_gpu)

        if epoch % args.save_freq == 0 or epoch == args.total_epochs:
            if args.local_rank == 0:
                state = {'state_dict': student_model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'epoch': epoch}
                ckpt_name = 'ckpt_ep' + str(epoch) + '.bin'
                save_checkpoint(state, args.exp_dir, ckpt_name, args.ckpt_keep_num)
                logging.info('Student state dict has been saved to \'{}\''.format(os.path.join(args.exp_dir, ckpt_name)))
        if args.local_rank == 0:
            logging.info('-' * 50)

    if args.local_rank == 0:
        elapsed = round(time.time() - st_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        logging.info('Finished, total training time (h:m:s): {}'.format(elapsed))


def train_epoch(teacher_model, student_model, epoch, optimizer, scheduler, data_loader, use_gpu):
    teacher_model.eval()
    student_model.train()

    losses, train_time, data_time = [AverageMeter() for _ in range(3)]
    ffn_losses, attn_losses = [AverageMeter() for _ in range(2)]
    st_time = time.time()

    def _update_losses(all_losses, loss, data_size):
        if args.distributed:
            loss = reduce_tensor(loss.detach(), args.world_size)
        all_losses.update(loss.item(), data_size)

    for batch_idx, data in enumerate(data_loader):
        data_time.update(time.time() - st_time)
        if use_gpu:
            data = [data_.cuda() for data_ in data]
        token_ids, segment_ids, position_ids, attn_mask, lm_labels = data

        with torch.no_grad():
            teacher_outputs = teacher_model(token_ids, segment_ids, position_ids, attn_mask)
        if args.student_model in nas_bert_models:
            student_outputs = student_model(token_ids, segment_ids, position_ids, attn_mask, *args.ffn_arch)
        else:
            student_outputs = student_model(token_ids, segment_ids, position_ids, attn_mask)
        loss, attn_loss, ffn_loss = calc_distil_losses(teacher_outputs, student_outputs, use_gpu)

        optimizer.zero_grad()
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        _update_losses(losses, loss, token_ids.size(0))
        _update_losses(attn_losses, attn_loss, token_ids.size(0))
        _update_losses(ffn_losses, ffn_loss, token_ids.size(0))

        if use_gpu:
            torch.cuda.synchronize()
        train_time.update(time.time() - st_time)

        if args.local_rank == 0 and \
                (batch_idx == 0 or (batch_idx + 1) % args.disp_freq == 0 or batch_idx + 1 == len(data_loader)):
            lr = scheduler.get_lr()[0]
            logging.info('Epoch: [{}/{}][{}/{}]\t'
                         'LR: {:.2e}\t'
                         'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Attn and ffn loss: {attn_loss.val:.4f} {ffn_loss.val:.4f} ({attn_loss.avg:.4f} {ffn_loss.avg:.4f})\t'
                         'Train time: {train_time.val:.4f}s ({train_time.avg:.4f}s)\t'
                         'Load data time: {data_time.val:.4f}s ({data_time.avg:.4f}s)'
                         .format(epoch, args.total_epochs, batch_idx + 1, len(data_loader), lr,
                                 loss=losses, attn_loss=attn_losses, ffn_loss=ffn_losses,
                                 train_time=train_time, data_time=data_time))

        st_time = time.time()


def calc_distil_losses(teacher_outputs, student_outputs, use_gpu):
    teacher_pred_logits, teacher_attn_outputs, teacher_ffn_outputs = teacher_outputs
    student_pred_logits, student_attn_outputs, student_ffn_outputs = student_outputs

    def _replace_attn_mask(attn_output):
        replace_values = torch.zeros_like(attn_output)
        if use_gpu:
            replace_values = replace_values.cuda()
        attn_output = torch.where(attn_output <= -1e2, replace_values, attn_output)
        return attn_output

    mse_loss = nn.MSELoss()
    attn_loss, ffn_loss = 0, 0
    ffn_loss += mse_loss(teacher_ffn_outputs[0], student_ffn_outputs[0])
    for layer_id in range(args.num_student_layers):
        teacher_layer_id = (layer_id + 1) * args.teacher_interval - 1
        attn_loss += mse_loss(_replace_attn_mask(teacher_attn_outputs[teacher_layer_id]),
                              _replace_attn_mask(student_attn_outputs[layer_id]))
        ffn_loss += mse_loss(teacher_ffn_outputs[teacher_layer_id + 1], student_ffn_outputs[layer_id + 1])

    hidden_loss = attn_loss + ffn_loss
    return hidden_loss, attn_loss, ffn_loss


if __name__ == '__main__':
    main()
