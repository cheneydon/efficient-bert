import os
import numpy as np
import torch
import torch.nn as nn
import logging
import argparse
import time
import datetime
from copy import deepcopy
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from deap import gp
from datasets import glue_tasks, squad_tasks, SquadResult, glue_train_tasks_to_ids
from models import select_config, select_model, nas_bert_models
from tokenizers import select_tokenizer
from metrics import compute_glue_metrics, compute_squad_metrics, all_glue_select_metrics, all_squad_select_metrics
from utils import AverageMeter, register_custom_ops, register_custom_ops3, set_seeds, setup_logger, get_entire_linear_idx, \
    get_entire_params, calc_params, reduce_tensor, soft_cross_entropy, save_checkpoint, load_pretrain_state_dict, \
    load_resume_state_dict, load_multi_task_state_dict, create_optimizer, create_scheduler, create_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--distributed', action='store_true', help='distributed mode')
parser.add_argument('--fp16', action='store_true', help='mixed precision training mode')
parser.add_argument('--opt_level', default='O1', type=str, help='fp16 optimization level')
parser.add_argument('--gpu_devices', default='0', type=str, help='available gpu devices')
parser.add_argument('--seed', default=42, type=int, help='seed')

parser.add_argument('--lowercase', action='store_true', help='whether to do lowercase')
parser.add_argument('--official_pretrain', action='store_true', help='whether to use official pretrain ckpt')
parser.add_argument('--temperature', default=1, type=float, help='temperature for soft cross entropy loss')
parser.add_argument('--hidden_ratio', default=1, type=float, help='ratio for hidden loss')
parser.add_argument('--pred_ratio', default=1, type=float, help='ratio for prediction loss')
parser.add_argument('--ffn_expr', default=[], nargs='+', help='feed-forward network expression')
parser.add_argument('--teacher_model', default='bert_base', type=str, help='teacher model name')
parser.add_argument('--student_model', default='tiny_bert', type=str, help='student model name')

parser.add_argument('--task', default='sst-2', type=str, help='task name')
parser.add_argument('--data_dir', default='', type=str, help='task dataset directory')
parser.add_argument('--vocab_path', default='', type=str, help='path to pretrained vocabulary file')
parser.add_argument('--merge_path', default='', type=str, help='path to pretrained merge file (for roberta)')
parser.add_argument('--max_seq_len', default=128, type=int, help='max length of input sequences')
parser.add_argument('--max_query_len', default=64, type=int, help='max length of input questions (for squad) or question-answer pairs (for multi-choice tasks)')
parser.add_argument('--trunc_stride', default=32, type=int, help='context truncate stride (for squad)')
parser.add_argument('--n_best_size', default=20, type=int, help='total number of top-n best predictions to generate (for squad)')
parser.add_argument('--max_answer_len', default=30, type=int, help='maximum length of an answer that can be generated (for squad)')
parser.add_argument('--null_score_diff_threshold', default=0, type=float, help='if null_score - best_non_null is greater than the threshold predict null (for squad)')

parser.add_argument('--start_epoch', default=1, type=int, help='start epoch (default is 1)')
parser.add_argument('--total_epochs', default=10, type=int, help='total epochs')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--lr', default=5e-5, type=float, help='initial learning rate')
parser.add_argument('--optim_type', default='adamw', type=str, help='optimizer type')
parser.add_argument('--sched_type', default='step', type=str, help='lr scheduler type')
parser.add_argument('--warmup_proportion', default=0.1, type=float, help='proportion of warmup steps')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay')
parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max gradient norm')
parser.add_argument('--disp_freq', default=50, type=int, help='display step frequency')
parser.add_argument('--val_freq', default=50, type=int, help='validate step frequency')
parser.add_argument('--ckpt_keep_num', default=1, type=int, help='max number of checkpoint files to keep')

parser.add_argument('--teacher_pretrain_path', default='', type=str, help='path to pretrained teacher state dict')
parser.add_argument('--student_pretrain_path', default='', type=str, help='path to pretrained student state dict')
parser.add_argument('--student_multi_task_pretrain_path', default='', type=str, help='path to multi-task pretrained student state dict')
parser.add_argument('--student_resume_path', default='', type=str, help='path to student resume checkpoint')
parser.add_argument('--cache_dir', default='./cache', type=str, help='cache directory to save processed dataset')
parser.add_argument('--exp_dir', default='./exp/tmp/', type=str, help='experiment directory')
parser.add_argument('--local_rank', default=0, type=int, help='DDP local rank')
parser.add_argument('--world_size', default=1, type=int, help='DDP world size')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices


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
    teacher_model = select_model(args.teacher_model, args.lowercase, args.task, return_hid=True)
    student_model = select_model(args.student_model, args.lowercase, args.task, return_hid=True)
    args.tokenizer = select_tokenizer(
        args.teacher_model, args.lowercase, args.task, args.vocab_path, args.max_seq_len, args.max_query_len, args.merge_path)
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

    # Create dataset
    dev_distributed = False if args.task in squad_tasks else args.distributed  # do not support distributed evaluation for squad tasks
    _, _, _, train_loader = create_dataset(
        args.student_model, args.task, args.data_dir, args.tokenizer, args.max_seq_len, args.max_query_len,
        args.trunc_stride, args.batch_size, use_gpu, args.distributed, 'train', args.local_rank, args.cache_dir)
    args.dev_examples, args.dev_encoded_inputs, dev_dataset, dev_loader = create_dataset(
        args.student_model, args.task, args.data_dir, args.tokenizer, args.max_seq_len, args.max_query_len,
        args.trunc_stride, args.batch_size * 4, use_gpu, dev_distributed, 'dev', args.local_rank, args.cache_dir)

    # Create optimization tools
    args.num_sched_steps = len(train_loader) * args.total_epochs
    args.num_warmup_steps = int(args.num_sched_steps * args.warmup_proportion)
    optimizer = create_optimizer(student_model, args.optim_type, args.lr, args.weight_decay, args.momentum)
    scheduler = create_scheduler(optimizer, args.sched_type, args.num_sched_steps, args.num_warmup_steps)

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
            load_pretrain_state_dict(args.teacher_model, teacher_model, ckpt_path, use_gpu, is_finetune=True)
            if args.local_rank == 0:
                logging.info('Loaded teacher pretrained state dict from \'{}\''.format(ckpt_path))
        else:
            if args.local_rank == 0:
                logging.info('Teacher pretrained state dict is not found in \'{}\''.format(ckpt_path))

    ckpt_path = args.student_pretrain_path
    if ckpt_path:
        if os.path.exists(ckpt_path):
            if args.official_pretrain:
                load_pretrain_state_dict(args.student_model, student_model, ckpt_path, use_gpu)
            else:
                load_pretrain_state_dict(args.student_model, student_model, ckpt_path, use_gpu, is_finetune=True)
            if args.local_rank == 0:
                logging.info('Loaded student pretrained state dict from \'{}\''.format(ckpt_path))
        else:
            if args.local_rank == 0:
                logging.info('Student pretrained state dict is not found in \'{}\''.format(ckpt_path))

    ckpt_path = args.student_multi_task_pretrain_path
    if ckpt_path:
        if os.path.exists(ckpt_path):
            if args.task == 'wnli':
                task_id = glue_train_tasks_to_ids['qnli']
            else:
                task_id = glue_train_tasks_to_ids[args.task]
            load_multi_task_state_dict(student_model, ckpt_path, task_id, is_finetune=True)
            if args.local_rank == 0:
                logging.info('Loaded student multi-task pretrained state dict from \'{}\''.format(ckpt_path))
        else:
            if args.local_rank == 0:
                logging.info('Student multi-task pretrained state dict is not found in \'{}\''.format(ckpt_path))

    ckpt_path = args.student_resume_path
    if ckpt_path:
        if os.path.exists(ckpt_path):
            checkpoint = load_resume_state_dict(student_model, ckpt_path, optimizer, scheduler)
            args.start_epoch = checkpoint['epoch'] + 1
            if args.local_rank == 0:
                logging.info('Loaded student resume checkpoint from \'{}\''.format(ckpt_path))
                logging.info('Start epoch: {}\tMetrics: {}'.format(args.start_epoch, checkpoint['metrics']))
        else:
            if args.local_rank == 0:
                logging.info('No checkpoint found in \'{}\''.format(ckpt_path))

    try:
        train(teacher_model, student_model, optimizer, scheduler, train_loader, dev_loader, use_gpu)
    except KeyboardInterrupt:
        print('Keyboard interrupt (process {}/{})'.format(args.local_rank + 1, args.world_size))


def train(teacher_model, student_model, optimizer, scheduler, train_loader, dev_loader, use_gpu):
    st_time = time.time()
    if args.local_rank == 0:
        logging.info('==> Start training')

    best_results = [0, None, None]
    for epoch in range(args.start_epoch, args.total_epochs + 1):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        best_results = train_epoch(
            teacher_model, student_model, epoch, optimizer, scheduler, train_loader, dev_loader, best_results, use_gpu)
        sel_metric, metrics = validate(student_model, dev_loader, use_gpu)

        if args.local_rank == 0:
            logging.info('-' * 50)
            best_sel_metric, best_metrics, best_idx = best_results
            state = {'state_dict': student_model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(),
                     'metrics': metrics,
                     'epoch': epoch}
            ckpt_name = 'ckpt_ep' + str(epoch) + '.bin'
            save_checkpoint(state, args.exp_dir, ckpt_name, args.ckpt_keep_num)
            if best_idx is not None:
                logging.info('Best select metric {} found in epoch {} step {}'.format(best_sel_metric, best_idx[0], best_idx[1]))
            logging.info('Best metrics: {}, current metrics: {}'.format(best_metrics, metrics))
            logging.info('Student state dict has been saved to \'{}\''.format(os.path.join(args.exp_dir, ckpt_name)))
            logging.info('-' * 50)

    if args.local_rank == 0:
        elapsed = round(time.time() - st_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        logging.info('Finished, total training time (h:m:s): {}'.format(elapsed))


def train_epoch(teacher_model, student_model, epoch, optimizer, scheduler, train_loader, dev_loader, best_results, use_gpu):
    teacher_model.eval()
    student_model.train()

    best_sel_metric, best_metrics, best_idx = best_results
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

        if args.task in glue_tasks:
            token_ids, segment_ids, position_ids, attn_mask, labels = data
        else:  # args.task in squad_tasks
            token_ids, segment_ids, position_ids, attn_mask, start_positions, end_positions, cls_index, p_mask, is_impossible = data
            labels = None

        with torch.no_grad():
            teacher_outputs = teacher_model(token_ids, segment_ids, position_ids, attn_mask)
        if args.student_model in nas_bert_models:
            student_outputs = student_model(token_ids, segment_ids, position_ids, attn_mask, *args.ffn_arch)
        else:
            student_outputs = student_model(token_ids, segment_ids, position_ids, attn_mask)
        loss, attn_loss, ffn_loss, pred_loss = calc_distil_losses(teacher_outputs, student_outputs, use_gpu, labels, attn_mask)

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
        _update_losses(pred_losses, pred_loss, token_ids.size(0))

        if use_gpu:
            torch.cuda.synchronize()
        train_time.update(time.time() - st_time)

        if args.local_rank == 0 and \
                (batch_idx == 0 or (batch_idx + 1) % args.disp_freq == 0 or batch_idx + 1 == len(train_loader)):
            lr = scheduler.get_lr()[0]
            logging.info('Epoch: [{}/{}][{}/{}]\t'
                         'LR: {:.2e}\t'
                         'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Attn, ffn and pred loss: {attn_loss.val:.4f} {ffn_loss.val:.4f} {pred_loss.val:.4f} '
                         '({attn_loss.avg:.4f} {ffn_loss.avg:.4f} {pred_loss.avg:.4f})\t'
                         'Train time: {train_time.val:.4f}s ({train_time.avg:.4f}s)\t'
                         'Load data time: {data_time.val:.4f}s ({data_time.avg:.4f}s)'
                         .format(epoch, args.total_epochs, batch_idx + 1, len(train_loader), lr,
                                 loss=losses, attn_loss=attn_losses, ffn_loss=ffn_losses, pred_loss=pred_losses,
                                 train_time=train_time, data_time=data_time))

        if (batch_idx + 1) % args.val_freq == 0 or batch_idx + 1 == len(train_loader):
            sel_metric, metrics = validate(student_model, dev_loader, use_gpu)
            is_best = sel_metric > best_sel_metric
            if is_best:
                best_sel_metric = sel_metric
                best_metrics = metrics
                best_idx = [epoch, batch_idx + 1]

                if args.local_rank == 0:
                    state = {'state_dict': student_model.state_dict(),
                             'metrics': best_metrics,
                             'epoch': best_idx[0],
                             'step': best_idx[1]}
                    save_path = os.path.join(args.exp_dir, 'best_model.bin')
                    torch.save(state, save_path)
                    logging.info('Best metric found: {}'.format(best_sel_metric))

            student_model.train()
        st_time = time.time()
    return best_sel_metric, best_metrics, best_idx


def validate(model, data_loader, use_gpu, verbose=False):
    model.eval()

    val_time = AverageMeter()
    all_metrics = {}
    st_time = time.time()

    with torch.no_grad():
        if args.task in glue_tasks:
            for batch_idx, data in enumerate(data_loader):
                if use_gpu:
                    data = [data_.cuda() for data_ in data]
                token_ids, segment_ids, position_ids, attn_mask, labels = data
                if args.student_model in nas_bert_models:
                    outputs = model(token_ids, segment_ids, position_ids, attn_mask, *args.ffn_arch)
                else:
                    outputs = model(token_ids, segment_ids, position_ids, attn_mask)
                preds, labels = outputs[0].detach().cpu().numpy(), labels.detach().cpu().numpy()
                preds = np.squeeze(preds) if args.task == 'sts-b' else np.argmax(preds, axis=1)
                metrics = compute_glue_metrics(args.task, preds, labels)

                # Average metrics
                if args.distributed:
                    for k, v in metrics.items():
                        metrics[k] = reduce_tensor(torch.tensor(metrics[k], dtype=torch.float64).cuda(), args.world_size).cpu().numpy()

                for k, v in metrics.items():
                    if all_metrics.get(k) is None:
                        all_metrics[k] = AverageMeter()
                    all_metrics[k].update(metrics[k], token_ids.size(0))
                avg_metrics = {k: v.avg for k, v in all_metrics.items()}

                if use_gpu:
                    torch.cuda.synchronize()
                val_time.update(time.time() - st_time)

                if verbose and args.local_rank == 0 and \
                        (batch_idx == 0 or (batch_idx + 1) % args.disp_freq == 0 or batch_idx + 1 == len(data_loader)):
                    logging.info('Iter: [{}/{}]\tVal time: {:.4f}s\tMetrics: {}'
                                 .format(batch_idx + 1, len(data_loader), val_time.avg, avg_metrics))
                st_time = time.time()

            sel_metric = avg_metrics[all_glue_select_metrics[args.task]]
            return sel_metric, avg_metrics

        else:  # args.task in squad_tasks
            all_results = []
            if verbose and args.local_rank == 0:
                logging.info('Collecting predicted results')
            for batch_idx, data in enumerate(data_loader):
                if use_gpu:
                    data = [data_.cuda() for data_ in data]
                token_ids, segment_ids, position_ids, attn_mask, example_indicies, cls_index, p_mask = data
                if args.student_model in nas_bert_models:
                    outputs = model(token_ids, segment_ids, position_ids, attn_mask, *args.ffn_arch)
                else:
                    outputs = model(token_ids, segment_ids, position_ids, attn_mask)

                for i, example_idx in enumerate(example_indicies):
                    encoded_input = args.dev_encoded_inputs[example_idx.item()]
                    unique_id = int(encoded_input.unique_id)
                    output = [output[i].detach().cpu().tolist() for output in outputs[:2]]
                    start_logits, end_logits = output
                    all_results.append(SquadResult(unique_id, start_logits, end_logits))

            if verbose and args.local_rank == 0:
                logging.info('Calculating metrics')
            metrics = compute_squad_metrics(
                args.student_model, args.task, args.tokenizer, args.dev_examples, args.dev_encoded_inputs, all_results,
                args.n_best_size, args.max_answer_len, args.null_score_diff_threshold)

            if use_gpu:
                torch.cuda.synchronize()
            val_time = round(time.time() - st_time)
            val_time = str(datetime.timedelta(seconds=val_time))
            if verbose and args.local_rank == 0:
                logging.info('Val time (h:m:s): {}\tMetrics: {}'.format(val_time, metrics))

            sel_metric = metrics[all_squad_select_metrics[args.task]]
            return sel_metric, metrics


def calc_distil_losses(teacher_outputs, student_outputs, use_gpu, labels=None, attn_mask=None):
    if args.task in glue_tasks:
        teacher_pred_logits, teacher_attn_outputs, teacher_ffn_outputs = teacher_outputs
        student_pred_logits, student_attn_outputs, student_ffn_outputs = student_outputs
    else:  # args.task in squad_tasks
        teacher_start_logits, teacher_end_logits, teacher_attn_outputs, teacher_ffn_outputs = teacher_outputs
        student_start_logits, student_end_logits, student_attn_outputs, student_ffn_outputs = student_outputs

        attn_mask = deepcopy(attn_mask)
        attn_mask[:, 0] = 1
        teacher_start_logits += attn_mask * -10000.0
        teacher_end_logits += attn_mask * -10000.0
        student_start_logits += attn_mask * -10000.0
        student_end_logits += attn_mask * -10000.0

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
    if args.task in glue_tasks:
        pred_loss = mse_loss(student_pred_logits, labels) if args.task == 'sts-b' else \
            soft_cross_entropy(student_pred_logits, teacher_pred_logits, args.temperature)
    else:  # args.task in squad_tasks
        start_loss = soft_cross_entropy(student_start_logits, teacher_start_logits, args.temperature, mean=False)
        end_loss = soft_cross_entropy(student_end_logits, teacher_end_logits, args.temperature, mean=False)
        pred_loss = (start_loss + end_loss).mean()

    total_loss = args.hidden_ratio * hidden_loss + args.pred_ratio * pred_loss
    return total_loss, attn_loss, ffn_loss, pred_loss


if __name__ == '__main__':
    main()
