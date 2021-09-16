import os
import numpy as np
import torch
import torch.nn as nn
import json
import logging
import argparse
import datetime
from deap import gp
from tqdm import tqdm
from models import select_config, select_model, xlnet_models, nas_bert_models
from tokenizers import select_tokenizer
from datasets import glue_tasks, glue_labels, squad_tasks, multi_choice_tasks, SquadResult
from metrics import compute_squad_metrics
from utils import set_seeds, setup_logger, calc_params, load_resume_state_dict, create_dataset, \
    register_custom_ops, register_custom_ops3, get_entire_linear_idx, get_entire_params

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_devices', default='4', type=str, help='available gpu devices')
parser.add_argument('--seed', default=42, type=int, help='seed')

parser.add_argument('--lowercase', action='store_true', help='whether to do lowercase')
parser.add_argument('--model_name', default='bert_base', type=str, help='model name')
parser.add_argument('--ffn_expr', default=[], nargs='+', help='feed-forward network expression')
parser.add_argument('--task', default='mnli', type=str, help='task name')
parser.add_argument('--data_dir', default='', type=str, help='task dataset directory')
parser.add_argument('--vocab_path', default='', type=str, help='path to pretrained vocabulary file')
parser.add_argument('--merge_path', default='', type=str, help='path to pretrained merge file (for roberta)')
parser.add_argument('--max_seq_len', default=128, type=int, help='max length of input sequences')
parser.add_argument('--max_query_len', default=64, type=int, help='max length of input questions (for squad) or question-answer pairs (for multi-choice tasks)')
parser.add_argument('--trunc_stride', default=32, type=int, help='context truncate stride (for squad)')
parser.add_argument('--n_best_size', default=20, type=int, help='total number of top-n best predictions to generate')
parser.add_argument('--max_answer_len', default=30, type=int, help='maximum length of an answer that can be generated')
parser.add_argument('--null_score_diff_threshold', default=0.0, type=float, help='if null_score - best_non_null is greater than the threshold predict null')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')

parser.add_argument('--resume_path', default='', type=str, help='path to resume checkpoint')
parser.add_argument('--result_dir', default='./test_results', type=str, help='test results directory')
parser.add_argument('--result_path', default='', type=str, help='test results saving path (for squad)')
parser.add_argument('--cache_dir', default='./cache', type=str, help='cache directory to save processed dataset')
parser.add_argument('--exp_dir', default='./exp/tmp/', type=str, help='experiment directory')
args = parser.parse_args()

if args.task not in squad_tasks:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices


def main():
    args.distributed = False  # Do not support distributed test
    args.local_rank = 0
    args.exp_dir = os.path.join(args.exp_dir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    setup_logger(args.exp_dir)
    logging.info(args)

    use_gpu = False
    if args.gpu_devices and torch.cuda.is_available():
        use_gpu = True
    if use_gpu:
        logging.info('Currently using GPU: {}'.format(args.gpu_devices))
    else:
        logging.info('Currently using CPU')
    set_seeds(args.seed, use_gpu)

    # Load model and tokenizer
    args.config = select_config(args.model_name, args.lowercase)
    model = select_model(args.model_name, args.lowercase, args.task)
    args.tokenizer = select_tokenizer(
        args.model_name, args.lowercase, args.task, args.vocab_path, args.max_seq_len, args.max_query_len, args.merge_path)
    if use_gpu:
        model = nn.DataParallel(model).cuda()
    logging.info('Model size: {:.2f}M'.format(calc_params(model) / 1e6))
    if args.model_name in nas_bert_models:
        logging.info('Sub model size: {:.2f}M'.format(get_entire_params(args.config.param_list, args.ffn_expr)))
    logging.info('Model config: {}'.format(args.config.__dict__))

    # Register custom operators for auto bert
    if args.model_name in nas_bert_models:
        if args.model_name == 'auto_tiny_bert':
            pset = register_custom_ops3()
        else:
            pset = register_custom_ops()
        if args.model_name == 'auto_bert_12':
            args.ffn_expr = np.reshape([[x, x] for x in args.ffn_expr], -1).tolist()
        entire_ffn_func = [gp.compile(expr, pset) for expr in args.ffn_expr]
        entire_linear_idx = get_entire_linear_idx(args.ffn_expr)
        args.ffn_arch = [entire_ffn_func, entire_linear_idx]

    # Create dataset
    test_examples, test_encoded_inputs, test_dataset, test_loader = create_dataset(
        args.model_name, args.task, args.data_dir, args.tokenizer, args.max_seq_len, args.max_query_len,
        args.trunc_stride, args.batch_size, use_gpu, args.distributed, 'test', args.local_rank, args.cache_dir)

    if args.resume_path:
        if os.path.exists(args.resume_path):
            checkpoint = load_resume_state_dict(model, args.resume_path)
            logging.info('Loaded checkpoint from \'{}\''.format(args.resume_path))
            logging.info('Metrics: {}'.format(checkpoint['metrics']))
        else:
            logging.info('No checkpoint found in \'{}\''.format(args.resume_path))

    logging.info('-' * 50)
    logging.info('==> Start evaluation')
    validate(model, test_examples, test_encoded_inputs, test_loader, use_gpu)
    logging.info('-' * 50)


def validate(model, examples, encoded_inputs, data_loader, use_gpu):
    model.eval()

    # Create results
    all_preds = []
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(data_loader)):
            if use_gpu:
                data = [data_.cuda() for data_ in data]

            if args.task in glue_tasks + multi_choice_tasks:
                token_ids, segment_ids, position_ids, attn_mask, labels = data
                if args.model_name in nas_bert_models:
                    outputs = model(token_ids, segment_ids, position_ids, attn_mask, *args.ffn_arch)
                else:
                    outputs = model(token_ids, segment_ids, position_ids, attn_mask)
                preds, labels = outputs.detach().cpu().numpy(), labels.detach().cpu().numpy()
                preds = np.squeeze(preds) if args.task == 'sts-b' else np.argmax(preds, axis=1)
                all_preds.extend(preds) if np.ndim(preds) > 0 else all_preds.append(preds)

            elif args.task in squad_tasks:
                token_ids, segment_ids, position_ids, attn_mask, example_indicies, cls_index, p_mask = data
                outputs = model(token_ids, segment_ids, position_ids, attn_mask.float(), cls_index=cls_index, p_mask=p_mask)

                for i, example_index in enumerate(example_indicies):
                    encoded_input = encoded_inputs[example_index.item()]
                    unique_id = int(encoded_input.unique_id)
                    output = [output[i].detach().cpu().tolist() for output in outputs]

                    if args.model_name in xlnet_models:
                        start_logits, start_top_index, end_logits, end_top_index, cls_logits = output
                    else:
                        start_logits, end_logits = output
                        start_top_index, end_top_index, cls_logits = None, None, None
                    all_preds.append(
                        SquadResult(unique_id, start_logits, end_logits, start_top_index, end_top_index, cls_logits))

    # Write results
    if args.task in glue_tasks:
        save_name_map = {'mnli': 'MNLI-m', 'mnli-mm': 'MNLI-mm', 'qqp': 'QQP', 'qnli': 'QNLI', 'sst-2': 'SST-2',
                         'cola': 'CoLA', 'sts-b': 'STS-B', 'mrpc': 'MRPC', 'rte': 'RTE', 'wnli': 'WNLI', 'ax': 'AX'}
        pred_map = {i: name for i, name in enumerate(glue_labels[args.task])}
        save_path = os.path.join(args.result_dir, save_name_map[args.task] + '.tsv')
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)

        with open(save_path, 'w') as f:
            f.write('index\tprediction\n')
            for id, pred in enumerate(all_preds):
                if args.task == 'sts-b':
                    if pred > 5:
                        pred = 5
                    f.write('%d\t%.3f\n' % (id, pred))
                else:
                    f.write('%d\t%s\n' % (id, pred_map[pred]))
        logging.info('Saved results to \'{}\''.format(save_path))

    if args.task in squad_tasks:
        logging.info('Creating answer texts')
        start_n_top, end_n_top = None, None
        if args.model_name in xlnet_models:
            start_n_top, end_n_top = args.config.start_n_top, args.config.end_n_top

        answers = compute_squad_metrics(
            args.model_name, args.task, args.tokenizer, examples, encoded_inputs, all_preds, args.n_best_size,
            args.max_answer_len, args.null_score_diff_threshold, start_n_top, end_n_top, return_text=True)

        with open(args.result_path, 'w') as f:
            json.dump(answers, f)
        logging.info('Saved results to \'{}\''.format(args.result_path))

    if args.task == 'swag':
        save_path = os.path.join(args.result_dir, args.task + '.csv')
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)

        with open(save_path, 'w') as f:
            f.write('pred\n')
            for pred in all_preds:
                f.write('%d\n' % pred)
        logging.info('Saved results to \'{}\''.format(save_path))


if __name__ == '__main__':
    main()
