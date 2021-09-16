import os
import re
import json
import argparse
import logging
import datetime
from pathlib import Path
from tokenizers import select_basic_tokenizer
from utils import create_pretrain_dataset, setup_logger


parser = argparse.ArgumentParser()
parser.add_argument('--lowercase', action='store_true', help='whether to do lowercase')
parser.add_argument('--teacher_model', default='bert_base', type=str, help='teacher model name')
parser.add_argument('--vocab_path', default='', type=str, help='path to pretrained vocabulary file')
parser.add_argument('--merge_path', default='', type=str, help='path to pretrained merge file (for roberta)')
parser.add_argument('--start_epoch', default=1, type=int, help='start epoch (default is 1)')
parser.add_argument('--total_epochs', default=10, type=int, help='total epochs')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--train_ratio', default=1, type=float, help='ratio of train dataset')
parser.add_argument('--val_ratio', default=0, type=float, help='ratio of val dataset')
parser.add_argument('--wiki_dir', default='', type=Path, help='directory to wikipedia dataset')
parser.add_argument('--book_dir', default='', type=Path, help='directory to bookcorpus dataset')
parser.add_argument('--concate_data_dir', default='', type=Path, help='directory to concatenated dataset')
parser.add_argument('--exp_dir', default='./exp/tmp/', type=str, help='experiment directory')
parser.add_argument('--local_rank', default=0, type=int, help='DDP local rank')
args = parser.parse_args()

args.exp_dir = os.path.join(args.exp_dir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
setup_logger(args.exp_dir)
if args.local_rank == 0:
    logging.info(args)

use_gpu = False
distributed = False
if args.book_dir and args.concate_data_dir:
    all_train_dir = [args.wiki_dir, args.book_dir]
else:
    all_train_dir = [args.wiki_dir]
tokenizer = select_basic_tokenizer(args.teacher_model, args.lowercase, args.vocab_path, args.merge_path)

all_num_data_epochs = []
total_examples = 0
for train_dir in all_train_dir:
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
total_num_data_epochs = all_num_data_epochs[0]

for epoch in range(args.start_epoch, args.total_epochs):
    train_dataset, train_loader = create_pretrain_dataset(
        all_train_dir, epoch - 1, tokenizer, total_num_data_epochs, args.local_rank, args.batch_size,
        use_gpu, distributed, 'train', args.train_ratio, args.val_ratio, args.concate_data_dir, read_only=False)
