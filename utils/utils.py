import sys
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from thop import profile


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%d %I:%M:%S')
    fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


def set_seeds(seed, use_gpu=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_gpu:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def count_flops_params(model, inputs):
    if not isinstance(inputs, tuple):
        inputs = (inputs, )
    flops, params = profile(model, inputs=inputs, verbose=False)
    return flops, params


def calc_params(model):
    return sum(p.numel() for p in model.parameters())


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= n
    return rt


def soft_cross_entropy(predicts, targets, t=1, mean=True):
    student_prob = F.log_softmax(predicts / t, dim=-1)
    teacher_prob = F.softmax(targets / t, dim=-1)
    out = -teacher_prob * student_prob
    if mean:
        out = out.mean()
    return out
