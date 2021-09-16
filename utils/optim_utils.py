import math
import torch
from torch.optim.lr_scheduler import LambdaLR


def add_weight_decay(model, weight_decay, no_decay_param=()):
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay_param)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay_param)], 'weight_decay': 0.0}
    ]

    return optimizer_grouped_parameters


def create_optimizer(model, optim_type, lr, weight_decay, momentum=0.9):
    assert optim_type in ['sgd', 'adam', 'adamw']

    no_decay_param = ['bias', 'layernorm.weight']
    parameters = add_weight_decay(model, weight_decay, no_decay_param)

    if optim_type == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum)
    elif optim_type == 'adam':
        return torch.optim.Adam(parameters, lr=lr)
    elif optim_type == 'adamw':
        return torch.optim.AdamW(parameters, lr=lr)


def create_scheduler(optimizer, sched_type, total_steps, warmup_steps=0, num_cycles=.5, last_epoch=-1):
    assert sched_type in ['step', 'cosine']

    if sched_type == 'step':
        def lr_lambda(cur_step):
            if cur_step < warmup_steps:
                return float(cur_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - cur_step) / float(max(1, total_steps - warmup_steps)))
        return LambdaLR(optimizer, lr_lambda, last_epoch)

    elif sched_type == 'cosine':
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0., 0.5 * (1. + math.cos(math.pi * float(num_cycles) * 2. * progress)))
        return LambdaLR(optimizer, lr_lambda, last_epoch)
