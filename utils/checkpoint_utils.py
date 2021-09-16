import os
import re
import shutil
import torch
import numpy as np
import models
from collections import OrderedDict


def save_checkpoint(state, save_dir, ckpt_name, keep_num, is_best=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, ckpt_name)
    torch.save(state, save_path)
    if is_best:
        shutil.copy(save_path, os.path.join(save_dir, 'best_model.bin'))

    ckpt_head = re.split(r'\d+', ckpt_name)[0]
    all_ckpt = np.array([file for file in os.listdir(save_dir) if re.match(ckpt_head, file) is not None])
    all_idx = np.int32([re.findall(r'\d+', ckpt)[0] for ckpt in all_ckpt])
    sorted_ckpt = all_ckpt[np.argsort(all_idx)[::-1]]
    remove_path = [os.path.join(save_dir, name) for name in sorted_ckpt[keep_num:]]
    for path in remove_path:
        os.remove(path)


def load_pretrain_state_dict(model_name, model, state_dict_path, add_module=False, load_lm_weights=False, is_finetune=False):
    raw_state_dict = torch.load(state_dict_path, map_location='cpu')

    if is_finetune:
        raw_state_dict = raw_state_dict['state_dict']
        new_state_dict = {}
        for n, p in raw_state_dict.items():
            if re.search(r'lm_head', n) is None:
                n = re.sub(r'module\.', '', n)
                n = 'module.' + n if add_module else n
                new_state_dict[n] = p
    else:
        new_state_dict = {}
        for n, p in raw_state_dict.items():
            # Bert & Roberta & TinyBert
            if model_name in models.bert_models + models.roberta_models and re.search(r'pooler|cls', n) is None:
                if re.match(r'roberta', n) is not None and re.search(r'token_type_embeddings', n) is not None:
                    continue
                n = re.sub(r'(bert|roberta|layer|self)\.', '', n)
                n = re.sub(r'word_embeddings', 'token_embeddings', n)
                n = re.sub(r'token_type_embeddings', 'segment_embeddings', n)
                n = re.sub(r'LayerNorm', 'layernorm', n)
                n = re.sub(r'gamma', 'weight', n)
                n = re.sub(r'beta', 'bias', n)
                n = re.sub(r'attention\.output', 'attention', n)
                n = re.sub(r'intermediate\.dense', 'ffn.dense1', n)
                n = re.sub(r'output\.dense', 'ffn.dense2', n)
                n = re.sub(r'output', 'ffn', n)
                n = 'module.' + n if add_module else n
                new_state_dict[n] = p

            # Xlnet
            if model_name in models.xlnet_models and re.search(r'mask_emb', n) is None:
                n = re.sub(r'layer\.', '', n)
                n = re.sub(r'transformer\.word_embedding', 'base_model.token_embeddings', n)
                n = re.sub(r'transformer', 'encoder', n)
                n = re.sub(r'seg_embed', 'seg_mat', n)
                n = re.sub(r'layer_norm', 'layernorm', n)
                n = re.sub(r'ff', 'ffn', n)
                n = re.sub(r'layer_1', 'dense1', n)
                n = re.sub(r'layer_2', 'dense2', n)
                n = 'module.' + n if add_module else n
                new_state_dict[n] = p

            # Gpt2
            if model_name in models.gpt_models:
                n = re.sub(r'wte', 'embeddings.token_embeddings', n)
                n = re.sub(r'wpe', 'embeddings.position_embeddings', n)
                n = re.sub(r'h\.', 'encoder.', n)
                n = re.sub(r'attn', 'attention', n)
                n = re.sub(r'c_attention', 'c_attn', n)
                n = re.sub(r'mlp', 'ffn', n)
                n = re.sub(r'ln_1', 'attention.layernorm', n)
                n = re.sub(r'ln_2', 'ffn.layernorm', n)
                n = re.sub(r'ln_f', 'layernorm', n)
                n = 'module.' + n if add_module else n
                if re.search(r'token_embeddings\.weight', n) is not None:
                    new_weights = model.state_dict()[n]
                    new_weights[:p.size(0)] = p
                    p = new_weights
                new_state_dict[n] = p

            # Bert LM weights
            if model_name in models.bert_models and load_lm_weights and re.match(r'cls\.predictions', n) is not None:
                n = re.sub(r'cls\.predictions', 'lm_head', n)
                n = re.sub(r'lm_head\.bias', 'lm_head.lm_bias', n)
                n = re.sub(r'transform\.', '', n)
                n = re.sub(r'LayerNorm\.gamma', 'layernorm.weight', n)
                n = re.sub(r'LayerNorm\.beta', 'layernorm.bias', n)
                n = re.sub(r'decoder', 'lm_decoder', n)
                n = 'module.' + n if add_module else n
                new_state_dict[n] = p

    model_state_dict = model.state_dict()
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)


def load_resume_state_dict(model, resume_path, optimizer=None, scheduler=None):
    checkpoint = torch.load(resume_path, map_location='cpu')
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except RuntimeError:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    return checkpoint


def load_multi_task_state_dict(model, state_dict_path, task_id, is_finetune=False, load_pred=True):
    raw_state_dict = torch.load(state_dict_path, map_location='cpu')['state_dict']
    new_state_dict = {}
    for n, p in raw_state_dict.items():
        if is_finetune:
            if load_pred:
                res = re.search(r'classifiers.(\d+)', n)
                if res is not None:
                    cur_task_id = int(res.groups()[0])
                    if cur_task_id == task_id:
                        n = n[:res.start()] + 'classifier' + n[res.end():]
                        new_state_dict[n] = p
                else:
                    new_state_dict[n] = p
            else:
                if re.search(r'classifiers', n) is None and re.search(r'cls_pooler', n) is None:
                    new_state_dict[n] = p
        else:
            res = re.search(r'classifier', n)
            if res is not None:
                n = n[:res.start()] + 'classifiers.' + str(task_id) + n[res.end():]
            new_state_dict[n] = p

    model_state_dict = model.state_dict()
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)


def load_supernet_state_dict_6_540_to_6_360(model, state_dict_path, raw_dim=540, new_dim=360):
    def _adj_param_dim(name, param):
        if re.search(r'embeddings', name) is not None:
            if re.search(r'segment_embeddings.weight', name) is not None:
                assert param.size() == torch.Size([2, raw_dim])
                param = param[:, :new_dim]
            elif re.search(r'position_embeddings.weight', name) is not None:
                assert param.size() == torch.Size([512, raw_dim])
                param = param[:, :new_dim]
            elif re.search(r'dense.weight', name) is not None:
                assert param.size() == torch.Size([raw_dim, 384])
                param = param[:new_dim, :]
            else:
                if re.search(r'token_embeddings', name) is None:
                    assert param.size() == torch.Size([raw_dim])
                    param = param[:new_dim]
        elif re.search(r'encoder', name) is not None:
            if re.search(r'weight', name) is not None and re.search(r'layernorm', name) is None:
                assert param.size() == torch.Size([raw_dim, raw_dim])
                param = param[:new_dim, :new_dim]
            else:
                assert param.size() == torch.Size([raw_dim])
                param = param[:new_dim]
        elif re.search(r'fit_dense', name) is not None:
            if re.search(r'weight', name) is not None:
                assert param.size() == torch.Size([768, raw_dim])
                param = param[:, :new_dim]
        else:
            raise KeyError('Do not support adjusting param dim of {}'.format(name))
        return param

    raw_state_dict = torch.load(state_dict_path, map_location='cpu')['state_dict']
    new_state_dict = {}
    for n, p in raw_state_dict.items():
        new_p = _adj_param_dim(n, p)
        new_state_dict[n] = new_p

    model_state_dict = model.state_dict()
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)


def load_supernet_state_dict_6_540_to_12_360(model, state_dict_path, raw_dim=540, new_dim=360):
    def _adj_param_dim(name, param):
        if re.search(r'embeddings', name) is not None:
            if re.search(r'segment_embeddings.weight', name) is not None:
                assert param.size() == torch.Size([2, raw_dim])
                param = param[:, :new_dim]
            elif re.search(r'position_embeddings.weight', name) is not None:
                assert param.size() == torch.Size([512, raw_dim])
                param = param[:, :new_dim]
            elif re.search(r'dense.weight', name) is not None:
                assert param.size() == torch.Size([raw_dim, 384])
                param = param[:new_dim, :]
            else:
                if re.search(r'token_embeddings', name) is None:
                    assert param.size() == torch.Size([raw_dim])
                    param = param[:new_dim]
        elif re.search(r'encoder', name) is not None:
            if re.search(r'weight', name) is not None and re.search(r'layernorm', name) is None:
                assert param.size() == torch.Size([raw_dim, raw_dim])
                param = param[:new_dim, :new_dim]
            else:
                assert param.size() == torch.Size([raw_dim])
                param = param[:new_dim]
        elif re.search(r'fit_dense', name) is not None:
            if re.search(r'weight', name) is not None:
                assert param.size() == torch.Size([768, raw_dim])
                param = param[:, :new_dim]
        else:
            raise KeyError('Do not support adjusting param dim of {}'.format(name))
        return param

    raw_state_dict = torch.load(state_dict_path, map_location='cpu')['state_dict']
    new_state_dict = {}
    for n, p in raw_state_dict.items():
        new_p = _adj_param_dim(n, p)
        res = re.search(r'encoder.(\d+)', n)
        if res is not None:
            raw_layer_id = int(res.groups()[0])
            new_layer_id1 = raw_layer_id * 2
            new_layer_id2 = new_layer_id1 + 1
            new_n1 = n[:res.start()] + 'encoder.' + str(new_layer_id1) + n[res.end():]
            new_n2 = n[:res.start()] + 'encoder.' + str(new_layer_id2) + n[res.end():]
            new_state_dict[new_n1] = new_p
            new_state_dict[new_n2] = new_p
        else:
            new_state_dict[n] = new_p

    model_state_dict = model.state_dict()
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)
