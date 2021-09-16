import os
import csv
import logging
import torch
import datasets
import models
from tqdm import tqdm
from torch.utils.data.dataset import TensorDataset


class GlueExample(object):
    def __init__(self, text_a, text_b, label, id, task_id=None):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.id = id
        self.task_id = task_id


def read_tsv(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter='\t', quotechar=None)
        data = []
        for line in reader:
            data.append(line)
        return data


def create_glue_examples(task, glue_dir, split):
    file_name = split + '.tsv'
    if task == 'mrpc':
        data_path = os.path.join(glue_dir, 'MRPC', file_name)
        text_a_id, text_b_id, label_id = [3, 4, 0] if split != 'test' else [3, 4, None]
    elif task == 'mnli':
        data_path = os.path.join(glue_dir, 'MNLI', file_name if split == 'train' else split + '_matched.tsv')
        text_a_id, text_b_id, label_id = [8, 9, -1] if split != 'test' else [8, 9, None]
    elif task == 'mnli-mm':
        data_path = os.path.join(glue_dir, 'MNLI', file_name if split == 'train' else split + '_mismatched.tsv')
        text_a_id, text_b_id, label_id = [8, 9, -1] if split != 'test' else [8, 9, None]
    elif task == 'ax':
        data_path = os.path.join(glue_dir, 'MNLI', 'diagnostic.tsv')
        text_a_id, text_b_id, label_id = [1, 2, None]
    elif task == 'cola':
        data_path = os.path.join(glue_dir, 'CoLA', file_name)
        text_a_id, text_b_id, label_id = [3, None, 1] if split != 'test' else [1, None, None]
    elif task == 'sst-2':
        data_path = os.path.join(glue_dir, 'SST-2', file_name)
        text_a_id, text_b_id, label_id = [0, None, 1] if split != 'test' else [1, None, None]
    elif task == 'sts-b':
        data_path = os.path.join(glue_dir, 'STS-B', file_name)
        text_a_id, text_b_id, label_id = [7, 8, -1] if split != 'test' else [7, 8, None]
    elif task == 'qqp':
        data_path = os.path.join(glue_dir, 'QQP', file_name)
        text_a_id, text_b_id, label_id = [3, 4, 5] if split != 'test' else [1, 2, None]
    elif task == 'qnli':
        data_path = os.path.join(glue_dir, 'QNLI', file_name)
        text_a_id, text_b_id, label_id = [1, 2, -1] if split != 'test' else [1, 2, None]
    elif task == 'rte':
        data_path = os.path.join(glue_dir, 'RTE', file_name)
        text_a_id, text_b_id, label_id = [1, 2, -1] if split != 'test' else [1, 2, None]
    elif task == 'wnli':
        data_path = os.path.join(glue_dir, 'WNLI', file_name)
        text_a_id, text_b_id, label_id = [1, 2, -1] if split != 'test' else [1, 2, None]
    else:
        raise KeyError('task \'{}\' is not valid'.format(task))

    labels = datasets.glue_labels[task]
    label_map = {label: i for i, label in enumerate(labels)}
    data = read_tsv(data_path)

    examples = []
    for i, line in enumerate(data):
        if i == 0 and (split == 'test' or (split != 'test' and task != 'cola')):
            continue
        text_a = line[text_a_id]
        text_b = line[text_b_id] if text_b_id is not None else None
        if split == 'test':
            label = None
        else:
            label = line[label_id]
            label = float(label) if task == 'sts-b' else label_map[label]

        id = int(line[0]) if split == 'test' else None
        task_id = datasets.glue_train_tasks_to_ids[task] if task in datasets.glue_train_tasks else None
        examples.append(GlueExample(text_a, text_b, label, id, task_id))
    return examples


def create_glue_dataset(model_name, task, glue_dir, tokenizer, max_seq_len, split, local_rank, cache_dir=''):
    if model_name in models.bert_models:
        model = 'bert'
    elif model_name in models.xlnet_models:
        model = 'xlnet'
    elif model_name in models.roberta_models:
        model = 'roberta'
    else:
        raise KeyError('model name \'{}\' is not valid'.format(model_name))

    cache_file = os.path.join(cache_dir, 'glue', '_'.join([model, task, split, str(max_seq_len)]))
    if tokenizer.lowercase:
        cache_file = os.path.join(cache_dir, 'glue', '_'.join([model, task, split, str(max_seq_len), 'lowercase']))

    if os.path.exists(cache_file):
        if local_rank == 0:
            logging.info('Loading {} cache file from \'{}\''.format(split, cache_file))
        cache_data = torch.load(cache_file)
        examples = cache_data['examples']
        encoded_inputs = cache_data['encoded_inputs']
    else:
        if local_rank == 0:
            logging.info('Creating {} examples'.format(split))
        examples = create_glue_examples(task, glue_dir, split)
        texts_a = [example.text_a for example in examples]
        texts_b = [example.text_b for example in examples]

        if local_rank == 0:
            logging.info('Creating {} dataset'.format(split))
        encoded_inputs = [tokenizer.encode(text_a, text_b)
                          for text_a, text_b in tqdm(zip(texts_a, texts_b), total=len(texts_a), disable=local_rank != 0)]

        if cache_dir and local_rank == 0:
            logging.info('Saving {} cache file to \'{}\''.format(split, cache_file))
            cache_dir = os.path.join(cache_dir, 'glue')
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            torch.save({'examples': examples, 'encoded_inputs': encoded_inputs}, cache_file)

    token_ids = torch.tensor([inp.token_ids for inp in encoded_inputs], dtype=torch.long)
    segment_ids = torch.tensor([inp.segment_ids for inp in encoded_inputs], dtype=torch.long)
    position_ids = torch.tensor([inp.position_ids for inp in encoded_inputs], dtype=torch.long)
    attn_mask = torch.tensor([inp.attn_mask for inp in encoded_inputs], dtype=torch.long)

    if split == 'test':
        ids = torch.tensor([example.id for example in examples], dtype=torch.long)
        dataset = TensorDataset(token_ids, segment_ids, position_ids, attn_mask, ids)
    else:
        labels = torch.tensor([example.label for example in examples], dtype=torch.float if task == 'sts-b' else torch.long)
        dataset = TensorDataset(token_ids, segment_ids, position_ids, attn_mask, labels)
    return examples, encoded_inputs, dataset


def create_ensemble_glue_examples(glue_dir, all_tasks, train_ratio, val_ratio, split):
    if isinstance(all_tasks, str):
        all_tasks = [all_tasks]

    all_examples = []
    for task in all_tasks:
        if split == 'dev':
            cur_examples = create_glue_examples(task, glue_dir, 'dev')
        else:
            cur_examples = create_glue_examples(task, glue_dir, 'train')
            num_train = int(len(cur_examples) * train_ratio)
            num_val = int(len(cur_examples) * val_ratio)
            if split == 'train':
                assert num_train > 0
                cur_examples = cur_examples[:num_train]
            else:  # split == 'val'
                assert 0 < num_val <= int(len(cur_examples) * (1 - train_ratio)) + 1
                cur_examples = cur_examples[-num_val:]
        all_examples.append(cur_examples)

    return all_examples


def create_ensemble_glue_dataset(model_name, all_tasks, glue_dir, tokenizer, max_seq_len, train_ratio, val_ratio, split, local_rank, cache_dir=''):
    if model_name in models.bert_models:
        model = 'bert'
    elif model_name in models.xlnet_models:
        model = 'xlnet'
    elif model_name in models.roberta_models:
        model = 'roberta'
    else:
        raise KeyError('model name \'{}\' is not valid'.format(model_name))

    if split == 'train':
        file_name = '_'.join([model, split, str(max_seq_len), str(train_ratio)])
    elif split == 'val':
        file_name = '_'.join([model, split, str(max_seq_len), str(val_ratio)])
    elif split == 'dev':
        file_name = '_'.join([model, split, str(max_seq_len)])
    else:
        raise KeyError('split \'{}\' is not valid'.format(split))

    if tokenizer.lowercase:
        file_name += '_lowercase'
    ensemble_task_str = '_'.join(all_tasks)
    file_name += '_' + ensemble_task_str
    cache_file = os.path.join(cache_dir, 'ensemble_glue', file_name)

    if os.path.exists(cache_file):
        if local_rank == 0:
            logging.info('Loading {} cache file from \'{}\''.format(split, cache_file))
        cache_data = torch.load(cache_file)
        examples = cache_data['examples']
        encoded_inputs = cache_data['encoded_inputs']
    else:
        if local_rank == 0:
            logging.info('Creating {} examples'.format(split))
        examples = create_ensemble_glue_examples(glue_dir, all_tasks, train_ratio, val_ratio, split)

        if local_rank == 0:
            logging.info('Creating {} dataset'.format(split))
        encoded_inputs = []
        for cur_examples in tqdm(examples, disable=local_rank != 0):
            texts_a = [example.text_a for example in cur_examples]
            texts_b = [example.text_b for example in cur_examples]

            cur_encoded_inputs = [tokenizer.encode(text_a, text_b) for text_a, text_b in
                                  tqdm(zip(texts_a, texts_b), total=len(cur_examples), disable=local_rank != 0)]
            encoded_inputs.append(cur_encoded_inputs)

        if cache_dir and local_rank == 0:
            logging.info('Saving {} cache file to \'{}\''.format(split, cache_file))
            cache_dir = os.path.join(cache_dir, 'ensemble_glue')
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            torch.save({'examples': examples, 'encoded_inputs': encoded_inputs}, cache_file)

    token_ids = [torch.tensor([inp.token_ids for inp in cur_encoded_inputs], dtype=torch.long) for cur_encoded_inputs in encoded_inputs]
    segment_ids = [torch.tensor([inp.segment_ids for inp in cur_encoded_inputs], dtype=torch.long) for cur_encoded_inputs in encoded_inputs]
    position_ids = [torch.tensor([inp.position_ids for inp in cur_encoded_inputs], dtype=torch.long) for cur_encoded_inputs in encoded_inputs]
    attn_mask = [torch.tensor([inp.attn_mask for inp in cur_encoded_inputs], dtype=torch.long) for cur_encoded_inputs in encoded_inputs]
    task_ids = [torch.tensor([example.task_id for example in cur_examples], dtype=torch.long) for cur_examples in examples]
    labels = [torch.tensor([example.label for example in cur_examples], dtype=torch.float) for cur_examples in examples]

    all_datasets = [
        TensorDataset(cur_task_ids, cur_token_ids, cur_segment_ids, cur_position_ids, cur_attn_mask, cur_labels)
        for cur_task_ids, cur_token_ids, cur_segment_ids, cur_position_ids, cur_attn_mask, cur_labels in
        zip(task_ids, token_ids, segment_ids, position_ids, attn_mask, labels)]
    return examples, encoded_inputs, all_datasets


def create_split_glue_dataset(model_name, task, glue_dir, tokenizer, max_seq_len, train_ratio, val_ratio, split,
                              local_rank, cache_dir=''):
    if model_name in models.bert_models:
        model = 'bert'
    elif model_name in models.xlnet_models:
        model = 'xlnet'
    elif model_name in models.roberta_models:
        model = 'roberta'
    else:
        raise KeyError('model name \'{}\' is not valid'.format(model_name))

    if split == 'train':
        file_name = '_'.join([model, task, split, str(max_seq_len), str(train_ratio)])
    elif split == 'val':
        file_name = '_'.join([model, task, split, str(max_seq_len), str(val_ratio)])
    else:
        raise KeyError('split \'{}\' is not valid'.format(split))

    if tokenizer.lowercase:
        file_name += '_lowercase'
    cache_file = os.path.join(cache_dir, 'split_glue', file_name)

    if os.path.exists(cache_file):
        if local_rank == 0:
            logging.info('Loading {} cache file from \'{}\''.format(split, cache_file))
        cache_data = torch.load(cache_file)
        examples = cache_data['examples']
        encoded_inputs = cache_data['encoded_inputs']
    else:
        if local_rank == 0:
            logging.info('Creating {} examples'.format(split))
        examples = create_ensemble_glue_examples(glue_dir, task, train_ratio, val_ratio, split)[0]
        texts_a = [example.text_a for example in examples]
        texts_b = [example.text_b for example in examples]

        if local_rank == 0:
            logging.info('Creating {} dataset'.format(split))
        encoded_inputs = [tokenizer.encode(text_a, text_b)
                          for text_a, text_b in tqdm(zip(texts_a, texts_b), total=len(texts_a), disable=local_rank != 0)]

        if cache_dir and local_rank == 0:
            logging.info('Saving {} cache file to \'{}\''.format(split, cache_file))
            cache_dir = os.path.join(cache_dir, 'split_glue')
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            torch.save({'examples': examples, 'encoded_inputs': encoded_inputs}, cache_file)

    token_ids = torch.tensor([inp.token_ids for inp in encoded_inputs], dtype=torch.long)
    segment_ids = torch.tensor([inp.segment_ids for inp in encoded_inputs], dtype=torch.long)
    position_ids = torch.tensor([inp.position_ids for inp in encoded_inputs], dtype=torch.long)
    attn_mask = torch.tensor([inp.attn_mask for inp in encoded_inputs], dtype=torch.long)
    labels = torch.tensor([example.label for example in examples], dtype=torch.float if task == 'sts-b' else torch.long)
    dataset = TensorDataset(token_ids, segment_ids, position_ids, attn_mask, labels)
    return examples, encoded_inputs, dataset
