import os
import csv
import json
import glob
import torch
import logging
import models
from tqdm import tqdm
from torch.utils.data.dataset import TensorDataset


class MultiChoiceExample(object):
    def __init__(self, question, contexts, endings, label, id=None):
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label
        self.id = id


def read_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        data = []
        for line in reader:
            data.append(line)
        return data


def read_txt(input_dir):
    data = []
    files = glob.glob(input_dir + '/*txt')
    for file in files:
        with open(file, 'r', encoding='utf-8') as fin:
            cur_data = json.load(fin)
            cur_data['race_id'] = file
            data.append(cur_data)
    return data


def create_multi_choice_examples(task, data_dir, split):
    if task == 'swag':
        data_path = os.path.join(data_dir, split + '.csv')
        data = read_csv(data_path)

        labels = ['0', '1', '2', '3']
        label_map = {label: i for i, label in enumerate(labels)}

        examples = []
        for line in data[1:]:
            question = line[5]
            contexts = [line[4], line[4], line[4], line[4]]
            endings = [line[7], line[8], line[9], line[10]]
            label = None if split == 'test' else label_map[line[11]]
            id = int(line[0]) if split == 'test' else None
            examples.append(MultiChoiceExample(question, contexts, endings, label, id))

    elif task == 'race':
        high_path = os.path.join(data_dir, '/'.join([split, 'high']))
        middle_path = os.path.join(data_dir, '/'.join([split, 'middle']))
        high_text = read_txt(high_path)
        middle_text = read_txt(middle_path)

        all_text = high_text + middle_text

        labels = ['0', '1', '2', '3']
        label_map = {label: i for i, label in enumerate(labels)}

        examples = []
        for cur_data in all_text:
            article = cur_data['article']
            for i in range(len(cur_data['answers'])):
                question = cur_data['questions'][i]
                options = cur_data['options'][i]
                contexts = [article, article, article, article]
                endings = [options[0], options[1], options[2], options[3]]
                label = label_map[str(ord(cur_data['answers'][i]) - ord('A'))]
                examples.append(MultiChoiceExample(question, contexts, endings, label))
    else:
        raise KeyError('task \'{}\' is not valid'.format(task))

    return examples


def create_multi_choice_dataset(model_name, task, data_dir, tokenizer, max_seq_len, max_query_len, split, local_rank,
                                cache_dir=''):
    if model_name in models.bert_models:
        model = 'bert'
    elif model_name in models.xlnet_models:
        model = 'xlnet'
    elif model_name in models.roberta_models:
        model = 'roberta'
    else:
        raise KeyError('model name \'{}\' is not valid'.format(model_name))

    cache_file = os.path.join(cache_dir, 'multi_choice', '_'.join([model, task, split, str(max_seq_len), str(max_query_len)]))
    if tokenizer.lowercase:
        cache_file = os.path.join(
            cache_dir, 'multi_choice', '_'.join([model, task, split, str(max_seq_len), str(max_query_len), 'lowercase']))

    if os.path.exists(cache_file):
        if local_rank == 0:
            logging.info('Loading {} cache file from \'{}\''.format(split, cache_file))
        cache_data = torch.load(cache_file)
        examples = cache_data['examples']
        encoded_inputs = cache_data['encoded_inputs']
    else:
        if local_rank == 0:
            logging.info('Creating {} examples'.format(split))
        examples = create_multi_choice_examples(task, data_dir, split)

        encoded_inputs = []
        if local_rank == 0:
            logging.info('Creating {} dataset'.format(split))

        for example in tqdm(examples, disable=local_rank != 0):
            choice_encoded_inputs = []
            for context, ending in zip(example.contexts, example.endings):
                text_a = context
                if example.question.find('_') != -1:
                    # this is for cloze question
                    text_b = example.question.replace('_', ending)
                else:
                    text_b = example.question + ' ' + ending
                encoded_input = tokenizer.encode(text_a, text_b)
                choice_encoded_inputs.append(encoded_input)
            encoded_inputs.append(choice_encoded_inputs)

        if cache_dir and local_rank == 0:
            logging.info('Saving {} cache file to \'{}\''.format(split, cache_file))
            cache_dir = os.path.join(cache_dir, 'multi_choice')
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            torch.save({'examples': examples,
                        'encoded_inputs': encoded_inputs}, cache_file)

    token_ids = torch.tensor(
        [[choice_inp.token_ids for choice_inp in encoded_input] for encoded_input in encoded_inputs], dtype=torch.long)
    segment_ids = torch.tensor(
        [[choice_inp.segment_ids for choice_inp in encoded_input] for encoded_input in encoded_inputs], dtype=torch.long)
    position_ids = torch.tensor(
        [[choice_inp.position_ids for choice_inp in encoded_input] for encoded_input in encoded_inputs], dtype=torch.long)
    attn_mask = torch.tensor(
        [[choice_inp.attn_mask for choice_inp in encoded_input] for encoded_input in encoded_inputs], dtype=torch.long)

    if split == 'test' and task != 'race':
        ids = torch.tensor([example.id for example in examples], dtype=torch.long)
        dataset = TensorDataset(token_ids, segment_ids, position_ids, attn_mask, ids)
    else:
        labels = torch.tensor([example.label for example in examples], dtype=torch.long)
        dataset = TensorDataset(token_ids, segment_ids, position_ids, attn_mask, labels)

    return examples, encoded_inputs, dataset
