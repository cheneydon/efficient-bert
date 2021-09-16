import os
import numpy as np
import torch
import json
import logging
import models
from tqdm import tqdm
from torch.utils.data.dataset import TensorDataset


class SquadExample(object):
    def __init__(self, qa_id, question, context, train_answer, val_answer, start_pos, end_pos, context_tokens,
                 is_impossible):
        self.qa_id = qa_id
        self.question = question
        self.context = context
        self.train_answer = train_answer
        self.val_answer = val_answer
        self.start_position = start_pos
        self.end_position = end_pos
        self.context_tokens = context_tokens
        self.is_impossible = is_impossible


class SquadResult(object):
    def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None, cls_logits=None):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits


def refine_subtoken_position(context_subtokens, subtoken_start_pos, subtoken_end_pos, tokenizer, annotated_answer):
    subtoken_answer = ' '.join(tokenizer.tokenize(annotated_answer))
    for new_st in range(subtoken_start_pos, subtoken_end_pos + 1):
        for new_ed in range(subtoken_end_pos, subtoken_start_pos - 1, -1):
            text_span = ' '.join(context_subtokens[new_st:(new_ed + 1)])
            if text_span == subtoken_answer:
                return new_st, new_ed
    return subtoken_start_pos, subtoken_end_pos


def get_char_to_word_positions(context, answer, start_char_pos, is_impossible):
    context_tokens = []
    char_to_word_offset = []
    is_prev_whitespace = True
    for c in context:
        is_whitespace = (c == ' ' or c == '\t' or c == '\r' or c == '\n' or ord(c) == 0x202F)
        if is_whitespace:
            is_prev_whitespace = True
        else:
            if is_prev_whitespace:
                context_tokens.append(c)
            else:
                context_tokens[-1] += c
            is_prev_whitespace = False
        char_to_word_offset.append(len(context_tokens) - 1)

    start_pos, end_pos = 0, 0
    if start_char_pos is not None and not is_impossible:
        start_pos = char_to_word_offset[start_char_pos]
        end_pos = char_to_word_offset[start_char_pos + len(answer) - 1]
    return start_pos, end_pos, context_tokens


def check_max_context_token(all_spans, cur_span_idx, pos):
    best_score, best_span_idx = None, None
    for span_idx, span in enumerate(all_spans):
        end = span.context_start_position + span.context_len - 1
        if pos < span.context_start_position or pos > end:
            continue
        num_left_context = pos - span.context_start_position
        num_right_context = end - pos
        score = min(num_left_context, num_right_context) + 0.01 * span.context_len
        if best_score is None or score > best_score:
            best_score = score
            best_span_idx = span_idx
    return cur_span_idx == best_span_idx


def create_squad_examples(task, squad_dir, split):
    if split == 'test':
        data_path = squad_dir
    else:
        if task == 'squad1.1':
            data_path = os.path.join(squad_dir, 'squad1.1', '{}-v1.1.json'.format(split))
        elif task == 'squad2.0':
            data_path = os.path.join(squad_dir, 'squad2.0', '{}-v2.0.json'.format(split))
        else:
            raise KeyError('task \'{}\' is not valid'.format(task))

    with open(data_path, 'r', encoding='utf-8') as reader:
        data = json.load(reader)['data']

    examples = []
    for i, entry in enumerate(data):
        for paragraph in entry['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                qa_id = qa['id']
                question = qa['question']
                start_char_pos = None
                train_answer = None
                val_answer = []

                if 'is_impossible' in qa:
                    is_impossible = qa['is_impossible']
                else:
                    is_impossible = False
                if not is_impossible:
                    if split == 'train':
                        train_answer = qa['answers'][0]['text']
                        start_char_pos = qa['answers'][0]['answer_start']
                    else:
                        val_answer = qa['answers']

                start_pos, end_pos, context_tokens = get_char_to_word_positions(
                    context, train_answer, start_char_pos, is_impossible)
                examples.append(SquadExample(qa_id, question, context, train_answer, val_answer, start_pos, end_pos,
                                             context_tokens, is_impossible))

    return examples


def create_squad_dataset(model_name, task, squad_dir, tokenizer, max_seq_len, max_query_len, trunc_stride, split,
                         local_rank, cache_dir=''):
    if model_name in models.bert_models:
        model = 'bert'
    elif model_name in models.xlnet_models:
        model = 'xlnet'
    elif model_name in models.roberta_models:
        model = 'roberta'
    else:
        raise KeyError('model name \'{}\' is not valid'.format(model_name))

    cache_file = os.path.join(
        cache_dir, 'squad', '_'.join([model, task, split, str(max_seq_len), str(max_query_len), str(trunc_stride)]))
    if tokenizer.lowercase:
        cache_file = os.path.join(
            cache_dir, 'squad', '_'.join([model, task, split, str(max_seq_len), str(max_query_len), str(trunc_stride), 'lowercase']))

    if os.path.exists(cache_file):
        if local_rank == 0:
            logging.info('Loading {} cache file from \'{}\''.format(split, cache_file))
        cache_data = torch.load(cache_file)
        examples = cache_data['examples']
        encoded_inputs = cache_data['encoded_inputs']
    else:
        if local_rank == 0:
            logging.info('Creating {} examples'.format(split))
        examples = create_squad_examples(task, squad_dir, split)

        # Defining helper methods
        unique_id = 1000000000
        encoded_inputs = []
        if local_rank == 0:
            logging.info('Creating {} dataset'.format(split))
        for example_idx, example in tqdm(enumerate(examples), total=len(examples), disable=local_rank != 0):
            if split == 'train' and not example.is_impossible:
                start_pos = example.start_position
                end_pos = example.end_position

                actual_answer = ' '.join(example.context_tokens[start_pos:(end_pos + 1)])
                cleaned_answer = ' '.join(example.train_answer.strip().split())
                if actual_answer.find(cleaned_answer) == -1:
                    if local_rank == 0:
                        logging.info('Could not find answer: {} vs. {}'.format(actual_answer, cleaned_answer))
                    continue

            tok_to_subtok_idx = []
            subtok_to_tok_idx = []
            context_subtokens = []
            for i, token in enumerate(example.context_tokens):
                tok_to_subtok_idx.append(len(context_subtokens))
                subtokens = tokenizer.tokenize(token)
                for subtoken in subtokens:
                    subtok_to_tok_idx.append(i)
                    context_subtokens.append(subtoken)

            if split == 'train' and not example.is_impossible:
                subtoken_start_pos = tok_to_subtok_idx[example.start_position]
                if example.end_position < len(example.context_tokens) - 1:
                    subtoken_end_pos = tok_to_subtok_idx[example.end_position + 1] - 1
                else:
                    subtoken_end_pos = len(context_subtokens) - 1
                subtoken_start_pos, subtoken_end_pos = refine_subtoken_position(
                    context_subtokens, subtoken_start_pos, subtoken_end_pos, tokenizer, example.train_answer)

            truncated_context = context_subtokens
            num_special_query = tokenizer.num_special_token_b_paired if model == 'xlnet' else tokenizer.num_special_token_a_paired
            len_question = min(len(tokenizer.tokenize(example.question)), max_query_len - num_special_query)
            added_trunc_size = max_seq_len - trunc_stride - len_question - tokenizer.num_special_token_paired

            spans = []
            while len(spans) * trunc_stride < len(context_subtokens):
                text_a = truncated_context if model == 'xlnet' else example.question
                text_b = example.question if model == 'xlnet' else truncated_context
                encoded_input = tokenizer.encode(text_a, text_b, added_trunc_size)

                context_start_pos = len(spans) * trunc_stride
                context_len = min(len(context_subtokens) - context_start_pos,
                                  max_seq_len - len_question - tokenizer.num_special_token_paired)
                context_end_pos = context_start_pos + context_len - 1

                if tokenizer.pad_token_id in encoded_input.token_ids:
                    non_padded_ids = encoded_input.token_ids[:encoded_input.token_ids.index(tokenizer.pad_token_id)]
                else:
                    non_padded_ids = encoded_input.token_ids
                tokens = tokenizer._ids_to_tokens(non_padded_ids)

                context_subtok_to_tok_idx = {}
                for i in range(context_len):
                    context_idx = i if model == 'xlnet' else len_question + tokenizer.num_special_token_a_paired + i
                    context_subtok_to_tok_idx[context_idx] = subtok_to_tok_idx[context_start_pos + i]

                # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
                cls_index = encoded_input.token_ids.index(tokenizer.cls_token_id)
                p_mask = np.array(encoded_input.segment_ids)
                p_mask = np.minimum(p_mask, 1)
                p_mask = 1 - p_mask if model != 'xlnet' else p_mask
                p_mask[np.where(np.array(encoded_input.token_ids) == tokenizer.sep_token_id)[0]] = 1
                p_mask[cls_index] = 0

                start_pos, end_pos = 0, 0
                span_is_impossible = example.is_impossible
                if split == 'train' and not span_is_impossible:
                    # For training, if our document chunk does not contain an annotation
                    # we throw it out, since there is nothing to predict.
                    if subtoken_start_pos >= context_start_pos and subtoken_end_pos <= context_end_pos:
                        context_offset = 0 if model == 'xlnet' else len_question + tokenizer.num_special_token_a_paired
                        start_pos = subtoken_start_pos - context_start_pos + context_offset
                        end_pos = subtoken_end_pos - context_start_pos + context_offset
                    else:
                        start_pos = cls_index
                        end_pos = cls_index
                        span_is_impossible = True

                encoded_input.start_position = start_pos
                encoded_input.end_position = end_pos
                encoded_input.cls_index = cls_index
                encoded_input.p_mask = p_mask.tolist()
                encoded_input.is_impossible = span_is_impossible

                # For computing metrics
                encoded_input.example_index = example_idx
                encoded_input.context_start_position = context_start_pos
                encoded_input.context_len = context_len
                encoded_input.tokens = tokens
                encoded_input.context_subtok_to_tok_idx = context_subtok_to_tok_idx
                encoded_input.is_max_context_token = {}
                encoded_input.unique_id = unique_id
                spans.append(encoded_input)
                unique_id += 1

                if encoded_input.overflow_token_ids is None:
                    break
                truncated_context = encoded_input.overflow_token_ids

            for span_idx in range(len(spans)):
                for context_idx in range(spans[span_idx].context_len):
                    is_max_context_token = check_max_context_token(spans, span_idx, span_idx * trunc_stride + context_idx)
                    idx = context_idx if model == 'xlnet' else \
                        len_question + tokenizer.num_special_token_a_paired + context_idx
                    spans[span_idx].is_max_context_token[idx] = is_max_context_token
            encoded_inputs.extend(spans)

        if cache_dir and local_rank == 0:
            logging.info('Saving {} cache file to \'{}\''.format(split, cache_file))
            cache_dir = os.path.join(cache_dir, 'squad')
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            torch.save({'examples': examples,
                        'encoded_inputs': encoded_inputs}, cache_file)

    token_ids = torch.tensor([inp.token_ids for inp in encoded_inputs], dtype=torch.long)
    segment_ids = torch.tensor([inp.segment_ids for inp in encoded_inputs], dtype=torch.long)
    position_ids = torch.tensor([inp.position_ids for inp in encoded_inputs], dtype=torch.long)
    all_attn_mask = torch.tensor([inp.attn_mask for inp in encoded_inputs], dtype=torch.long)
    start_positions = torch.tensor([inp.start_position for inp in encoded_inputs], dtype=torch.long)
    end_positions = torch.tensor([inp.end_position for inp in encoded_inputs], dtype=torch.long)
    all_cls_index = torch.tensor([inp.cls_index for inp in encoded_inputs], dtype=torch.long)
    all_p_mask = torch.tensor([inp.p_mask for inp in encoded_inputs], dtype=torch.float)
    all_is_impossible = torch.tensor([inp.is_impossible for inp in encoded_inputs], dtype=torch.float)

    if split == 'train':
        dataset = TensorDataset(token_ids, segment_ids, position_ids, all_attn_mask, start_positions, end_positions,
                                all_cls_index, all_p_mask, all_is_impossible)
    else:
        all_example_index = torch.arange(token_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(token_ids, segment_ids, position_ids, all_attn_mask, all_example_index,
                                all_cls_index, all_p_mask)

    return examples, encoded_inputs, dataset
