import models
from .base_tokenizer import BaseTokenizer, EncodedInput
from .bert_tokenizer import BertBasicTokenizer, BertTokenizer
from .xlnet_tokenizer import XlnetBasicTokenizer, XlnetTokenizer
from .roberta_tokenizer import RobertaBasicTokenizer, RobertaTokenizer
from .gpt2_tokenizer import Gpt2BasicTokenizer, Gpt2Tokenizer


def select_basic_tokenizer(model_name, lowercase, vocab_path, merge_path=None):
    if model_name in models.bert_models:
        return BertBasicTokenizer(lowercase, vocab_path)
    elif model_name in models.xlnet_models:
        return XlnetBasicTokenizer(lowercase, vocab_path)
    elif model_name in models.roberta_models:
        return RobertaBasicTokenizer(lowercase, vocab_path, merge_path)
    elif model_name in models.gpt_models:
        return Gpt2BasicTokenizer(lowercase, vocab_path, merge_path)
    else:
        raise KeyError('Basic tokenizer of \'{}\' is not found'.format(model_name))


def select_tokenizer(model_name, lowercase, task, vocab_path, max_seq_len, max_query_len=None, merge_path=None):
    if model_name in models.bert_models:
        return BertTokenizer(lowercase, task, vocab_path, max_seq_len, max_query_len)
    elif model_name in models.xlnet_models:
        return XlnetTokenizer(lowercase, task, vocab_path, max_seq_len, max_query_len)
    elif model_name in models.roberta_models:
        return RobertaTokenizer(lowercase, task, vocab_path, merge_path, max_seq_len, max_query_len)
    elif model_name in models.gpt_models:
        return Gpt2Tokenizer(lowercase, task, vocab_path, merge_path, max_seq_len, max_query_len)
    else:
        raise KeyError('Tokenizer of \'{}\' is not found'.format(model_name))
