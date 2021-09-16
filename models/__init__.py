from .bert import gelu, BertEmbedding, BertFeedForwardNetwork, BertAttention, \
    BertTransformerBlock, BertClsPooler, BertMaskedLMHead, BertSingle, Bert
from .xlnet import XlnetSingle, Xlnet
from .roberta import RobertaSingle, Roberta
from .gpt2 import Gpt2
from .mobile_bert import MobileBertEmbedding, MobileBertSingle, MobileBert
from .tiny_bert import TinyBertSingle, TinyBert
from .auto_bert import AutoBertSingle, AutoBert, MultiTaskBert, MultiTaskAutoBert, AutoTinyBertSingle, AutoTinyBert
from .supernet import SupernetSingle, Supernet
from .configs import BertBaseConfig, BertLargeConfig, XlnetBaseConfig, XlnetLargeConfig, \
    RobertaBaseConfig, RobertaLargeConfig, Gpt2SmallConfig, Gpt2mediumConfig, MobileBertConfig, TinyBertConfig, \
    AutoBertConfig, AutoBertSmallConfig, AutoBert12Config, AutoTinyBertConfig, SupernetConfig

nas_bert_models = ['auto_bert', 'auto_bert_small', 'auto_bert_12', 'mt_auto_bert', 'auto_tiny_bert']
bert_models = ['bert_base', 'bert_large', 'mobile_bert', 'tiny_bert', 'mt_bert_base', 'supernet'] + nas_bert_models
xlnet_models = ['xlnet_base', 'xlnet_large']
roberta_models = ['roberta_base', 'roberta_large']
gpt_models = ['gpt2_small', 'gpt2_medium']
all_models = bert_models + xlnet_models + roberta_models + gpt_models


def select_config(model_name, lowercase=None):
    if model_name in ['bert_base', 'mt_bert_base']:
        return BertBaseConfig(lowercase)
    elif model_name == 'bert_large':
        return BertLargeConfig(lowercase)
    elif model_name == 'xlnet_base':
        return XlnetBaseConfig()
    elif model_name == 'xlnet_large':
        return XlnetLargeConfig()
    elif model_name == 'roberta_base':
        return RobertaBaseConfig()
    elif model_name == 'roberta_large':
        return RobertaLargeConfig()
    elif model_name == 'gpt2_small':
        return Gpt2SmallConfig()
    elif model_name == 'gpt2_medium':
        return Gpt2mediumConfig()
    elif model_name == 'mobile_bert':
        return MobileBertConfig(lowercase)
    elif model_name == 'tiny_bert':
        return TinyBertConfig(lowercase)
    elif model_name in ['auto_bert', 'mt_auto_bert']:
        return AutoBertConfig(lowercase)
    elif model_name == 'auto_bert_small':
        return AutoBertSmallConfig(lowercase)
    elif model_name == 'auto_bert_12':
        return AutoBert12Config(lowercase)
    elif model_name == 'auto_tiny_bert':
        return AutoTinyBertConfig(lowercase)
    elif model_name == 'supernet':
        return SupernetConfig(lowercase)
    else:
        raise KeyError('Config for model \'{}\' is not found'.format(model_name))


def select_single_model(model_name, lowercase, use_lm=False):
    config = select_config(model_name, lowercase)
    if model_name in bert_models:
        if model_name in ['auto_bert', 'auto_bert_small', 'auto_bert_12']:
            return AutoBertSingle(config, use_lm)
        elif model_name == 'mobile_bert':
            return MobileBertSingle(config, use_lm)
        elif model_name == 'tiny_bert':
            return TinyBertSingle(config, use_lm)
        elif model_name == 'auto_tiny_bert':
            return AutoTinyBertSingle(config, use_lm)
        elif model_name == 'supernet':
            return SupernetSingle(config, use_lm)
        else:
            return BertSingle(config, use_lm)
    elif model_name in xlnet_models:
        return XlnetSingle(config)
    elif model_name in roberta_models:
        return RobertaSingle(config)


def select_model(model_name, lowercase, task, num_add_tokens=0, return_hid=False):
    config = select_config(model_name, lowercase)
    if model_name in bert_models:
        if model_name in ['auto_bert', 'auto_bert_small', 'auto_bert_12']:
            return AutoBert(config, task, return_hid)
        elif model_name == 'mt_auto_bert':
            return MultiTaskAutoBert(config, task, return_hid)
        elif model_name == 'mt_bert_base':
            return MultiTaskBert(config, return_hid)
        elif model_name == 'mobile_bert':
            return MobileBert(config, task, return_hid)
        elif model_name == 'tiny_bert':
            return TinyBert(config, task, return_hid)
        elif model_name == 'auto_tiny_bert':
            return AutoTinyBert(config, task, return_hid)
        elif model_name == 'supernet':
            return Supernet(config, task, return_hid)
        else:
            return Bert(config, task, return_hid)
    elif model_name in xlnet_models:
        return Xlnet(config, task)
    elif model_name in roberta_models:
        return Roberta(config, task)
    elif model_name in gpt_models:
        return Gpt2(config, num_add_tokens, use_lm=True)
