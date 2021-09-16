
class BertBaseConfig(object):
    def __init__(self, lowercase=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.position_size = 512
        self.segment_size = 2
        self.hidden_size = 768
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 12
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = 3072
        self.num_layers = 12
        self.pad_token_id = 0
        self.sep_token_id = 102


class BertLargeConfig(object):
    def __init__(self, lowercase=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.position_size = 512
        self.segment_size = 2
        self.hidden_size = 1024
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 16
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = 4096
        self.num_layers = 24
        self.pad_token_id = 0
        self.sep_token_id = 102


class XlnetBaseConfig(object):
    def __init__(self):
        self.vocab_size = 32000
        self.position_size = 512
        self.segment_size = 2
        self.hidden_size = 768
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 12
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = 3072
        self.num_layers = 12
        self.pad_token_id = 5
        self.start_n_top = 5
        self.end_n_top = 5


class XlnetLargeConfig(object):
    def __init__(self):
        self.vocab_size = 32000
        self.position_size = 512
        self.segment_size = 2
        self.hidden_size = 1024
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 16
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = 4096
        self.num_layers = 24
        self.pad_token_id = 5
        self.start_n_top = 5
        self.end_n_top = 5


class RobertaBaseConfig(object):
    def __init__(self):
        self.vocab_size = 50265
        self.position_size = 514
        self.hidden_size = 768
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 12
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = 3072
        self.num_layers = 12
        self.pad_token_id = 1


class RobertaLargeConfig(object):
    def __init__(self):
        self.vocab_size = 50265
        self.position_size = 514
        self.hidden_size = 1024
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 16
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = 4096
        self.num_layers = 24
        self.pad_token_id = 1


class Gpt2SmallConfig(object):
    def __init__(self):
        self.vocab_size = 50257
        self.position_size = 1024
        self.hidden_size = 768
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 12
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = 3072
        self.num_layers = 12


class Gpt2mediumConfig(object):
    def __init__(self):
        self.vocab_size = 50257
        self.position_size = 1024
        self.hidden_size = 1024
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 16
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = 3072
        self.num_layers = 24


class MobileBertConfig(object):
    def __init__(self, lowercase=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.position_size = 512
        self.segment_size = 2
        self.embed_size = 128
        self.hidden_size = 512
        self.inner_hidden_size = 128
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 4
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = 512
        self.num_layers = 24
        self.pad_token_id = 0
        self.sep_token_id = 102
        self.is_tiny = True
        self.use_opt = True


class TinyBertConfig(object):
    def __init__(self, lowercase=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.position_size = 512
        self.segment_size = 2
        self.hidden_size = 312  # 768
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 12
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = 1200  # 3072
        self.num_layers = 4  # 6
        self.pad_token_id = 0
        self.sep_token_id = 102
        self.use_fit_dense = True
        self.fit_size = 768


class AutoBertConfig(object):
    def __init__(self, lowercase=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.position_size = 512
        self.segment_size = 2
        self.hidden_size = 540
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 12
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = self.hidden_size
        self.num_layers = 6
        self.pad_token_id = 0
        self.sep_token_id = 102
        self.use_fit_dense = True
        self.use_opt = False
        self.embed_size = 128
        self.fit_size = 768
        self.max_stacked_ffn = 4
        self.expansion_ratio_map = {
            '1_1': 1, '1_2': 1 / 2, '1_3': 1 / 3, '1_4': 1 / 4,
            '2_1': 1, '2_2': 1 / 2, '2_3': 1 / 3, '2_4': 1 / 4,
            '3_1': 1, '3_2': 1 / 2, '3_3': 1 / 3, '3_4': 1 / 4,
            '4_1': 1, '4_2': 1 / 2, '4_3': 1 / 3, '4_4': 1 / 4,
        }
        self.param_list = {
            'embed_fit_dense': 4.8088, 'attn': 1.1696,
            '1_1': 0.5854, '1_2': 0.2935, '1_3': 0.1962, '1_4': 0.1476,
            '2_1': 1.1707, '2_2': 0.5870, '2_3': 0.3924, '2_4': 0.2951,
            '3_1': 1.7561, '3_2': 0.8805, '3_3': 0.5886, '3_4': 0.4427,
            '4_1': 2.3414, '4_2': 1.1740, '4_3': 0.7848, '4_4': 0.5902,
        }


class AutoBertSmallConfig(object):
    def __init__(self, lowercase=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.position_size = 512
        self.segment_size = 2
        self.hidden_size = 360
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 12
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = self.hidden_size
        self.num_layers = 6
        self.pad_token_id = 0
        self.sep_token_id = 102
        self.use_fit_dense = True
        self.use_opt = False
        self.embed_size = 128
        self.fit_size = 768
        self.max_stacked_ffn = 4
        self.expansion_ratio_map = {
            '1_1': 1, '1_2': 1 / 2, '1_3': 1 / 3, '1_4': 1 / 4,
            '2_1': 1, '2_2': 1 / 2, '2_3': 1 / 3, '2_4': 1 / 4,
            '3_1': 1, '3_2': 1 / 2, '3_3': 1 / 3, '3_4': 1 / 4,
            '4_1': 1, '4_2': 1 / 2, '4_3': 1 / 3, '4_4': 1 / 4,
        }
        self.param_list = {
            'embed_fit_dense': 4.5084, 'attn': 0.5206,
            '1_1': 0.2606, '1_2': 0.1309, '1_3': 0.0876, '1_4': 0.0660,
            '2_1': 0.5213, '2_2': 0.2617, '2_3': 0.1752, '2_4': 0.1319,
            '3_1': 0.7819, '3_2': 0.3926, '3_3': 0.2628, '3_4': 0.1979,
            '4_1': 1.0426, '4_2': 0.5234, '4_3': 0.3504, '4_4': 0.2639
        }


class AutoBert12Config(object):
    def __init__(self, lowercase=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.position_size = 512
        self.segment_size = 2
        self.hidden_size = 360 #540
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 12
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = self.hidden_size
        self.num_layers = 12
        self.pad_token_id = 0
        self.sep_token_id = 102
        self.use_fit_dense = True
        self.use_opt = False
        self.embed_size = 128
        self.fit_size = 768
        self.max_stacked_ffn = 4
        self.expansion_ratio_map = {
            '1_1': 1, '1_2': 1 / 2, '1_3': 1 / 3, '1_4': 1 / 4,
            '2_1': 1, '2_2': 1 / 2, '2_3': 1 / 3, '2_4': 1 / 4,
            '3_1': 1, '3_2': 1 / 2, '3_3': 1 / 3, '3_4': 1 / 4,
            '4_1': 1, '4_2': 1 / 2, '4_3': 1 / 3, '4_4': 1 / 4,
        }
        self.param_list = {
            'embed_fit_dense': 4.5084, 'attn': 0.5206,
            '1_1': 0.3904, '1_2': 0.1958, '1_3': 0.1309, '1_4': 0.0984,
            '2_1': 0.7808, '2_2': 0.3915, '2_3': 0.2617, '2_4': 0.1968,
            '3_1': 1.1713, '3_2': 0.5873, '3_3': 0.3926, '3_4': 0.2952,
            '4_1': 1.5617, '4_2': 0.7830, '4_3': 0.5234, '4_4': 0.3937}


class AutoTinyBertConfig(object):
    def __init__(self, lowercase=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.position_size = 512
        self.segment_size = 2
        self.hidden_size = 768
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 12
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = 3072
        self.num_layers = 6
        self.pad_token_id = 0
        self.sep_token_id = 102
        self.use_fit_dense = True
        self.fit_size = 768
        self.max_stacked_ffn = 4
        self.expansion_ratio_map = {
            '1_1': 1, '1_2': 1 / 2, '1_3': 1 / 3, '1_4': 1 / 4,
            '2_1': 1, '2_2': 1 / 2, '2_3': 1 / 3, '2_4': 1 / 4,
            '3_1': 1, '3_2': 1 / 2, '3_3': 1 / 3, '3_4': 1 / 4,
            '4_1': 1, '4_2': 1 / 2, '4_3': 1 / 3, '4_4': 1 / 4,
        }
        self.param_list = {
            'embed_fit_dense': 24.4278, 'attn': 2.3639,
            '1_1': 4.7240, '1_2': 2.3631, '1_3': 1.5762, '1_4': 1.1827,
            '2_1': 9.4479, '2_2': 4.7263, '2_3': 3.1524, '2_4': 2.3654,
            '3_1': 14.1719, '3_2': 7.0894, '3_3': 4.7286, '3_4': 3.5482,
            '4_1': 18.8959, '4_2': 9.4525, '4_3': 6.3048, '4_4': 4.7309,
        }


class SupernetConfig(object):
    def __init__(self, lowercase=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.position_size = 512
        self.segment_size = 2
        self.hidden_size = 540
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 12
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = self.hidden_size
        self.num_layers = 6
        self.pad_token_id = 0
        self.sep_token_id = 102
        self.use_mobile_embed = True
        self.use_fit_dense = True
        self.use_opt = False
        self.embed_size = 128
        self.fit_size = 768
        self.num_stacked_ffn = 4
        self.ffn_expansion_ratio = 1
