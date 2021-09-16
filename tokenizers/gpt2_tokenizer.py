import numpy as np
import tokenizers


class Gpt2BasicTokenizer(tokenizers.RobertaBasicTokenizer):
    def __init__(self, lowercase, vocab_path, merge_path, unk_token='<|unk|>', sep_token='<|sep|>',
                 pad_token='<|pad|>', bos_token='<|start|>', eos_token='<|end|>'):
        super(Gpt2BasicTokenizer, self).__init__(lowercase, vocab_path, merge_path)

        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.special_tokens = {self.unk_token, self.sep_token, self.pad_token, self.bos_token, self.eos_token}

        self.unk_token_id = self._get_token_id(self.unk_token)
        self.sep_token_id = self._get_token_id(self.sep_token)
        self.pad_token_id = self._get_token_id(self.pad_token)
        self.bos_token_id = self._get_token_id(self.bos_token)
        self.eos_token_id = self._get_token_id(self.eos_token)

        self.id_to_token_map = {v: k for k, v in self.vocab_map.items()}

    def _get_token_id(self, token):
        if self.vocab_map.get(token) is not None:
            return self.vocab_map[token]
        else:
            self.vocab_map[token] = len(self.vocab_map)
            self.num_add_tokens += 1
            return self.vocab_map[token]


class Gpt2Tokenizer(Gpt2BasicTokenizer):
    def __init__(self, lowercase, task, vocab_path, merge_path, max_seq_len, max_abstract_len=None):
        super(Gpt2Tokenizer, self).__init__(lowercase, vocab_path, merge_path)

        self.task = task
        self.max_seq_len = max_seq_len
        self.max_query_len = max_abstract_len

        # single_tokens: <|start|> X <|end|>, paired_tokens: <|start|> A <|sep|> B <|end|>
        self.num_special_token_single = 2
        self.num_special_token_paired = 3
        self.num_special_token_a_paired = 2
        self.num_special_token_b_paired = 1

    def _combine_and_pad(self, token_ids_a, token_ids_b=None):
        token_ids = [self.pad_token_id] * self.max_seq_len
        segment_ids = None
        attn_mask = [1] * self.max_seq_len
        token_ids = np.array(token_ids)
        attn_mask = np.array(attn_mask)

        token_ids[0] = self.bos_token_id
        token_ids[1:len(token_ids_a) + 1] = token_ids_a

        if token_ids_b is None:
            token_ids[len(token_ids_a) + 1] = self.eos_token_id
            attn_mask[:len(token_ids_a) + 2] = 0
        else:
            token_ids[-self.max_query_len - 1] = self.sep_token_id
            token_ids[-self.max_query_len:-self.max_query_len + len(token_ids_b)] = token_ids_b
            token_ids[-self.max_query_len + len(token_ids_b)] = self.eos_token_id
            attn_mask[:len(token_ids_a) + 1] = 0
            attn_mask[-self.max_query_len - 1:-self.max_query_len + len(token_ids_b) + 1] = 0

        token_ids = token_ids.tolist()
        attn_mask = attn_mask.tolist()
        position_ids = [i for i in range(self.max_seq_len)]
        return token_ids, segment_ids, position_ids, attn_mask
