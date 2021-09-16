import torch
import regex as re
import json
import tokenizers
from functools import lru_cache


@lru_cache()
def bytes_to_unicodes():
    """
    Returns list of utf-8 byte and a mapping to unicode strings.
    We specifically avoids mapping to whitespace/control characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    """
    bytes = list(range(ord('!'), ord('~') + 1))+list(range(ord('¡'), ord('¬') + 1))+list(range(ord('®'), ord('ÿ') + 1))
    unicodes = bytes[:]
    n = 0
    for b in range(2**8):
        if b not in bytes:
            bytes.append(b)
            unicodes.append(2**8 + n)
            n += 1
    unicodes = [chr(n) for n in unicodes]
    return dict(zip(bytes, unicodes))


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class RobertaBasicTokenizer(tokenizers.BaseTokenizer):
    def __init__(self, lowercase, vocab_path, merge_path, unk_token='<unk>', sep_token='</s>', pad_token='<pad>',
                 cls_token='<s>', mask_token='<mask>'):
        super(RobertaBasicTokenizer, self).__init__(lowercase)

        with open(vocab_path, encoding='utf-8') as f:
            self.vocab_map = json.load(f)
        self.id_to_token_map = {v: k for k, v in self.vocab_map.items()}

        self.byte_encoder = bytes_to_unicodes()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        with open(merge_path, encoding='utf-8') as merges_handle:
            bpe_merges = merges_handle.read().split('\n')[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

        self.cache = {}
        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token
        self.special_tokens = {self.unk_token, self.sep_token, self.pad_token, self.cls_token, self.mask_token}

        self.unk_token_id = self.vocab_map.get(self.unk_token)
        self.sep_token_id = self.vocab_map.get(self.sep_token)
        self.pad_token_id = self.vocab_map.get(self.pad_token)
        self.cls_token_id = self.vocab_map.get(self.cls_token)
        self.mask_token_id = self.vocab_map.get(self.mask_token)

    def _tokenize(self, text):
        bpe_tokens = []
        for token in re.findall(self.pattern, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8')) # Maps all our bytes to unicode strings, avoiding controle tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self._byte_level_tokenize(token).split(' '))
        return bpe_tokens

    def _byte_level_tokenize(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def _tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            token_id = self.vocab_map.get(token, self.vocab_map.get(self.unk_token))
            ids.append(token_id)
        return ids

    def _ids_to_tokens(self, token_ids):
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token_map.get(token_id, self.unk_token)
            tokens.append(token)
        return tokens

    def _tokens_to_string(self, tokens):
        text = ''.join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors='replace')
        return text


class RobertaTokenizer(RobertaBasicTokenizer):
    def __init__(self, lowercase, task, vocab_path, merge_path, max_seq_len, max_query_len=None):
        super(RobertaTokenizer, self).__init__(lowercase, vocab_path, merge_path)

        self.task = task
        self.max_seq_len = max_seq_len
        self.max_query_len = max_query_len

        # single_tokens: <s> X </s>, paired_tokens: <s> A </s></s> B </s>
        self.num_special_token_single = 2
        self.num_special_token_paired = 4
        self.num_special_token_a_paired = 3
        self.num_special_token_b_paired = 1

    def _combine_and_pad(self, token_ids_a, token_ids_b):
        token_ids = [self.cls_token_id] + token_ids_a + [self.sep_token_id]
        segment_ids = [0] * len(token_ids)  # Will not be used in roberta model
        if token_ids_b is not None:
            token_ids += [self.sep_token_id] + token_ids_b + [self.sep_token_id]
            segment_ids += [0] + [1] * len(token_ids_b + [self.sep_token_id])
        attn_mask = [0] * len(token_ids)

        if len(token_ids) < self.max_seq_len:
            dif = self.max_seq_len - len(token_ids)
            token_ids += [self.pad_token_id] * dif
            segment_ids += [0] * dif
            attn_mask += [1] * dif

        # Replace non-padding symbols with their position numbers, position numbers begin at pad_token_id + 1
        mask = token_ids.ne(self.pad_token_id).long()
        incremental_indicies = torch.cumsum(mask, dim=1) * mask
        position_ids = incremental_indicies + self.pad_token_id
        return token_ids, segment_ids, position_ids, attn_mask
