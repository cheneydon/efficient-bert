import unicodedata
import tokenizers
from collections import OrderedDict


def _is_whitespace(char):
    if char == ' ' or char == '\t' or char == '\n' or char == '\r':
        return True
    cat = unicodedata.category(char)
    if cat == 'Zs':
        return True
    return False


def _is_control(char):
    if char == '\t' or char == '\n' or char == '\r':
        return False
    cat = unicodedata.category(char)
    if cat.startswith('C'):
        return True
    return False


def _is_punctuation(char):
    cp = ord(char)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith('P'):
        return True
    return False


def load_vocab(vocab_path):
    vocab_map = OrderedDict()
    with open(vocab_path, 'r', encoding='utf-8') as f:
        tokens = f.readlines()
    for idx, token in enumerate(tokens):
        token = token.rstrip('\n')
        vocab_map[token] = idx
    return vocab_map


class BertBasicTokenizer(tokenizers.BaseTokenizer):
    def __init__(self, lowercase, vocab_path, unk_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]',
                 cls_token='[CLS]', mask_token='[MASK]'):
        super(BertBasicTokenizer, self).__init__(lowercase)

        self.lowercase = lowercase
        self.vocab_map = load_vocab(vocab_path)
        self.id_to_token_map = OrderedDict([(id, token) for token, id in self.vocab_map.items()])

        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token
        self.special_tokens = {self.unk_token, self.sep_token, self.pad_token, self.cls_token, self.mask_token}

        self.unk_token_id = self.vocab_map[self.unk_token]
        self.sep_token_id = self.vocab_map[self.sep_token]
        self.pad_token_id = self.vocab_map[self.pad_token]
        self.cls_token_id = self.vocab_map[self.cls_token]
        self.mask_token_id = self.vocab_map[self.mask_token]

    def _tokenize(self, text):
        text = self._basic_tokenize(text)
        all_subtokens = []
        for token in text:
            for subtoken in self._wordpiece_tokenize(token):
                all_subtokens.append(subtoken)
        return all_subtokens

    def _basic_tokenize(self, text):
        text = self._clean_text(text)

        split_tokens = []
        for token in text.strip().split():
            if self.lowercase and token not in self.special_tokens:
                token = token.lower()
                token = self._strip_accents(token)
            split_tokens.extend(self._split_on_punc(token))
        return ' '.join(split_tokens).strip().split()

    def _clean_text(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)

    def _strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize('NFD', text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == 'Mn':
                continue
            output.append(char)
        return ''.join(output)

    def _split_on_punc(self, token):
        """Splits punctuation on a piece of text."""
        output = []
        start_new_token = True
        for char in token:
            if _is_punctuation(char):
                output.append([char])
                start_new_token = True
            else:
                if start_new_token:
                    output.append([])
                start_new_token = False
                output[-1].append(char)
        return [''.join(token) for token in output]

    def _wordpiece_tokenize(self, raw_token, max_word_len=100):
        output = []
        for token in raw_token.strip().split():
            chars = list(token)
            if len(chars) > max_word_len:
                output.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            valid_subtokens = []
            while start < len(chars):
                end = len(chars)
                valid = None
                while start < end:
                    subtoken = ''.join(chars[start:end])
                    if start > 0:
                        subtoken = '##' + subtoken
                    if subtoken in self.vocab_map:
                        valid = subtoken
                        break
                    end -= 1
                if valid is None:
                    is_bad = True
                    break
                valid_subtokens.append(valid)
                start = end

            if is_bad:
                output.append(self.unk_token)
            else:
                output.extend(valid_subtokens)
        return output

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
        text = ' '.join(tokens).replace(' ##', '').strip()
        return text


class BertTokenizer(BertBasicTokenizer):
    def __init__(self, lowercase, task, vocab_path, max_seq_len, max_query_len=None):
        super(BertTokenizer, self).__init__(lowercase, vocab_path)

        self.task = task
        self.max_seq_len = max_seq_len
        self.max_query_len = max_query_len

        # single_tokens: [CLS] X [SEP], paired_tokens: [CLS] A [SEP] B [SEP]
        self.num_special_token_single = 2
        self.num_special_token_paired = 3
        self.num_special_token_a_paired = 2
        self.num_special_token_b_paired = 1

    def _combine_and_pad(self, token_ids_a, token_ids_b):
        token_ids = [self.cls_token_id] + token_ids_a + [self.sep_token_id]
        segment_ids = [0] * len(token_ids)
        if token_ids_b is not None:
            token_ids += token_ids_b + [self.sep_token_id]
            segment_ids += [1] * len(token_ids_b + [self.sep_token_id])
        attn_mask = [0] * len(token_ids)

        if len(token_ids) < self.max_seq_len:
            dif = self.max_seq_len - len(token_ids)
            token_ids += [self.pad_token_id] * dif
            segment_ids += [0] * dif
            attn_mask += [1] * dif

        position_ids = [i for i in range(self.max_seq_len)]
        return token_ids, segment_ids, position_ids, attn_mask
