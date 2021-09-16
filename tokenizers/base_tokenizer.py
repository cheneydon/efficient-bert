import re
import itertools
import datasets


class EncodedInput(object):
    def __init__(self, token_ids, segment_ids, position_ids, attn_mask, overflow_token_ids):
        self.token_ids = token_ids
        self.segment_ids = segment_ids
        self.position_ids = position_ids
        self.attn_mask = attn_mask
        self.overflow_token_ids = overflow_token_ids


class BaseTokenizer(object):
    def __init__(self, lowercase):
        self.lowercase = lowercase
        self.model_name = None
        self.task = None
        self.max_seq_len = None
        self.max_query_len = None
        self.special_tokens = None

        self.num_add_tokens = 0
        self.num_special_token_single = None
        self.num_special_token_paired = None
        self.num_special_token_a_paired = None
        self.num_special_token_b_paired = None

    def tokenize(self, text):
        if self.lowercase:
            text = self._lowercase_text(text)

        if not text.strip():
            return []
        tokenized_text = []
        text_list = [text]
        for tok in self.special_tokens:
            tokenized_text = []
            for sub_text in text_list:
                if sub_text not in self.special_tokens:
                    tokenized_text += self._split_on_special_token(tok, sub_text)
                else:
                    tokenized_text += [sub_text]
            text_list = tokenized_text

        return list(itertools.chain.from_iterable([
            self._tokenize(token) if token not in self.special_tokens else [token] for token in tokenized_text]))

    def encode(self, text_a, text_b=None, added_trunc_size=None):
        token_ids_a = self._get_input_ids(text_a)
        token_ids_b = self._get_input_ids(text_b) if text_b is not None else None
        token_ids_a, token_ids_b, overflow_token_ids = self._truncate_tokens(token_ids_a, token_ids_b, added_trunc_size)
        token_ids, segment_ids, position_ids, attn_mask = self._combine_and_pad(token_ids_a, token_ids_b)
        encoded_input = EncodedInput(token_ids, segment_ids, position_ids, attn_mask, overflow_token_ids)
        return encoded_input

    def decode(self, token_ids, clean_up_tokenization_spaces=True):
        """ Converts a sequence of token ids in a string.
        """
        filtered_tokens = self._ids_to_tokens(token_ids)

        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separatly for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self._tokens_to_string(current_sub_text))
        text = ' '.join(sub_texts)

        if clean_up_tokenization_spaces:
            clean_text = self._clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def _lowercase_text(self, text):
        # convert non-special tokens to lowercase
        escaped_special_toks = [re.escape(s_tok) for s_tok in self.special_tokens]
        pattern = r'(' + r'|'.join(escaped_special_toks) + r')|' + r'(.+?)'
        return re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)

    def _split_on_special_token(self, special_token, text):
        result = []
        split_text = text.split(special_token)
        for i, sub_text in enumerate(split_text):
            sub_text = sub_text.strip()
            if i == 0 and not sub_text:
                result += [special_token]
            elif i == len(split_text) - 1:
                if sub_text:
                    result += [sub_text]
            else:
                if sub_text:
                    result += [sub_text]
                result += [special_token]
        return result

    def _get_input_ids(self, text):
        if isinstance(text, str):
            return self._tokens_to_ids(self.tokenize(text))
        elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
            return self._tokens_to_ids(text)
        elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
            return text
        else:
            raise ValueError('Input is not valid, should be a string, a list/tuple of strings or a list/tuple of integers.')

    def _truncate_tokens(self, token_ids_a, token_ids_b, added_trunc_size=None):
        overflow_token_ids = None
        if self.task in datasets.glue_tasks:
            token_ids_a, token_ids_b = self._truncate_glue(token_ids_a, token_ids_b)
            return token_ids_a, token_ids_b, overflow_token_ids
        elif self.task in datasets.squad_tasks:
            token_ids_a, token_ids_b, overflow_token_ids = self._truncate_squad(token_ids_a, token_ids_b, added_trunc_size)
            return token_ids_a, token_ids_b, overflow_token_ids
        elif self.task in datasets.multi_choice_tasks:
            token_ids_a, token_ids_b = self._truncate_multi_choice(token_ids_a, token_ids_b)
            return token_ids_a, token_ids_b, overflow_token_ids
        elif self.task in datasets.summarization_tasks:
            token_ids_a, token_ids_b = self._truncate_summarization(token_ids_a, token_ids_b)
            return token_ids_a, token_ids_b, overflow_token_ids

    def _truncate_glue(self, token_ids_a, token_ids_b):
        total_len = self._get_total_len(token_ids_a, token_ids_b)
        if total_len > self.max_seq_len:
            num_remove = total_len - self.max_seq_len
            for _ in range(num_remove):
                if token_ids_b is None or len(token_ids_a) > len(token_ids_b):
                    token_ids_a = token_ids_a[:-1]
                else:
                    token_ids_b = token_ids_b[:-1]
        return token_ids_a, token_ids_b

    def _truncate_squad(self, token_ids_a, token_ids_b, added_trunc_size):
        overflow_token_ids = None
        len_a = len(token_ids_a) + self.num_special_token_a_paired
        if len_a > self.max_query_len:
            num_remove = len_a - self.max_query_len
            token_ids_a = token_ids_a[:-num_remove]

        total_len = self._get_total_len(token_ids_a, token_ids_b)
        if total_len > self.max_seq_len:
            num_remove = total_len - self.max_seq_len
            trunc_size = min(len(token_ids_b), added_trunc_size + num_remove)
            overflow_token_ids = token_ids_b[-trunc_size:]
            token_ids_b = token_ids_b[:-num_remove]
        return token_ids_a, token_ids_b, overflow_token_ids

    def _truncate_multi_choice(self, token_ids_a, token_ids_b):
        len_b = len(token_ids_b) + self.num_special_token_b_paired
        if len_b > self.max_query_len:
            num_remove = len_b - self.max_query_len
            token_ids_b = token_ids_b[:-num_remove]

        total_len = self._get_total_len(token_ids_a, token_ids_b)
        if total_len > self.max_seq_len:
            num_remove = total_len - self.max_seq_len
            token_ids_a = token_ids_a[:-num_remove]
        return token_ids_a, token_ids_b

    def _truncate_summarization(self, token_ids_a, token_ids_b):
        len_b = len(token_ids_b) + self.num_special_token_b_paired
        if len_b > self.max_query_len:
            num_remove = len_b - self.max_query_len
            token_ids_b = token_ids_b[:-num_remove]

        len_a = len(token_ids_a) + self.num_special_token_a_paired
        max_len = self.max_seq_len - self.max_query_len
        if len_a > max_len:
            num_remove = len_a - max_len
            token_ids_a = token_ids_a[:-num_remove]

        return token_ids_a, token_ids_b

    def _get_total_len(self, token_ids_a, token_ids_b=None):
        if token_ids_b is None:
            return len(token_ids_a) + self.num_special_token_single
        return len(token_ids_a) + len(token_ids_b) + self.num_special_token_paired

    @staticmethod
    def _clean_up_tokenization(out_string):
        """ Clean up a list of simple English tokenization artifacts like spaces before punctuations and abreviated forms.
        """
        out_string = (
            out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" do not", " don't")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return out_string

    def _tokenize(self, text):
        raise NotImplementedError

    def _tokens_to_ids(self, tokens):
        raise NotImplementedError

    def _ids_to_tokens(self, token_ids):
        raise NotImplementedError

    def _tokens_to_string(self, tokens):
        raise NotImplementedError

    def _combine_and_pad(self, token_ids_a, token_ids_b):
        raise NotImplementedError
