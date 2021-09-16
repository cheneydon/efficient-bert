import sentencepiece as spm
import unicodedata
import datasets
import tokenizers

SPIECE_UNDERLINE = u'▁'


class XlnetBasicTokenizer(tokenizers.BaseTokenizer):
    def __init__(self, lowercase, vocab_path, unk_token='<unk>', sep_token='<sep>', pad_token='<pad>',
                 cls_token='<cls>', mask_token='<mask>'):
        super(XlnetBasicTokenizer, self).__init__(lowercase)

        self.lowercase = lowercase
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_path)

        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token
        self.special_tokens = {self.unk_token, self.sep_token, self.pad_token, self.cls_token, self.mask_token}

        self.unk_token_id = self.sp_model.PieceToId(self.unk_token)
        self.sep_token_id = self.sp_model.PieceToId(self.sep_token)
        self.pad_token_id = self.sp_model.PieceToId(self.pad_token)
        self.cls_token_id = self.sp_model.PieceToId(self.cls_token)
        self.mask_token_id = self.sp_model.PieceToId(self.mask_token)

    def _tokenize(self, text):
        text = self._prepocess_text(text)
        text = self._sentence_piece_tokenize(text)
        return text

    def _prepocess_text(self, text):
        output = ' '.join(text.strip().split())
        output = output.replace("``", '"').replace("''", '"')
        output = unicodedata.normalize('NFKD', output)
        output = ''.join([c for c in output if not unicodedata.combining(c)])
        if self.lowercase:
            output = output.lower()
        return output

    def _sentence_piece_tokenize(self, text):
        pieces = self.sp_model.EncodeAsPieces(text)
        new_pieces = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == str(',') and piece[-2].isdigit():
                cur_pieces = self.sp_model.EncodeAsPieces(piece[:-1].replace(SPIECE_UNDERLINE, ''))
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)
        return new_pieces

    def _tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            token_id = self.sp_model.PieceToId(token)
            ids.append(token_id)
        return ids

    def _ids_to_tokens(self, token_ids):
        tokens = []
        for token_id in token_ids:
            token = self.sp_model.IdToPiece(token_id)
            tokens.append(token)
        return tokens

    def _tokens_to_string(self, tokens):
        text = ''.join(tokens).replace(SPIECE_UNDERLINE, ' ').strip()
        return text


class XlnetTokenizer(XlnetBasicTokenizer):
    def __init__(self, lowercase, task, vocab_path, max_seq_len, max_query_len=None):
        super(XlnetTokenizer, self).__init__(lowercase, vocab_path)

        self.task = task
        self.max_seq_len = max_seq_len
        self.max_query_len = max_query_len

        # single_tokens: X <sep><cls>, paired_tokens: A <sep> B <sep><cls>
        self.num_special_token_single = 2
        self.num_special_token_paired = 3
        self.num_special_token_a_paired = 1
        self.num_special_token_b_paired = 2

    def _truncate_squad(self, token_ids_a, token_ids_b, added_trunc_size):
        overflow_token_ids = None
        len_b = len(token_ids_b) + self.num_special_token_b_paired
        if len_b > self.max_query_len:
            num_remove = len_b - self.max_query_len
            token_ids_b = token_ids_b[:-num_remove]

        total_len = self._get_total_len(token_ids_a, token_ids_b)
        if total_len > self.max_seq_len:
            num_remove = total_len - self.max_seq_len
            trunc_size = min(len(token_ids_a), added_trunc_size + num_remove)
            overflow_token_ids = token_ids_a[-trunc_size:]
            token_ids_a = token_ids_a[:-num_remove]
        return token_ids_a, token_ids_b, overflow_token_ids

    def _combine_and_pad(self, token_ids_a, token_ids_b):
        token_ids = token_ids_a + [self.sep_token_id]
        segment_ids = [0] * len(token_ids)
        if token_ids_b is not None:
            token_ids += token_ids_b + [self.sep_token_id]
            segment_ids += [1] * len(token_ids_b + [self.sep_token_id])
        token_ids += [self.cls_token_id]
        segment_ids += [2]
        attn_mask = [0] * len(token_ids)

        pad_segment_id = 3 if self.task in datasets.squad_tasks else 4
        if len(token_ids) < self.max_seq_len:
            dif = self.max_seq_len - len(token_ids)
            token_ids = [self.pad_token_id] * dif + token_ids
            segment_ids = [pad_segment_id] * dif + segment_ids
            attn_mask = [1] * dif + attn_mask

        position_ids = [i for i in range(self.max_seq_len)]  # Will not be used in xlnet model
        return token_ids, segment_ids, position_ids, attn_mask
