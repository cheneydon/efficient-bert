import torch
import torch.nn as nn
import datasets
import models
from torch.nn import functional as F


class XlnetAttention(nn.Module):
    def __init__(self, config):
        super(XlnetAttention, self).__init__()

        self.num_attn_heads = config.num_attn_heads
        self.attn_head_size = config.hidden_size // self.num_attn_heads
        self.all_head_size = self.attn_head_size * self.num_attn_heads
        self.scale = 1 / (self.attn_head_size ** 0.5)

        self.q = nn.Parameter(torch.FloatTensor(config.hidden_size, self.num_attn_heads, self.attn_head_size))
        self.k = nn.Parameter(torch.FloatTensor(config.hidden_size, self.num_attn_heads, self.attn_head_size))
        self.v = nn.Parameter(torch.FloatTensor(config.hidden_size, self.num_attn_heads, self.attn_head_size))
        self.r = nn.Parameter(torch.FloatTensor(config.hidden_size, self.num_attn_heads, self.attn_head_size))
        self.o = nn.Parameter(torch.FloatTensor(config.hidden_size, self.num_attn_heads, self.attn_head_size))

        self.r_w_bias = nn.Parameter(torch.FloatTensor(self.num_attn_heads, self.attn_head_size))
        self.r_r_bias = nn.Parameter(torch.FloatTensor(self.num_attn_heads, self.attn_head_size))
        self.r_s_bias = nn.Parameter(torch.FloatTensor(self.num_attn_heads, self.attn_head_size))
        self.seg_mat = nn.Parameter(torch.FloatTensor(2, self.num_attn_heads, self.attn_head_size))

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(config.attn_dropout_prob)
        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, position_embed, segment_embed, attn_mask):
        query = torch.einsum('ibh,hnd->ibnd', hidden_states, self.q)
        key = torch.einsum('ibh,hnd->ibnd', hidden_states, self.k)
        value = torch.einsum('ibh,hnd->ibnd', hidden_states, self.v)
        key_r = torch.einsum('ibh,hnd->ibnd', position_embed, self.r)

        # Context based attention score
        ac = torch.einsum('ibnd,jbnd->bnij', query + self.r_w_bias, key)
        # Position based attention score
        bd = torch.einsum('ibnd,jbnd->bnij', query + self.r_r_bias, key_r)
        bd = self._rel_shift(bd, ac.size(3))
        # Segment based attention score
        ef = torch.einsum('ibnd,snd->ibns', query + self.r_s_bias, self.seg_mat)
        ef = torch.einsum('ijbs,ibns->bnij', segment_embed, ef)

        big_num = 65500 if attn_mask.dtype == torch.float16 else 1e30
        attn_score = (ac + bd + ef) * self.scale
        attn_score = attn_score - big_num * torch.einsum("ijbn->bnij", attn_mask)
        attn_prob = self.attn_dropout(self.softmax(attn_score))

        attn_out = torch.einsum('bnij,jbnd->ibnd', attn_prob, value)
        attn_out = self.hidden_dropout(torch.einsum('ibnd,hnd->ibh', attn_out, self.o))
        output = self.layernorm(attn_out + hidden_states)
        return output

    @staticmethod
    def _rel_shift(x, klen):
        x_size = x.shape
        x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        x = x[:, :, 1:, :]
        x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        # Note: the tensor-slice form was faster in my testing than torch.index_select
        #       However, tracing doesn't like the nature of the slice, and if klen changes
        #       during the run then it'll fail, whereas index_select will be fine.
        x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        # x = x[:, :, :, :klen]
        return x


class XlnetFeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super(XlnetFeedForwardNetwork, self).__init__()

        self.dense1 = nn.Linear(config.hidden_size, config.ffn_hidden_size)
        self.activation = models.gelu
        self.dense2 = nn.Linear(config.ffn_hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states):
        output = self.dense1(hidden_states)
        output = self.dropout(self.activation(output))
        output = self.dropout(self.dense2(output))
        output = self.layernorm(hidden_states + output)
        return output


class XlnetTransformerBlock(nn.Module):
    def __init__(self, config):
        super(XlnetTransformerBlock, self).__init__()

        self.rel_attn = XlnetAttention(config)
        self.ffn = XlnetFeedForwardNetwork(config)

    def forward(self, h, r, segment_embed, attn_mask, return_attn_output=False):
        attn_output = self.rel_attn(h, r, segment_embed, attn_mask)
        output = self.ffn(attn_output)
        if return_attn_output:
            return attn_output, output
        return output


class XlnetBaseModel(nn.Module):
    def __init__(self, config):
        super(XlnetBaseModel, self).__init__()

        self.hidden_size = config.hidden_size
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, token_ids, segment_ids, attn_mask):
        token_ids = token_ids.transpose(0, 1).contiguous()
        segment_ids = segment_ids.transpose(0, 1).contiguous()
        attn_mask = attn_mask.transpose(0, 1).contiguous()
        seq_len, bsz = token_ids.size()

        attn_mask = attn_mask[None, :, :, None]
        non_target_mask = -torch.eye(seq_len).to(attn_mask)
        non_target_mask = ((attn_mask + non_target_mask[:, :, None, None]) > 0).to(attn_mask)

        token_embed = self.dropout(self.token_embeddings(token_ids))
        segment_embed = (segment_ids[:, None] != segment_ids[None, :]).long()
        segment_embed = F.one_hot(segment_embed, num_classes=2).to(next(self.parameters()).dtype)
        position_embed = self._get_position_embedding(seq_len, bsz)

        return token_embed, segment_embed, position_embed, non_target_mask

    def _get_position_embedding(self, seq_len, bsz):
        freq_seq = torch.arange(0, self.hidden_size, 2., dtype=torch.float)
        inv_freq = 1 / torch.pow(10000, (freq_seq / self.hidden_size))
        pos_seq = torch.arange(seq_len, -seq_len, -1.)

        sinusoid_inp = torch.einsum('i,d->id', pos_seq, inv_freq)
        pos_embed = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_embed = pos_embed[:, None, :].expand(-1, bsz, -1)
        pos_embed = self.dropout(pos_embed.to(next(self.parameters())))
        return pos_embed


class XlnetClsPooler(nn.Module):
    def __init__(self, config):
        super(XlnetClsPooler, self).__init__()

        self.dense1 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.dense2 = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states, cls_index, start_positions=None, start_states=None):
        hidden_size = hidden_states.size(-1)
        if start_positions is not None:
            start_positions = start_positions[:, None, None].expand(-1, -1, hidden_size)  # (batch_size, 1, hidden_size)
            start_states = hidden_states.gather(1, start_positions).squeeze(1)  # (batch_size, hidden_size)

        cls_index = cls_index[:, None, None].expand(-1, -1, hidden_size)  # (batch_size, 1, hidden_size)
        cls_token_state = hidden_states.gather(1, cls_index).squeeze(1)  # (batch_size, hidden_size)

        x = self.dense1(torch.cat([start_states, cls_token_state], dim=-1))
        x = self.activation(x)
        x = self.dense2(x).squeeze(-1)
        return x


class StartLogitPooler(nn.Module):
    def __init__(self, config):
        super(StartLogitPooler, self).__init__()

        self.dense = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, p_mask):
        x = self.dense(hidden_states).squeeze(-1)

        big_num = 65500 if next(self.parameters()).dtype == torch.float16 else 1e30
        x = x * (1 - p_mask) - big_num * p_mask
        return x


class EndLogitPooler(nn.Module):
    def __init__(self, config):
        super(EndLogitPooler, self).__init__()

        self.dense1 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.dense2 = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, p_mask, start_positions=None, start_states=None):
        if start_positions is not None:
            seq_len, hidden_size = hidden_states.size(1), hidden_states.size(2)
            start_positions = start_positions[:, None, None].expand(-1, -1, hidden_size)  # (batch_size, 1, hidden_size)
            start_states = hidden_states.gather(1, start_positions)  # (batch_size, 1, hidden_size)
            start_states = start_states.expand(-1, seq_len, -1)  # (batch_size, seq_len, hidden_size)

        x = self.dense1(torch.cat([hidden_states, start_states], dim=-1))
        x = self.activation(x)
        x = self.layernorm(x)
        x = self.dense2(x).squeeze(-1)

        big_num = 65500 if next(self.parameters()).dtype == torch.float16 else 1e30
        x = x * (1 - p_mask) - big_num * p_mask
        return x


class XlnetSingle(nn.Module):
    def __init__(self, config):
        super(XlnetSingle, self).__init__()

        self.base_model = XlnetBaseModel(config)
        self.encoder = nn.ModuleList([XlnetTransformerBlock(config) for _ in range(config.num_layers)])
        self._init_weights()

    def forward(self, token_ids, segment_ids, position_ids, attn_mask):
        all_hidden_states = []
        token_embed, segment_embed, position_embed, attn_mask = self.base_model(token_ids, segment_ids, attn_mask)
        all_hidden_states.append(token_embed)

        output = token_embed
        for layer in self.encoder:
            attn_output, output = layer(output, position_embed, segment_embed, attn_mask, return_attn_output=True)
            all_hidden_states.append(attn_output)
            all_hidden_states.append(output)
        return output, all_hidden_states

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
            elif isinstance(m, XlnetAttention):
                for param in [m.q, m.k, m.v, m.r, m.o, m.r_w_bias, m.r_r_bias, m.r_s_bias, m.seg_mat]:
                    param.data.normal_(mean=0.0, std=0.02)


class Xlnet(nn.Module):
    def __init__(self, config, task):
        super(Xlnet, self).__init__()

        self.task = task
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top
        self.base_model = XlnetBaseModel(config)
        self.encoder = nn.ModuleList([XlnetTransformerBlock(config) for _ in range(config.num_layers)])

        if task in datasets.glue_tasks:
            self.num_classes = datasets.glue_num_classes[task]
            self.cls_pooler = models.BertClsPooler(config)
            self.classifier = nn.Linear(config.hidden_size, self.num_classes)
        elif task in datasets.squad_tasks:
            self.num_classes = 2
            self.cls_pooler = XlnetClsPooler(config)
            self.start_pooler = StartLogitPooler(config)
            self.end_pooler = EndLogitPooler(config)
        elif task in datasets.multi_choice_tasks:
            self.num_classes = 1
            self.cls_pooler = models.BertClsPooler(config)
            self.classifier = nn.Linear(config.hidden_size, self.num_classes)
        self._init_weights()

    def forward(self, token_ids, segment_ids, position_ids, attn_mask, start_positions=None,
                cls_index=None, p_mask=None):
        if self.task in datasets.multi_choice_tasks:
            num_choices = token_ids.size(1)
            token_ids = token_ids.view(-1, token_ids.size(-1))
            attn_mask = attn_mask.view(-1, attn_mask.size(-1))
        token_embed, segment_embed, position_embed, attn_mask = self.base_model(token_ids, segment_ids, attn_mask)

        output = token_embed
        for layer in self.encoder:
            output = layer(output, position_embed, segment_embed, attn_mask)

        if self.task in datasets.glue_tasks:
            output = self.cls_pooler(output[:, -1])
            output = self.classifier(output).squeeze(-1)
            return output

        elif self.task in datasets.squad_tasks:
            start_logits = self.start_pooler(output, p_mask)
            if start_positions is None:
                end_logits = self.end_pooler(output, p_mask, start_positions)
                cls_logits = self.cls_pooler(output, cls_index, start_positions)
                return cls_logits, start_logits, end_logits
            else:
                # During inference, compute the end logits based on beam search
                seq_len, hidden_size = output.size(1), output.size(2)
                start_log_probs = F.softmax(start_logits, dim=-1)
                start_top_log_probs, start_top_index = torch.topk(start_log_probs, self.start_n_top, dim=-1)  # (batch_size, start_n_top)

                start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hidden_size)  # (batch_size, start_n_top, hidden_size)
                start_states = torch.gather(output, 1, start_top_index_exp)
                start_states = start_states.unsqueeze(1).expand(-1, seq_len, -1, -1)  # (batch_size, seq_len, start_n_top, hidden_size)
                output_exp = output.unsqueeze(2).expand_as(start_states)  # (batch_size, seq_len, start_n_top, hidden_size)
                p_mask = p_mask.unsqueeze(-1)
                end_logits = self.end_pooler(output_exp, p_mask, start_states=start_states)
                end_log_probs = F.softmax(end_logits, dim=-1)
                end_top_log_probs, end_top_index = torch.topk(end_log_probs, self.end_n_top, dim=-1)  # (batch_size, end_n_top, start_n_top)
                end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
                end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

                start_states = torch.einsum('blh,bl->bh', output, start_log_probs)  # Get the representation of START as weighted sum of hidden states
                cls_logits = self.cls_pooler(output, cls_index, start_states=start_states)
                return start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits

        elif self.task in datasets.multi_choice_tasks:
            output = self.cls_pooler(output[:, -1])
            output = self.classifier(output)
            output = output.view(-1, num_choices)
            return output

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
            elif isinstance(m, XlnetAttention):
                for param in [m.q, m.k, m.v, m.r, m.o, m.r_w_bias, m.r_r_bias, m.r_s_bias, m.seg_mat]:
                    param.data.normal_(mean=0.0, std=0.02)
