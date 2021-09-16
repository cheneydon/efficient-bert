import math
import torch
import torch.nn as nn
import models


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        """ Conv1D layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2)
            Basically works like a Linear layer but the weights are transposed
        """
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class Gpt2Embedding(nn.Module):
    def __init__(self, config, num_add_tokens=0):
        super(Gpt2Embedding, self).__init__()

        self.token_embeddings = nn.Embedding(config.vocab_size + num_add_tokens, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.position_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, token_ids, position_ids):
        token_embeddings = self.token_embeddings(token_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = token_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Gpt2Attention(nn.Module):
    def __init__(self, config):
        super(Gpt2Attention, self).__init__()

        hidden_size = config.hidden_size
        n_ctx = config.position_size
        self.register_buffer('bias', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.register_buffer('masked_bias', torch.tensor(-1e4))
        self.split_size = hidden_size
        self.num_attn_heads = config.num_attn_heads
        self.attn_head_size = config.hidden_size // self.num_attn_heads
        self.all_head_size = self.attn_head_size * self.num_attn_heads

        self.softmax = nn.Softmax(dim=-1)
        self.c_attn = Conv1D(hidden_size * 3, hidden_size)
        self.c_proj = Conv1D(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(config.attn_dropout_prob)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states, attn_mask):
        x = self.c_attn(self.layernorm(hidden_states))
        query, key, value = x.split(self.split_size, dim=-1)
        query = query.view(x.size(0), -1, self.num_attn_heads, self.attn_head_size).permute(0, 2, 1, 3)
        key = key.view(x.size(0), -1, self.num_attn_heads, self.attn_head_size).permute(0, 2, 1, 3)
        value = value.view(x.size(0), -1, self.num_attn_heads, self.attn_head_size).permute(0, 2, 1, 3)

        attn_score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attn_head_size)
        nd, ns = attn_score.size(-2), attn_score.size(-1)
        mask = self.bias[:, :, ns - nd : ns, :ns]
        attn_score = torch.where(mask.bool(), attn_score, self.masked_bias.to(attn_score.dtype))  # Mask the upper triangular part of the attention score

        attn_mask = attn_mask[:, None, None, :]
        attn_score += attn_mask * -10000.0
        attn_prob = self.attn_dropout(self.softmax(attn_score))

        context = torch.matmul(attn_prob, value)
        context = context.permute(0, 2, 1, 3).contiguous().view(value.size(0), -1, self.all_head_size)
        context = self.hidden_dropout(self.c_proj(context))
        output = hidden_states + context
        return output, attn_score


class Gpt2FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super(Gpt2FeedForwardNetwork, self).__init__()

        self.c_fc = Conv1D(config.ffn_hidden_size, config.hidden_size)
        self.c_proj = Conv1D(config.hidden_size, config.ffn_hidden_size)
        self.activation = models.gelu
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states):
        x = self.layernorm(hidden_states)
        x = self.activation(self.c_fc(x))
        x = self.dropout(self.c_proj(x))
        output = hidden_states + x
        return output


class Gpt2TransformerBlock(nn.Module):
    def __init__(self, config):
        super(Gpt2TransformerBlock, self).__init__()

        self.attention = Gpt2Attention(config)
        self.ffn = Gpt2FeedForwardNetwork(config)

    def forward(self, hidden_states, attn_mask):
        attn_output, attn_score = self.attention(hidden_states, attn_mask)
        output = self.ffn(attn_output)
        return output, attn_score


class Gpt2(nn.Module):
    def __init__(self, config, num_add_tokens=0, use_lm=False, return_hid=False):
        super(Gpt2, self).__init__()

        self.num_add_tokens = num_add_tokens
        self.use_lm = use_lm
        self.return_hid = return_hid
        self.embeddings = Gpt2Embedding(config, num_add_tokens)
        self.encoder = nn.ModuleList([Gpt2TransformerBlock(config) for _ in range(config.num_layers)])
        self.layernorm = nn.LayerNorm(config.hidden_size)

        if self.use_lm:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size + num_add_tokens, bias=False)
            self.lm_head.weight = self.embeddings.token_embeddings.weight
        self._init_weights()

    def forward(self, token_ids, position_ids, attn_mask):
        all_attn_outputs, all_ffn_outputs = [], []
        output = self.embeddings(token_ids, position_ids)

        for layer in self.encoder:
            all_ffn_outputs.append(output)
            output, attn_output = layer(output, attn_mask)
            all_attn_outputs.append(attn_output)
        output = self.layernorm(output)
        all_ffn_outputs.append(output)

        if self.use_lm:
            output = self.lm_head(output)
        if self.return_hid:
            return output, all_attn_outputs, all_ffn_outputs
        return output

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding, Conv1D)):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(m, (nn.Linear, Conv1D)) and m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
