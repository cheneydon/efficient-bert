import torch.nn as nn
import datasets
import models


class SupernetFeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super(SupernetFeedForwardNetwork, self).__init__()

        ffn_hidden_size = int(config.ffn_hidden_size * config.ffn_expansion_ratio)
        self.dense1 = nn.Linear(config.hidden_size, ffn_hidden_size)
        self.activation = models.gelu
        self.dense2 = nn.Linear(ffn_hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states):
        output = self.activation(self.dense1(hidden_states))
        output = self.dropout(self.dense2(output))
        output = self.layernorm(hidden_states + output)
        return output


class SupernetTransformerBlock(nn.Module):
    def __init__(self, config):
        super(SupernetTransformerBlock, self).__init__()

        self.attention = models.BertAttention(config)
        self.all_ffn = nn.ModuleList([SupernetFeedForwardNetwork(config) for _ in range(config.num_stacked_ffn)])

    def forward(self, hidden_states, attn_mask, ret_all_ffn_hidden_states=False):
        all_ffn_hidden_states = []
        output, attn_score = self.attention(hidden_states, attn_mask)
        for layer in self.all_ffn:
            output = layer(output)
            all_ffn_hidden_states.append(output)
        if ret_all_ffn_hidden_states:
            return (output, all_ffn_hidden_states), attn_score
        return output, attn_score


class SupernetSingle(nn.Module):
    def __init__(self, config, use_lm=False, ret_all_ffn_hidden_states=False):
        super(SupernetSingle, self).__init__()

        self.use_fit_dense = config.use_fit_dense
        self.ret_all_ffn_hidden_states = ret_all_ffn_hidden_states
        self.use_lm = use_lm
        if config.use_mobile_embed:
            self.embeddings = models.MobileBertEmbedding(config)
        else:
            self.embeddings = models.BertEmbedding(config)
        self.encoder = nn.ModuleList([SupernetTransformerBlock(config) for _ in range(config.num_layers)])

        if self.use_fit_dense:
            self.fit_dense = nn.Linear(config.hidden_size, config.fit_size)
        if self.use_lm:
            self.lm_head = models.BertMaskedLMHead(config, self.embeddings.token_embeddings.weight)
        self._init_weights()

    def forward(self, token_ids, segment_ids, position_ids, attn_mask):
        all_attn_outputs, all_ffn_outputs = [], []
        output = self.embeddings(token_ids, segment_ids, position_ids)
        if self.use_fit_dense:
            all_ffn_outputs.append(self.fit_dense(output))
        else:
            all_ffn_outputs.append(output)

        for layer in self.encoder:
            output, attn_output = layer(output, attn_mask, self.ret_all_ffn_hidden_states)
            all_attn_outputs.append(attn_output)
            if self.use_fit_dense:
                if self.ret_all_ffn_hidden_states:
                    all_ffn_outputs.append((self.fit_dense(output[0]), output[1]))
                    output = output[0]
                else:
                    all_ffn_outputs.append(self.fit_dense(output))
            else:
                all_ffn_outputs.append(output)

        if self.use_lm:
            output = self.lm_head(output)
        return output, all_attn_outputs, all_ffn_outputs

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()


class Supernet(nn.Module):
    def __init__(self, config, task, return_hidden_states=False, ret_all_ffn_hidden_states=False):
        super(Supernet, self).__init__()

        self.use_fit_dense = config.use_fit_dense
        self.ret_all_ffn_hidden_states = ret_all_ffn_hidden_states
        self.task = task
        self.return_hidden_states = return_hidden_states
        if config.use_mobile_embed:
            self.embeddings = models.MobileBertEmbedding(config)
        else:
            self.embeddings = models.BertEmbedding(config)
        self.encoder = nn.ModuleList([SupernetTransformerBlock(config) for _ in range(config.num_layers)])

        if self.use_fit_dense:
            self.fit_dense = nn.Linear(config.hidden_size, config.fit_size)
        if task in datasets.glue_tasks:
            self.num_classes = datasets.glue_num_classes[task]
            self.cls_pooler = models.BertClsPooler(config)
        elif task in datasets.squad_tasks:
            self.num_classes = 2
        elif task in datasets.multi_choice_tasks:
            self.num_classes = 1
            self.cls_pooler = models.BertClsPooler(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_classes)
        self._init_weights()

    def forward(self, token_ids, segment_ids, position_ids, attn_mask):
        if self.task in datasets.multi_choice_tasks:
            num_choices = token_ids.size(1)
            token_ids = token_ids.view(-1, token_ids.size(-1))
            segment_ids = segment_ids.view(-1, segment_ids.size(-1))
            position_ids = position_ids.view(-1, position_ids.size(-1))
            attn_mask = attn_mask.view(-1, attn_mask.size(-1))

        all_attn_outputs, all_ffn_outputs = [], []
        output = self.embeddings(token_ids, segment_ids, position_ids)
        if self.use_fit_dense:
            all_ffn_outputs.append(self.fit_dense(output))
        else:
            all_ffn_outputs.append(output)

        for layer in self.encoder:
            output, attn_output = layer(output, attn_mask, self.ret_all_ffn_hidden_states)
            all_attn_outputs.append(attn_output)
            if self.use_fit_dense:
                if self.ret_all_ffn_hidden_states:
                    all_ffn_outputs.append((self.fit_dense(output[0]), output[1]))
                    output = output[0]
                else:
                    all_ffn_outputs.append(self.fit_dense(output))
            else:
                all_ffn_outputs.append(output)

        if self.task in datasets.glue_tasks:
            output = self.cls_pooler(output[:, 0])
            output = self.classifier(output).squeeze(-1)
            if self.return_hidden_states:
                return output, all_attn_outputs, all_ffn_outputs
            return output
        elif self.task in datasets.squad_tasks:
            output = self.classifier(output)
            start_logits, end_logits = output.split(1, dim=-1)
            if self.return_hidden_states:
                return start_logits.squeeze(-1), end_logits.squeeze(-1), all_attn_outputs, all_ffn_outputs
            return start_logits.squeeze(-1), end_logits.squeeze(-1)
        elif self.task in datasets.multi_choice_tasks:
            output = self.cls_pooler(output[:, 0])
            output = self.classifier(output)
            output = output.view(-1, num_choices)
            if self.return_hidden_states:
                return output, all_attn_outputs, all_ffn_outputs
            return output

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
