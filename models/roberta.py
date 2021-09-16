import torch.nn as nn
import datasets
import models


class RobertaEmbedding(nn.Module):
    def __init__(self, config):
        super(RobertaEmbedding, self).__init__()

        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.position_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, token_ids, position_ids):
        token_embeddings = self.token_embeddings(token_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = token_embeddings + position_embeddings
        embeddings = self.dropout(self.layernorm(embeddings))
        return embeddings


class RobertaPooler(nn.Module):
    def __init__(self, config):
        super(RobertaPooler, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)
        output = self.activation(self.dense(output))
        output = self.dropout(output)
        return output


class RobertaSingle(models.BertSingle):
    def __init__(self, config):
        super(RobertaSingle, self).__init__(config)
        self.embeddings = RobertaEmbedding(config)


class Roberta(nn.Module):
    def __init__(self, config, task):
        super(Roberta, self).__init__()

        self.task = task
        self.embeddings = RobertaEmbedding(config)
        self.encoder = nn.ModuleList([models.BertTransformerBlock(config) for _ in range(config.num_layers)])

        if task in datasets.glue_tasks:
            self.num_classes = datasets.glue_num_classes[task]
            self.cls_pooler = RobertaPooler(config)
        elif task in datasets.squad_tasks:
            self.num_classes = 2
        elif task in datasets.multi_choice_tasks:
            self.num_classes = 1
            self.cls_pooler = RobertaPooler(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_classes)
        self._init_weights()

    def forward(self, token_ids, segment_ids, position_ids, attn_mask):
        if self.task in datasets.multi_choice_tasks:
            num_choices = token_ids.size(1)
            token_ids = token_ids.view(-1, token_ids.size(-1))
            position_ids = position_ids.view(-1, position_ids.size(-1))
            attn_mask = attn_mask.view(-1, attn_mask.size(-1))

        output = self.embeddings(token_ids, position_ids)
        for layer in self.encoder:
            output = layer(output, attn_mask)

        if self.task in datasets.glue_tasks:
            output = self.cls_pooler(output[:, 0])
            output = self.classifier(output).squeeze(-1)
            return output
        elif self.task in datasets.squad_tasks:
            output = self.classifier(output)
            start_logits, end_logits = output.split(1, dim=-1)
            return start_logits.squeeze(-1), end_logits.squeeze(-1)
        elif self.task in datasets.multi_choice_tasks:
            output = self.cls_pooler(output[:, 0])
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
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
