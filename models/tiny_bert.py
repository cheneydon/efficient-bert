import torch.nn as nn
import datasets
import models


class TinyBertSingle(nn.Module):
    def __init__(self, config, use_lm=False):
        super(TinyBertSingle, self).__init__()

        self.use_fit_dense = config.use_fit_dense
        self.use_lm = use_lm
        self.embeddings = models.BertEmbedding(config)
        self.encoder = nn.ModuleList([models.BertTransformerBlock(config) for _ in range(config.num_layers)])

        if self.use_fit_dense:
            self.fit_dense = nn.Linear(config.hidden_size, config.fit_size)
            # self.fit_denses = nn.ModuleList([nn.Linear(config.hidden_size, config.fit_size) for _ in range(config.num_layers + 1)])
        if self.use_lm:
            self.lm_head = models.BertMaskedLMHead(config, self.embeddings.token_embeddings.weight)
        self._init_weights()

    def forward(self, token_ids, segment_ids, position_ids, attn_mask):
        all_attn_outputs, all_ffn_outputs = [], []
        output = self.embeddings(token_ids, segment_ids, position_ids)
        if self.use_fit_dense:
            all_ffn_outputs.append(self.fit_dense(output))
            # all_ffn_outputs.append(self.fit_denses[0](output))
        else:
            all_ffn_outputs.append(output)

        for i, layer in enumerate(self.encoder):
            output, attn_output = layer(output, attn_mask)
            all_attn_outputs.append(attn_output)
            if self.use_fit_dense:
                all_ffn_outputs.append(self.fit_dense(output))
                # all_ffn_outputs.append(self.fit_denses[i + 1](output))
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


class TinyBert(nn.Module):
    def __init__(self, config, task, return_hid=False):
        super(TinyBert, self).__init__()

        self.use_fit_dense = config.use_fit_dense
        self.task = task
        self.return_hid = return_hid
        self.embeddings = models.BertEmbedding(config)
        self.encoder = nn.ModuleList([models.BertTransformerBlock(config) for _ in range(config.num_layers)])

        if self.use_fit_dense:
            self.fit_dense = nn.Linear(config.hidden_size, config.fit_size)
            # self.fit_denses = nn.ModuleList([nn.Linear(config.hidden_size, config.fit_size) for _ in range(config.num_layers + 1)])
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
            # all_ffn_outputs.append(self.fit_denses[0](output))
        else:
            all_ffn_outputs.append(output)

        for i, layer in enumerate(self.encoder):
            output, attn_output = layer(output, attn_mask)
            all_attn_outputs.append(attn_output)
            if self.use_fit_dense:
                all_ffn_outputs.append(self.fit_dense(output))
                # all_ffn_outputs.append(self.fit_denses[i + 1](output))
            else:
                all_ffn_outputs.append(output)

        if self.task in datasets.glue_tasks:
            output = self.cls_pooler(output[:, 0])
            output = self.classifier(output).squeeze(-1)
            if self.return_hid:
                return output, all_attn_outputs, all_ffn_outputs
            return output
        elif self.task in datasets.squad_tasks:
            output = self.classifier(output)
            start_logits, end_logits = output.split(1, dim=-1)
            if self.return_hid:
                return start_logits.squeeze(-1), end_logits.squeeze(-1), all_attn_outputs, all_ffn_outputs
            return start_logits.squeeze(-1), end_logits.squeeze(-1)
        elif self.task in datasets.multi_choice_tasks:
            output = self.cls_pooler(output[:, 0])
            output = self.classifier(output)
            output = output.view(-1, num_choices)
            if self.return_hid:
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
