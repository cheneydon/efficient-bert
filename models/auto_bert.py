import torch.nn as nn
import datasets
import models


class AutoBertFeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super(AutoBertFeedForwardNetwork, self).__init__()

        self.ffn_hidden_size = config.ffn_hidden_size
        self.dense1 = nn.Linear(config.hidden_size, config.ffn_hidden_size)
        self.dense2 = nn.Linear(config.ffn_hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states, ffn_func, expansion_ratio):
        expansion_hidden_size = int(expansion_ratio * self.ffn_hidden_size)
        wb1 = [self.dense1.weight.t()[:, :expansion_hidden_size], self.dense1.bias[:expansion_hidden_size]]
        wb2 = [self.dense2.weight.t()[:expansion_hidden_size, :], self.dense2.bias]
        try:
            output = self.dropout(ffn_func(hidden_states, wb1, wb2))
        except RuntimeError as e:
            print(e)
            return None
        output = self.layernorm(hidden_states + output)
        return output


class AutoBertTransformerBlock(nn.Module):
    def __init__(self, config):
        super(AutoBertTransformerBlock, self).__init__()

        self.attention = models.BertAttention(config)
        self.all_ffn = nn.ModuleList([AutoBertFeedForwardNetwork(config) for _ in range(config.max_stacked_ffn)])

    def forward(self, hidden_states, attn_mask, ffn_func, num_stacked_ffn, expansion_ratio):
        output, attn_score = self.attention(hidden_states, attn_mask)
        all_ffn_hidden_states = []
        for layer in self.all_ffn[:num_stacked_ffn]:
            output = layer(output, ffn_func, expansion_ratio)
            all_ffn_hidden_states.append(output)
            if output is None:
                return [None] * 2
        return output, attn_score


class AutoBertSingle(nn.Module):
    def __init__(self, config, use_lm=False):
        super(AutoBertSingle, self).__init__()

        self.use_fit_dense = config.use_fit_dense
        self.use_lm = use_lm
        self.expansion_ratio_map = config.expansion_ratio_map
        self.embeddings = models.MobileBertEmbedding(config)
        self.encoder = nn.ModuleList([AutoBertTransformerBlock(config) for _ in range(config.num_layers)])

        if self.use_fit_dense:
            self.fit_dense = nn.Linear(config.hidden_size, config.fit_size)
        if self.use_lm:
            self.lm_head = models.BertMaskedLMHead(config, self.embeddings.token_embeddings.weight)
        self._init_weights()

    def forward(self, token_ids, segment_ids, position_ids, attn_mask, entire_ffn_func, entire_linear_idx):
        all_attn_outputs, all_ffn_outputs = [], []
        output = self.embeddings(token_ids, segment_ids, position_ids)
        if self.use_fit_dense:
            all_ffn_outputs.append(self.fit_dense(output))
        else:
            all_ffn_outputs.append(output)

        for i, layer in enumerate(self.encoder):
            cur_linear_idx = entire_linear_idx[i]
            num_stacked_ffn, expansion_ratio = int(cur_linear_idx.split('_')[0]), self.expansion_ratio_map[cur_linear_idx]
            output, attn_output = layer(
                output, attn_mask, entire_ffn_func[i], num_stacked_ffn, expansion_ratio)
            if output is None:
                return [None] * 3

            all_attn_outputs.append(attn_output)
            if self.use_fit_dense:
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


class AutoBert(nn.Module):
    def __init__(self, config, task, return_hid=False):
        super(AutoBert, self).__init__()

        self.use_fit_dense = config.use_fit_dense
        self.task = task
        self.return_hid = return_hid
        self.expansion_ratio_map = config.expansion_ratio_map
        self.embeddings = models.MobileBertEmbedding(config)
        self.encoder = nn.ModuleList([AutoBertTransformerBlock(config) for _ in range(config.num_layers)])

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

    def forward(self, token_ids, segment_ids, position_ids, attn_mask, entire_ffn_func, entire_linear_idx):
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

        for i, layer in enumerate(self.encoder):
            cur_linear_idx = entire_linear_idx[i]
            num_stacked_ffn, expansion_ratio = int(cur_linear_idx.split('_')[0]), self.expansion_ratio_map[cur_linear_idx]
            output, attn_output = layer(
                output, attn_mask, entire_ffn_func[i], num_stacked_ffn, expansion_ratio)
            if output is None:
                return [None] * 3

            all_attn_outputs.append(attn_output)
            if self.use_fit_dense:
                all_ffn_outputs.append(self.fit_dense(output))
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


class MultiTaskBert(nn.Module):
    def __init__(self, config, return_hid=False):
        super(MultiTaskBert, self).__init__()

        self.return_hid = return_hid
        self.embeddings = models.BertEmbedding(config)
        self.encoder = nn.ModuleList([models.BertTransformerBlock(config) for _ in range(config.num_layers)])

        self.cls_pooler = models.BertClsPooler(config)
        self.classifiers = nn.ModuleList([])
        for task in datasets.glue_train_tasks:
            num_classes = datasets.glue_num_classes[task]
            self.classifiers.append(nn.Linear(config.hidden_size, num_classes))
        self._init_weights()

    def forward(self, task_id, token_ids, segment_ids, position_ids, attn_mask):
        all_attn_outputs, all_ffn_outputs = [], []
        output = self.embeddings(token_ids, segment_ids, position_ids)
        all_ffn_outputs.append(output)
        for layer in self.encoder:
            output, attn_output = layer(output, attn_mask)
            all_attn_outputs.append(attn_output)
            all_ffn_outputs.append(output)

        output = self.cls_pooler(output[:, 0])
        output = self.classifiers[task_id](output).squeeze(-1)
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


class MultiTaskAutoBert(nn.Module):
    def __init__(self, config, task, return_hid=False):
        super(MultiTaskAutoBert, self).__init__()

        self.use_fit_dense = config.use_fit_dense
        self.task = task
        self.return_hid = return_hid
        self.expansion_ratio_map = config.expansion_ratio_map
        self.embeddings = models.MobileBertEmbedding(config)
        self.encoder = nn.ModuleList([AutoBertTransformerBlock(config) for _ in range(config.num_layers)])

        if self.use_fit_dense:
            self.fit_dense = nn.Linear(config.hidden_size, config.fit_size)

        self.cls_pooler = models.BertClsPooler(config)
        self.classifiers = nn.ModuleList([])
        for task in datasets.glue_train_tasks:
            num_classes = datasets.glue_num_classes[task]
            self.classifiers.append(nn.Linear(config.hidden_size, num_classes))
        self._init_weights()

    def forward(self, task_id, token_ids, segment_ids, position_ids, attn_mask, entire_ffn_func, entire_linear_idx):
        all_attn_outputs, all_ffn_outputs = [], []
        output = self.embeddings(token_ids, segment_ids, position_ids)
        if self.use_fit_dense:
            all_ffn_outputs.append(self.fit_dense(output))
        else:
            all_ffn_outputs.append(output)

        for i, layer in enumerate(self.encoder):
            cur_linear_idx = entire_linear_idx[i]
            num_stacked_ffn, expansion_ratio = int(cur_linear_idx.split('_')[0]), self.expansion_ratio_map[cur_linear_idx]
            output, attn_output = layer(
                output, attn_mask, entire_ffn_func[i], num_stacked_ffn, expansion_ratio)
            if output is None:
                return [None] * 3

            all_attn_outputs.append(attn_output)
            if self.use_fit_dense:
                all_ffn_outputs.append(self.fit_dense(output))
            else:
                all_ffn_outputs.append(output)

        output = self.cls_pooler(output[:, 0])
        output = self.classifiers[task_id](output).squeeze(-1)
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


class AutoTinyBertSingle(nn.Module):
    def __init__(self, config, use_lm=False):
        super(AutoTinyBertSingle, self).__init__()

        self.use_fit_dense = config.use_fit_dense
        self.use_lm = use_lm
        self.expansion_ratio_map = config.expansion_ratio_map
        self.embeddings = models.BertEmbedding(config)
        self.encoder = nn.ModuleList([AutoBertTransformerBlock(config) for _ in range(config.num_layers)])

        if self.use_fit_dense:
            self.fit_dense = nn.Linear(config.hidden_size, config.fit_size)
        if self.use_lm:
            self.lm_head = models.BertMaskedLMHead(config, self.embeddings.token_embeddings.weight)
        self._init_weights()

    def forward(self, token_ids, segment_ids, position_ids, attn_mask, entire_ffn_func, entire_linear_idx):
        all_attn_outputs, all_ffn_outputs = [], []
        output = self.embeddings(token_ids, segment_ids, position_ids)
        if self.use_fit_dense:
            all_ffn_outputs.append(self.fit_dense(output))
        else:
            all_ffn_outputs.append(output)

        for i, layer in enumerate(self.encoder):
            cur_linear_idx = entire_linear_idx[i]
            num_stacked_ffn, expansion_ratio = int(cur_linear_idx.split('_')[0]), self.expansion_ratio_map[cur_linear_idx]
            output, attn_output = layer(
                output, attn_mask, entire_ffn_func[i], num_stacked_ffn, expansion_ratio)
            if output is None:
                return [None] * 3

            all_attn_outputs.append(attn_output)
            if self.use_fit_dense:
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


class AutoTinyBert(nn.Module):
    def __init__(self, config, task, return_hid=False):
        super(AutoTinyBert, self).__init__()

        self.use_fit_dense = config.use_fit_dense
        self.task = task
        self.return_hid = return_hid
        self.expansion_ratio_map = config.expansion_ratio_map
        self.embeddings = models.BertEmbedding(config)
        self.encoder = nn.ModuleList([AutoBertTransformerBlock(config) for _ in range(config.num_layers)])

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

    def forward(self, token_ids, segment_ids, position_ids, attn_mask, entire_ffn_func, entire_linear_idx):
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

        for i, layer in enumerate(self.encoder):
            cur_linear_idx = entire_linear_idx[i]
            num_stacked_ffn, expansion_ratio = int(cur_linear_idx.split('_')[0]), self.expansion_ratio_map[cur_linear_idx]
            output, attn_output = layer(
                output, attn_mask, entire_ffn_func[i], num_stacked_ffn, expansion_ratio)
            if output is None:
                return [None] * 3

            all_attn_outputs.append(attn_output)
            if self.use_fit_dense:
                all_ffn_outputs.append(self.fit_dense(output))
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
