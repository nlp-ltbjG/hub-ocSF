import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF


class BertNerModel(nn.Module):
    def __init__(self, bert_path, num_labels, use_crf=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.use_crf = use_crf
        self.num_labels = num_labels
        if use_crf:
            self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        seq_out = self.dropout(out.last_hidden_state)
        logits = self.fc(seq_out)

        loss = None
        if labels is not None:
            if self.use_crf:
                crf_labels = labels.clone()
                crf_labels[crf_labels == -100] = 0
                mask = attention_mask.bool()
                loss = -self.crf(logits, crf_labels, mask=mask, reduction="mean")
            else:
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return logits, loss

    def predict(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        seq_out = self.dropout(out.last_hidden_state)
        logits = self.fc(seq_out)

        if self.use_crf:
            mask = attention_mask.bool()
            return self.crf.decode(logits, mask=mask)
        else:
            return logits.argmax(dim=-1).tolist()
