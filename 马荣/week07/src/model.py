import torch
import torch.nn as nn
import transformers
from transformers import BertModel
from TorchCRF import CRF


def _load_bert(bert_path: str) -> BertModel:
    prev = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    bert = BertModel.from_pretrained(bert_path)
    transformers.logging.set_verbosity(prev)
    return bert


class BertNER(nn.Module):
    def __init__(self, bert_path: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = _load_bert(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        seq_output = outputs.last_hidden_state  # [B, L, H]
        logits = self.classifier(self.dropout(seq_output))  # [B, L, num_labels]

        return logits


class BertCRFNER(nn.Module):
    def __init__(self, bert_path, dropout, num_labels):
        super().__init__()
        self.bert = _load_bert(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels
        self.crf = CRF(num_labels)

    def _get_emissions(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        seq_output = outputs.last_hidden_state
        return self.classifier(self.dropout(seq_output))  # (B, L, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        emissions = self._get_emissions(input_ids, attention_mask, token_type_ids)

        loss = None
        if labels is not None:
            # TorchCRF starts from position 0, so remove CLS before loss.
            emissions_crf = emissions[:, 1:, :]
            labels_crf = labels[:, 1:].clone()

            # labels == -100 covers SEP/PAD from dataset alignment; attention_mask
            # also keeps padded tokens out if a different label filler is used.
            mask = attention_mask[:, 1:].bool() & labels_crf.ne(-100)

            # CRF still indexes labels at masked positions, so use any legal id.
            labels_crf[labels_crf == -100] = 0

            # TorchCRF returns per-sample log likelihood.
            loss = -self.crf(emissions_crf, labels_crf, mask=mask).mean()

        return emissions, loss

    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> list[list[int]]:
        emissions = self._get_emissions(input_ids, attention_mask, token_type_ids)

        # Remove CLS. Valid decoded length is non-pad length minus CLS and SEP.
        emissions_crf = emissions[:, 1:, :]
        valid_lengths = attention_mask.long().sum(dim=1) - 2  #每条样本的真实文本长度
        positions = torch.arange(emissions_crf.size(1), device=attention_mask.device)#
        mask = positions.unsqueeze(0) < valid_lengths.unsqueeze(1)#发生广播，对应位置比较大小，生成True,False  mask

        return self.crf.viterbi_decode(emissions_crf, mask)


class BertCRFNEROriginal(BertCRFNER):
    """这里将CLS和SEP都当成标签0来训练"""
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        emissions = self._get_emissions(input_ids, attention_mask, token_type_ids)
        mask = attention_mask.bool()

        loss = None
        if labels is not None:
            labels_crf = labels.clone()
            labels_crf[labels_crf == -100] = 0  #[labels_crf == -100]会生成一个bool矩阵
            loss = -self.crf(emissions, labels_crf, mask=mask).mean()

        return emissions, loss
    
    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> list[list[int]]:
        """Viterbi 解码，返回 list[list[int]]，每条序列长度等于实际token数（不含PAD）。"""
        emissions = self._get_emissions(input_ids, attention_mask, token_type_ids)
        mask = attention_mask.bool()
        return self.crf.viterbi_decode(emissions, mask)


def build_model(
    use_crf: bool,
    use_crf_cls:bool,
    bert_path: str,
    num_labels: int,
    dropout: float = 0.1,
) -> nn.Module:
    """模型工厂函数。"""
    model_cls = BertCRFNER if use_crf else BertNER if use_crf_cls else BertCRFNEROriginal
    model = model_cls(bert_path=bert_path, num_labels=num_labels, dropout=dropout)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_name = "BERT + CRF" if use_crf else "BERT + Linear"
    print(f"模型：{model_name}")
    print(f"  标签数：{num_labels}")
    print(f"  参数总量：{total_params / 1e6:.1f}M")
    print(f"  可训练参数：{trainable_params / 1e6:.1f}M")
    return model
