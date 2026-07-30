from typing import Dict, List

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from torchcrf import CRF


class BertForNER(BertPreTrainedModel):
    """BERT 模型用于命名实体识别，支持线性分类头和 CRF 层"""
    
    def __init__(self, config, use_crf=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.use_crf = use_crf
        
        # BERT 模型
        self.bert = BertModel(config)
        
        # Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 分类器
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        
        # CRF 层（可选）
        if use_crf:
            self.crf = CRF(self.num_labels, batch_first=True)
        
        # 初始化权重
        self.init_weights()
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):
        # 获取 BERT 输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            # 创建 mask，忽略 -100 的位置
            mask = (labels != -100).long()
            
            if self.use_crf:
                # CRF 损失
                loss = -self.crf(
                    logits,
                    labels,
                    mask=mask.bool(),
                    reduction="mean",
                )
            else:
                # 交叉熵损失
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    logits.view(-1, self.num_labels),
                    labels.view(-1),
                )
        
        return {
            "loss": loss,
            "logits": logits,
        }
    
    def predict(self, input_ids, attention_mask=None, token_type_ids=None):
        """预测标签序列"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        if self.use_crf:
            # 使用 CRF 解码
            mask = attention_mask.bool()
            best_paths = self.crf.decode(logits, mask=mask)
            return best_paths
        else:
            # 使用 argmax 解码
            predictions = logits.argmax(dim=-1)
            return predictions


def build_model(
    use_crf: bool = False,
    bert_path: str = "bert-base-chinese",
    num_labels: int = 9,
    dropout: float = 0.1,
):
    """构建 BERT NER 模型"""
    from transformers import BertConfig
    
    config = BertConfig.from_pretrained(bert_path)
    config.num_labels = num_labels
    config.hidden_dropout_prob = dropout
    
    model = BertForNER.from_pretrained(
        bert_path,
        config=config,
        use_crf=use_crf,
    )
    
    return model
