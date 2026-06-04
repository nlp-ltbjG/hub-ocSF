import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torchcrf import CRF
import re

class OptimizedBertCRFNER(nn.Module):
    def __init__(self, bert_path: str, num_labels: int, lstm_hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        # 融合最后4层输出，因此输入维度为 hidden_size * 4
        bert_hidden_size = self.bert.config.hidden_size
        
        # BiLSTM 增强位置与时序信息
        self.lstm = nn.LSTM(
            input_size=bert_hidden_size * 4, 
            hidden_size=lstm_hidden_dim // 2, 
            bidirectional=True, 
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden_dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels
        
        # 定义常见的机构后缀用于后处理边界修正
        self.org_suffix_pattern = re.compile(r'(公司|集团|局|院|所|中心|大学)$')

    def _get_bert_multilayer_features(self, input_ids, attention_mask, token_type_ids):
        """提取BERT最后四层隐藏状态并拼接"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        # 取最后4层 [B, L, H] -> concat -> [B, L, 4*H]
        last_four_layers = torch.cat(outputs.hidden_states[-4:], dim=-1)
        return last_four_layers

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, use_amp=False):
        # 1. 提取多层BERT特征
        bert_out = self._get_bert_multilayer_features(input_ids, attention_mask, token_type_ids)
        
        # 2. BiLSTM 编码
        lstm_out, _ = self.lstm(bert_out)
        emissions = self.classifier(self.dropout(lstm_out))
        
        mask = attention_mask.bool()
        loss = None
        
        if labels is not None:
            labels_crf = labels.clone()
            labels_crf[labels_crf == -100] = 0
            
            # 混合精度训练 (AMP)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = -self.crf(emissions, labels_crf, mask=mask, reduction="mean")
                
        return emissions, loss

    @torch.no_grad()
    def decode(self, input_ids, attention_mask, token_type_ids, use_amp=False):
        """带 AMP 支持的 Viterbi 解码"""
        with torch.cuda.amp.autocast(enabled=use_amp):
            emissions, _ = self.forward(input_ids, attention_mask, token_type_ids)
        mask = attention_mask.bool()
        best_paths = self.crf.decode(emissions, mask=mask)
        return best_paths

    def post_process_entities(self, text, predicted_tags, id2tag):
        """基于规则的后处理边界修正 (以 ORG 为例)"""
        entities = []
        current_entity = ""
        for char, tag_id in zip(text, predicted_tags):
            tag = id2tag[tag_id]
            if tag.startswith('B-') or tag.startswith('S-'):
                if current_entity: entities.append(current_entity)
                current_entity = char
            elif tag.startswith('I-') or tag.startswith('E-'):
                current_entity += char
            else: # O 标签
                if current_entity: entities.append(current_entity)
                current_entity = ""
        if current_entity: entities.append(current_entity)
        
        # 规则修正：如果实体结尾不是常见后缀，尝试向后匹配文本中的后缀
        refined_entities = []
        for ent in entities:
            match = self.org_suffix_pattern.search(ent)
            if not match and ent in text:
                idx = text.find(ent) + len(ent)
                suffix_match = self.org_suffix_pattern.match(text[idx:])
                if suffix_match:
                    ent += suffix_match.group(0)
            refined_entities.append(ent)
        return refined_entities

    def export_onnx(self, dummy_input, save_path="optimized_ner.onnx"):
        """导出 ONNX 模型用于 TensorRT/ONNX Runtime 加速"""
        self.eval()
        torch.onnx.export(
            self, 
            dummy_input, 
            save_path, 
            opset_version=14,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["emissions"]
        )
        print(f"✅ ONNX 模型已成功导出至: {save_path}")
