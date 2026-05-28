import torch.nn as nn
import transformers
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self,pool,model_path,drop_out,nums_labels):
        super().__init__()
        self.pool=pool
        _prev_verbosity = transformers.logging.get_verbosity()#获取当前日志等级
        transformers.logging.set_verbosity_error()#设置日志等级为error
        self.bert=BertModel.from_pretrained(model_path)
        transformers.logging.set_verbosity(_prev_verbosity)#再设置回原来等级，这三步可以实现省略显示warning日志
        hidden_size=self.bert.config.hidden_size

        self.drop_out=nn.Dropout(drop_out)
        self.classifier=nn.Linear(hidden_size,nums_labels)
        
    def forward(self,input_ids,attention_mask,token_type_ids):
        outputs=self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        last_hidden = outputs.last_hidden_state # BERT 最后一层 Transformer 输出的所有 token 向量[B,L,H]
        vec=self._pool(last_hidden,attention_mask) # [B,H]
        vec=self.drop_out(vec)
        logits=self.classifier(vec) # [B,nums_labels]
        return logits

    def _pool(self,last_hidden, # [B,L,H]
              attention_mask # [B,L]
              ):
        if self.pool=="cls":
            return last_hidden[:,0,:]
        
        mask=attention_mask.unsqueeze(-1).float()#[B,L,1]

        if self.pool=="mean":
            sum_hidden=(mask*last_hidden).sum(dim=1)
            count=mask.sum(dim=1).clamp(min=1e9)
            return sum_hidden/count
        
        if self.pool=="max":
            hidden=last_hidden+(1-mask)*-1e9
            return hidden.max(dim=1).values #[B,H]  hidden.max(dim=1)返回的是二元组values  ：最大值本身 indices ：最大值所在的位置下标
        
def build_model(bert_path: str, num_labels: int, pool: str = "cls") -> BertClassifier:
    """工厂函数，统一构建入口，便于 train.py 调用。"""
    model = BertClassifier(bert_path, num_labels=num_labels, pool=pool)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_bert   = sum(p.numel() for p in model.bert.parameters()) / 1e6
    n_head   = sum(p.numel() for p in model.classifier.parameters()) / 1e3
    print(f"模型参数量: {n_params:.1f}M  "
          f"(BERT: {n_bert:.1f}M, 分类头: {n_head:.1f}K)")
    print(f"池化策略: {pool}")
    return model




