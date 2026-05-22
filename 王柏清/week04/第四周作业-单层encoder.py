import torch
import torch.nn as nn
import math
from transformers import BertModel

# ---------------------- 1.多头自注意力 ----------------------
class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, v)

        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size()[:-2] + (self.all_head_size,))
        return context

# ---------------------- 2.注意力输出层 ----------------------
class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# ---------------------- 3.前馈中间层 ----------------------
class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

# ---------------------- 4.前馈输出层 ----------------------
class BertOutput(nn.Module):
    def __init__(self, intermediate_size, hidden_size):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# ---------------------- 5.单层Encoder ----------------------
class BertLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size):
        super().__init__()
        self.attention = BertSelfAttention(hidden_size, num_attention_heads)
        self.attention_output = BertSelfOutput(hidden_size)
        self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        self.output = BertOutput(intermediate_size, hidden_size)

    def forward(self, hidden_states):
        attn_out = self.attention(hidden_states)
        attn_out = self.attention_output(attn_out, hidden_states)
        mid_out = self.intermediate(attn_out)
        layer_out = self.output(mid_out, attn_out)
        return layer_out

# ---------------------- 6.完整DiyBert（仅embedding+单层Encoder） ----------------------
class DiyBert(nn.Module):
    def __init__(self, vocab_size=21128, hidden_size=768,
                 num_attention_heads=12, intermediate_size=3072,
                 max_position_embeddings=512, type_vocab_size=2):
        super().__init__()
        # embedding层
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

        # 单层encoder
        self.encoder_layer = BertLayer(hidden_size, num_attention_heads, intermediate_size)
        # pooler层
        self.pooler = nn.Linear(hidden_size, hidden_size)

    def get_embedding_output(self, input_ids, token_type_ids=None):
        """单独提取embedding层输出，用于公平对比"""
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        we = self.word_embeddings(input_ids)
        pe = self.position_embeddings(position_ids)
        te = self.token_type_embeddings(token_type_ids)
        embeddings = we + pe + te
        embeddings = self.LayerNorm(embeddings)
        return embeddings

    def forward(self, input_ids, token_type_ids=None):
        embeddings = self.get_embedding_output(input_ids, token_type_ids)
        sequence_output = self.encoder_layer(embeddings)

        # pooler层
        pooled_output = sequence_output[:, 0]
        pooled_output = self.pooler(pooled_output)
        pooled_output = torch.tanh(pooled_output)
        
        return sequence_output, pooled_output


# ====================== 对比单层Encoder输出 ======================
if __name__ == "__main__":
    # 1.加载官方bert，开启隐藏状态输出，拿到中间层结果
    bert = BertModel.from_pretrained(
        r"D:\八斗大模型\往期\第六周\bert-base-chinese",
        return_dict=False,
        output_hidden_states=True  # 开启中间层输出
    )
    bert.eval()

    # 2.自己搭建的单层bert
    diy_bert = DiyBert()
    diy_bert.eval()

    # 3.权重赋值
    with torch.no_grad():
        # embedding权重
        diy_bert.word_embeddings.weight.copy_(bert.embeddings.word_embeddings.weight)
        diy_bert.position_embeddings.weight.copy_(bert.embeddings.position_embeddings.weight)
        diy_bert.token_type_embeddings.weight.copy_(bert.embeddings.token_type_embeddings.weight)
        diy_bert.LayerNorm.weight.copy_(bert.embeddings.LayerNorm.weight)
        diy_bert.LayerNorm.bias.copy_(bert.embeddings.LayerNorm.bias)

        # 只拷贝bert第0层（第一层）encoder权重
        layer0 = bert.encoder.layer[0]
        diy_bert.encoder_layer.attention.query.weight.copy_(layer0.attention.self.query.weight)
        diy_bert.encoder_layer.attention.query.bias.copy_(layer0.attention.self.query.bias)
        diy_bert.encoder_layer.attention.key.weight.copy_(layer0.attention.self.key.weight)
        diy_bert.encoder_layer.attention.key.bias.copy_(layer0.attention.self.key.bias)
        diy_bert.encoder_layer.attention.value.weight.copy_(layer0.attention.self.value.weight)
        diy_bert.encoder_layer.attention.value.bias.copy_(layer0.attention.self.value.bias)

        diy_bert.encoder_layer.attention_output.dense.weight.copy_(layer0.attention.output.dense.weight)
        diy_bert.encoder_layer.attention_output.dense.bias.copy_(layer0.attention.output.dense.bias)
        diy_bert.encoder_layer.attention_output.LayerNorm.weight.copy_(layer0.attention.output.LayerNorm.weight)
        diy_bert.encoder_layer.attention_output.LayerNorm.bias.copy_(layer0.attention.output.LayerNorm.bias)

        diy_bert.encoder_layer.intermediate.dense.weight.copy_(layer0.intermediate.dense.weight)
        diy_bert.encoder_layer.intermediate.dense.bias.copy_(layer0.intermediate.dense.bias)

        diy_bert.encoder_layer.output.dense.weight.copy_(layer0.output.dense.weight)
        diy_bert.encoder_layer.output.dense.bias.copy_(layer0.output.dense.bias)
        diy_bert.encoder_layer.output.LayerNorm.weight.copy_(layer0.output.LayerNorm.weight)
        diy_bert.encoder_layer.output.LayerNorm.bias.copy_(layer0.output.LayerNorm.bias)

        # pooler权重
        diy_bert.pooler.weight.copy_(bert.pooler.dense.weight)
        diy_bert.pooler.bias.copy_(bert.pooler.dense.bias)

    # 4.测试输入
    x = [2450, 15486, 102, 2110]
    input_ids = torch.LongTensor([x])

    # 5.相同输入，相同层数，相同权重
    with torch.no_grad():
        # ① 自建模型：embedding → 单层encoder输出
        diy_emb = diy_bert.get_embedding_output(input_ids)
        diy_seq = diy_bert.encoder_layer(diy_emb)

        # ② 官方模型：先拿到embedding输出，再拿到第0层encoder的输出
        bert_all_hidden = bert(input_ids)[2]  # hidden_states是所有层的输出，长度13（embedding+12层）
        bert_emb = bert_all_hidden[0]  # 第0个是embedding输出
        bert_layer0_seq = bert_all_hidden[1]  # 第1个是第0层encoder的输出

    # 6.输出对比
    print("自建单层Encoder输出 shape:", diy_seq.shape)
    print("官方BERT第0层Encoder输出 shape:", bert_layer0_seq.shape)
    print("-" * 60)
    print("自建模型CLS向量前10维：\n", diy_seq[0,0,:10])
    print("官方BERT第0层CLS前10维：\n", bert_layer0_seq[0,0,:10])
    print("-" * 60)
    print("单层Encoder输出是否近似相等：", torch.allclose(diy_seq, bert_layer0_seq, atol=1e-5))
