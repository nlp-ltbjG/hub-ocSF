import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ─── 自定义编码器层（兼容官方 nn.TransformerEncoder 的调用签名）───
class MyTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # self.feed_forward = nn.Sequential(
        #     nn.Linear(d_model, dim_feedforward),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(dim_feedforward, d_model)
        # )
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        attn_out, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                     key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)
        #  Feed-forward 子层
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(ff_out)
        src = self.norm2(src)
        return src

# ─── 对比函数 ─────────────────────────────────────────
def compare_tensors(tensor_a, tensor_b, name="Tensor", atol=1e-5, rtol=1e-3):
    """比较两个张量的形状和数值是否接近"""
    a = tensor_a.detach().numpy() if isinstance(tensor_a, torch.Tensor) else tensor_a
    b = tensor_b.detach().numpy() if isinstance(tensor_b, torch.Tensor) else tensor_b
    if a.shape != b.shape:
        print(f"❌ {name} shape mismatch: {a.shape} vs {b.shape}")
        return False, None
    max_abs_diff = np.max(np.abs(a - b))
    is_close = np.allclose(a, b, atol=atol, rtol=rtol)
    if is_close:
        print(f"✅ {name} passed. Max diff: {max_abs_diff}")
    else:
        print(f"❌ {name} failed. Max diff: {max_abs_diff}")
    return is_close, max_abs_diff

# ─── 验证函数 ─────────────────────────────────────────
def validate_encoder(num_layers=2, d_model=8, nhead=2, dim_feedforward=32, batch_size=2, seq_len=5):
    torch.manual_seed(42)
    src = torch.rand(batch_size, seq_len, d_model)

    # 官方单层 + 编码器
    official_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                batch_first=True, dropout=0.0)
    official_encoder = nn.TransformerEncoder(official_layer, num_layers).eval()

    # 自定义单层（先创建，再拷贝官方参数）
    diy_layer = MyTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=0.0)
    diy_layer.load_state_dict(official_layer.state_dict())   # 确保权重完全相同
    diy_encoder = nn.TransformerEncoder(diy_layer, num_layers).eval()

    # 逐层对比（此时权重相同，差异应接近浮点误差）
    output_official = src
    output_diy = src
    for i in range(num_layers):
        with torch.no_grad():
            output_official = official_encoder.layers[i](output_official)
            output_diy = diy_encoder.layers[i](output_diy)
            print(f"\n--- 对比 Layer {i+1} ---")
            compare_tensors(output_diy, output_official, f"Layer {i+1} Output")

    # 整体编码器输出对比
    final_official = official_encoder(src)
    final_diy = diy_encoder(src)
    compare_tensors(final_diy, final_official, "Final Encoder Output")
if __name__ == "__main__":
    validate_encoder()