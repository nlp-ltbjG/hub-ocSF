# DeepSeek-V4-Pro / MiniMax-M3 / GLM-5.2 架构对比

> 本文档基于 `model_code/` 目录下的配置与建模文件（`deepseek-ai_DeepSeek-V4-Pro_config.json`、`deepseek_v4_modeling.py`、`minimax_m3.json`、`minimax_m3_vl_modeling.py`、`glm52_readme.md`）整理。DeepSeek-V4-Pro 与 MiniMax-M3 的数值来自代码，GLM-5.2 的细节来自其 README（未公开完整 modeling 代码，部分架构项标注为 README 声明）。

## 一、总体定位

| 维度 | DeepSeek-V4-Pro | MiniMax-M3 (VL) | GLM-5.2 |
|---|---|---|---|
| 模态 | 纯文本 | 多模态（文+图+视频，VL） | 纯文本 |
| 上下文长度 | 1M tokens（`max_position_embeddings=1048576`） | 1M tokens（`1048576`） | 1M tokens（solid 1M） |
| 架构类 | `DeepseekV4ForCausalLM` | `MiniMaxM3SparseForConditionalGeneration` | GLM-MoE-DSA（transformers 文档名 `glm_moe_dsa`） |
| 许可证 | Apache 2.0（HF 建模文件） | Apache 2.0 | MIT（Pure Open） |
| 发布形态 | FP8 量化权重 + BF16 建模 | BF16，多模态投影器 | MIT 开源，支持 SGLang/vLLM/Transformers/KTransformers/Unsloth |

## 二、核心规模参数

| 参数 | DeepSeek-V4-Pro | MiniMax-M3 |
|---|---|---|
| hidden_size | 7168 | 6144 |
| num_hidden_layers | 61 | 60 |
| num_attention_heads | 128 | 64 |
| head_dim | **512** | 128 |
| num_key_value_heads | **1（Shared-KV MQA）** | 4（GQA） |
| vocab_size | 129280 | 200064 |
| rope_theta（主） | 10000 | 5,000,000 |
| torch_dtype | bfloat16 | bfloat16 |

## 三、注意力机制

### 3.1 注意力头与 KV 共享

| 项 | DeepSeek-V4-Pro | MiniMax-M3 |
|---|---|---|
| KV 头数 | 1（MQA，K 与 V 共享同一张量，`kv_proj` 只投影一次） | 4（GQA，独立 q/k/v_proj） |
| QK-Norm | 自有 `DeepseekV4RMSNorm`（带 weight）+ `DeepseekV4UnweightedRMSNorm`（q_b_norm） | Gemma 风格 `MiniMaxM3VLRMSNorm`（weight+1），`use_qk_norm=true`，`qk_norm_type=per_head` |
| Attention Sink | 每头可学习 sink 参数（`self.sinks`），eager 路径将 sink 拼到 logits 做 softmax 后丢弃 | 无独立 sink 参数 |
| 输出投影 | 分组低秩：`o_groups=16, o_lora_rank=1024`（`o_a_proj` 分组降维 → `o_b_proj` 混合），降低 `heads*head_dim→hidden` 的巨大开销 | 标准 `o_proj` 线性层 |
| Q 低秩 | `q_a_proj→q_a_norm→q_b_proj`，`q_lora_rank=1536` | 无 Q 低秩，直接 `q_proj` |

### 3.2 RoPE

| 项 | DeepSeek-V4-Pro | MiniMax-M3 |
|---|---|---|
| RoPE 类型 | 双 rope 体系：`main`（θ=10000，滑动层用）+ `compress`（θ=160000，CSA/HCA 压缩层用），yarn 缩放（factor=16, original_max=65536, beta_fast=32, beta_slow=1） | 单一 default RoPE，θ=5,000,000，`partial_rotary_factor=0.5`，`rotary_dim=64` |
| 旋转方式 | **交错式（interleaved）** RoPE，成对通道，`repeat_interleave(2)`，对输出 rope 切片施加共轭旋转（-sin）以抵消 V 上的 rope | 标准（非交错），`torch.cat([freqs,freqs])` 复制，前半旋转后半直通 |
| 部分旋转 | `rope_head_dim=64`（head_dim=512 中仅尾部 64 维上 rope） | `rotary_dim=64`（head_dim=128 中前 64 维旋转） |

### 3.3 稀疏注意力（长上下文核心）

DeepSeek-V4-Pro 采用**三态注意力 + KV 压缩**；MiniMax-M3 采用**块稀疏索引**；GLM-5.2 采用 IndexShare。

| 项 | DeepSeek-V4-Pro | MiniMax-M3 | GLM-5.2 |
|---|---|---|---|
| 层类型 | `sliding_attention`（滑动窗 128）/ `compressed_sparse_attention`（CSA）/ `heavily_compressed_attention`（HCA），按 `compress_ratios` 交错排布：`[128,128,4,128,4,...,0]` | 前 3 层 dense，其余 `minimax_m3_sparse`（`moe_layer_freq`/`sparse_attention_freq` 控制） | DSA（Dilated Sparse Attention），IndexShare 每 4 层稀疏层复用同一 indexer |
| 压缩率 | CSA：每 4 token 压一个；HCA：每 128 token 压一个 | 不做 KV 压缩，直接在原始 key 上分块 | 复用 indexer 降低 per-token FLOPs 2.9×（1M ctx） |
| Indexer | Lightning Indexer：在压缩 KV 上用 `∑_h w·ReLU(q·K)` 打分，每 query 取 top-`index_topk=1024` 个压缩块；`index_n_heads=64, index_head_dim=128` | Lightning Indexer：在原始 key 上分块（`index_block_size=128`），每块 max-pool，每 query 取 top-`index_topk_blocks=16` + `index_local_blocks` 个局部块；`index_n_heads=4` | 每 4 稀疏层共享一个 indexer |
| 选择粒度 | 压缩条目级（每条代表 4 或 128 token） | key 块级（每块 128 key） | 块级 |
| CSA 重叠 | 双序列 Ca/Cb，相邻窗共享前一窗 Ca，窗宽 2×stride | 无双序列重叠 | — |
| 因果性处理 | indexer 对未来压缩块置 -inf，无效索引置 -1 | `token_future` 掩码 + topk 后 `-1` 右填充 | — |
| 缓存层 | `DeepseekV4HCACache`/`DeepseekV4CSACache`（含 compressor buffer、overlap 状态，不可回滚，`_is_stateful=True`） | `MiniMaxM3VLSparseCacheLayer`/`SparseStaticCacheLayer`（额外维护 `idx_keys` 历史） | — |

## 四、MoE / FFN

| 项 | DeepSeek-V4-Pro | MiniMax-M3 |
|---|---|---|
| 路由专家数 | 384（`n_routed_experts`） | 128（`num_local_experts`） |
| 共享专家 | 1（`n_shared_experts`） | 1（`n_shared_experts`） |
| 每 token 激活专家 | 6（`num_experts_per_tok`） | 4（`num_experts_per_tok`） |
| 专家中间维 | 3072（`moe_intermediate_size`） | 3072（`intermediate_size`） |
| dense 层中间维 | —（仅 MoE） | 12288（`dense_intermediate_size`，前 3 层 dense MLP 用） |
| 共享专家中间维 | 同 MoE | 3072（`shared_intermediate_size`） |
| 路由打分函数 | `sqrtsoftplus`（`scoring_func`） | `sigmoid`（`scoring_func`） |
| 路由方法 | `noaux_tc`（带偏置 topk + 归一化）+ **HashRouter**（前若干层用冻结 `tid2eid[input_ids]` 表选专家） | sigmoid + `use_routing_bias`，`e_score_correction_bias` |
| 路由缩放 | `routed_scaling_factor=2.5` | `routed_scaling_factor=2.0` |
| 专家权重精度 | `expert_dtype=fp4` | bf16（无量化配置） |
| 激活函数 | `silu`（`hidden_act`），`swiglu_limit=10.0` clamp | `swigluoai`（自定义 SwiGLU，`swiglu_alpha=1.702, swiglu_limit=7.0`，`gate*sigmoid(gate*alpha)`，`(up+1)*glu`） |
| 整体量化 | FP8（`e4m3`, 动态 activation, `ue8m0` scale, block 128×128） | 无 |

## 五、残差连接 / 模块组织

| 项 | DeepSeek-V4-Pro | MiniMax-M3 |
|---|---|---|
| 残差结构 | **Manifold-Constrained Hyper-Connections (mHC)**：`hc_mult=4` 路并行残差流 `[B,S,4,D]`，每层 2 个 `DeepseekV4HyperConnection`（attn 位 + mlp 位），`comb` 矩阵经 Sinkhorn-Knopp（`hc_sinkhorn_iters=20`）投影到双随机流形，`pre` 塌缩流、`post` 放置子层输出 | 标准残差：`residual + sublayer(norm(x))` |
| 最终聚合 | `DeepseekV4HyperHead` 将 4 路流塌缩回单序列后过 norm | 直接 `norm(hidden_states)` |
| 归一化 | `DeepseekV4RMSNorm`（标准 RMSNorm，weight 乘） | `MiniMaxM3VLRMSNorm`（Gemma 风格，`(x*(1+w))`，`use_gemma_norm=true`） |
| MTP（多 token 预测） | `num_nextn_predict_layers=1`（权重 `mtp.*` 在加载时被忽略，仅推理投机解码） | `num_mtp_modules=7, num_nextn_predict_layers=1`（同样 `mtp.*` 忽略） |
| GLM-5.2 MTP | 改进的 MTP 层用于投机解码，acceptance length 最多 +20% |

## 六、多模态（仅 MiniMax-M3）

MiniMax-M3 为 VL 模型，DeepSeek-V4-Pro 与 GLM-5.2（本仓库内）为纯文本。

- 视觉编码器：CLIP 风格 `clip_vision_model`，32 层，`hidden_size=1280`，`num_attention_heads=16`，`image_size=2016`，`patch_size=14`，`intermediate_size=5120`，`projection_dim=6144`。
- 位置编码：视觉侧 **3D RoPE**（`rope_mode=3d`，`rope_theta=10000`）。
- 图像 token 压缩：`patch_merge`，`spatial_merge_size=2`，`temporal_patch_size=2`；`image_seq_length=576`，支持 `dynamic_res` 与多分辨率 `image_grid_pinpoints`。
- 视频：`vision_segment_max_frames=4`，`video_token_index` 与 `image_token_index` 分离。
- 投影器：`multimodal_projector_bias=true`，`projector_hidden_size=6144`，`projector_hidden_act=gelu`，`vision_feature_select_strategy=full`。

## 七、训练/推理工程细节

| 项 | DeepSeek-V4-Pro | MiniMax-M3 |
|---|---|---|
| 注意力后端 | **仅 eager**（FA2/3 与 SDPA/Flex 均关）：head_dim=512 超 FA 上限 256；SDPA 不带 sink；Flex 的 BlockMask 无法匹配压缩器运行时拼接的 KV 长度 | 支持 SDPA（`_supports_sdpa=true`），Flash/Flex 关闭，兼容 `MiniMaxAI/msa`，`_can_compile_fullgraph=true` |
| fullgraph 编译 | 关闭（压缩器缓存不兼容 StaticCache） | 开启 |
| FP32 保留模块 | `attn_hc/ffn_hc/hc_head/sinks/position_bias/q_a_norm/kv_norm` 等（strict）；压缩器/indexer 的 kv_proj/gate_proj 在 BF16 非严格保留 | 无显式 FP32 保留列表 |
| 有状态性 | `_is_stateful=true`（压缩器 rolling-window 不可回滚，禁用 assisted/prompt-lookup/contrastive 生成） | — |

## 八、GLM-5.2 差异点（README 声明）

- **IndexShare**：每 4 层稀疏注意力复用同一 indexer，1M 上下文下 per-token FLOPs 降低 2.9×（DeepSeek-V4 的 indexer 每层独立、与压缩器耦合；MiniMax 的 indexer 每稀疏层独立且作用在原始 key 上）。
- **MTP 改进**：投机解码 acceptance length 最多 +20%。
- **1M solid context**：强调稳定维持长程任务。
- 提供灵活 thinking effort 级别（编码场景性能/延迟权衡）。
- 旗舰模型，对比对象含 DeepSeek-V4-Pro、MiniMax M3（见 README benchmark 表）。

## 九、关键架构差异速览

1. **KV 共享 vs GQA**：DeepSeek-V4-Pro 用 Shared-KV MQA（1 KV 头，K=V 同张量）+ 分组低秩输出投影；MiniMax-M3 用 GQA（4 KV 头）+ per-head QK-norm。
2. **长上下文路线**：DeepSeek-V4-Pro 走"滑动窗 + KV 压缩(CSA/HCA) + 压缩域 Lightning Indexer"三态；MiniMax-M3 走"原始 key 分块 + 块级 top-k 稀疏"；GLM-5.2 走"DSA + 跨层 indexer 共享"。
3. **残差**：DeepSeek-V4-Pro 独有 mHC 多流残差 + Sinkhorn 双随机约束；另两者为标准残差。
4. **MoE 精度与路由**：DeepSeek-V4-Pro 专家 fp4 + FP8 权重 + Hash/Topk 双路由；MiniMax-M3 全 BF16 + sigmoid 路由 + bias。
5. **激活函数**：DeepSeek-V4-Pro 为 SiLU clamp；MiniMax-M3 为自定义 swigluoai（sigmoid 门控 + alpha）。
6. **模态**：仅 MiniMax-M3 含视觉/视频分支（3D RoPE + patch_merge）。
7. **RoPE**：DeepSeek-V4-Pro 双 theta（main/compress）+ 交错式；MiniMax-M3 单一大 theta + 部分旋转。

## 十、benchmark 对照（取自 GLM-5.2 README）

| Benchmark | GLM-5.2 | GLM-5.1 | Qwen3.7-Max | MiniMax M3 | DeepSeek-V4-Pro |
|---|---:|---:|---:|---:|---:|
| HLE | 40.5 | 31 | 41.4 | 37 | 37.7 |
| AIME 2026 | 99.2 | 95.3 | 97 | - | 94.6 |
| GPQA-Diamond | 91.2 | 86.2 | 90 | 93 | 90.1 |
| SWE-bench Pro | 62.1 | 58.4 | 60.6 | 59 | 55.4 |
| Terminal Bench 2.1 (Terminus-2) | 81.0 | 63.5 | 75 | 65 | 64 |
| FrontierSWE (Dominance) | 74.4 | 30.5 | - | - | 29.0 |
| MCP-Atlas (Public) | 76.8 | 71.8 | 76.4 | 74.2 | 73.6 |

> 完整 benchmark 与评测脚注见 `glm52_readme.md`。
