# config.py
import torch

# 全局配置
CONFIG = {
    # 数据配置
    "data_dir": r"D:\八斗大模型\现期\第五周\金庸书籍语料",           # 金庸小说存放目录
    "seq_len": 128,                   # 训练时每个样本长度（切分数据用）
    
    # 模型配置
    "embed_dim": 256,
    "num_layers": 6,
    "num_heads": 8,
    "ff_dim": 1024,
    "max_seq_len": 512,               # 模型最大支持长度
    "dropout": 0.1,
    
    # 训练配置
    "batch_size": 32,
    "lr": 3e-4,
    "epochs": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_path": r"D:\八斗大模型\现期\第五周\作业\best_model.pth"  
}
