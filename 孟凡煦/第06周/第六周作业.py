import jieba
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

# ===================== 全局配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128  # 文本最大长度
BATCH_SIZE = 32
EPOCHS = 3  # 训练轮数（快速验证效果）

# ===================== 1. 加载中文数据集（情感分类：正面/负面） =====================
# 加载ChnSentiCorp中文情感分类数据集
dataset = load_dataset("seamew/ChnSentiCorp")
train_texts = [item["text"] for item in dataset["train"]]
train_labels = [item["label"] for item in dataset["train"]]
test_texts = [item["text"] for item in dataset["test"]]
test_labels = [item["label"] for item in dataset["test"]]

# 简化数据（加速训练，新手可删除这行）
train_texts, train_labels = train_texts[:5000], train_labels[:5000]
test_texts, test_labels = test_texts[:1000], test_labels[:1000]

print(f"训练集大小：{len(train_texts)}，测试集大小：{len(test_texts)}")

# ===================== 方法1：传统机器学习 - TF-IDF + SVM =====================
print("\n" + "="*50)
print("方法1：传统机器学习（TF-IDF + SVM）训练中...")

# 中文分词
def tokenize(text):
    return " ".join(jieba.lcut(text))

train_cut = [tokenize(t) for t in train_texts]
test_cut = [tokenize(t) for t in test_texts]

# TF-IDF特征提取
tfidf = TfidfVectorizer(max_features=5000)
train_tfidf = tfidf.fit_transform(train_cut)
test_tfidf = tfidf.transform(test_cut)

# 训练SVM
svm_model = SVC(kernel="linear")
svm_model.fit(train_tfidf, train_labels)

# 预测与评估
svm_preds = svm_model.predict(test_tfidf)
svm_acc = accuracy_score(test_labels, svm_preds)
print("传统机器学习结果：")
print(classification_report(test_labels, svm_preds, digits=4))

# ===================== 方法2：浅层深度学习 - TextCNN =====================
print("\n" + "="*50)
print("方法2：浅层深度学习（TextCNN）训练中...")

# 构建词汇表
vocab = set()
for text in train_cut:
    vocab.update(text.split())
vocab = ["<PAD>", "<UNK>"] + list(vocab)
word2idx = {w: i for i, w in enumerate(vocab)}
VOCAB_SIZE = len(vocab)

# 文本转索引
def text2idx(text, max_len):
    words = text.split()
    idx = [word2idx.get(w, 1) for w in words]
    idx = idx[:max_len] + [0]*(max_len - len(idx))
    return idx

train_idx = [text2idx(t, MAX_LEN) for t in train_cut]
test_idx = [text2idx(t, MAX_LEN) for t in test_cut]

# 转换为Tensor
train_x = torch.LongTensor(train_idx).to(DEVICE)
train_y = torch.LongTensor(train_labels).to(DEVICE)
test_x = torch.LongTensor(test_idx).to(DEVICE)
test_y = torch.LongTensor(test_labels).to(DEVICE)

# TextCNN模型定义
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 128, 3)
        self.conv2 = nn.Conv1d(embed_dim, 128, 5)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.embedding(x).transpose(1,2)  # [B, embed, L]
        x1 = self.pool(torch.relu(self.conv1(x))).squeeze(-1)
        x2 = self.pool(torch.relu(self.conv2(x))).squeeze(-1)
        x = torch.cat([x1, x2], dim=1)
        return self.fc(x)

# 训练TextCNN
cnn_model = TextCNN(VOCAB_SIZE).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=1e-3)

cnn_model.train()
for epoch in range(EPOCHS):
    loss_sum = 0
    for i in range(0, len(train_x), BATCH_SIZE):
        x = train_x[i:i+BATCH_SIZE]
        y = train_y[i:i+BATCH_SIZE]
        optimizer.zero_grad()
        out = cnn_model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    print(f"TextCNN Epoch {epoch+1}, Loss: {loss_sum:.2f}")

# 评估
cnn_model.eval()
with torch.no_grad():
    cnn_out = cnn_model(test_x)
    cnn_preds = torch.argmax(cnn_out, dim=1).cpu().numpy()
cnn_acc = accuracy_score(test_labels, cnn_preds)
print("TextCNN结果：")
print(classification_report(test_labels, cnn_preds, digits=4))

# ===================== 方法3：预训练模型 - BERT + LoRA（高效微调） =====================
print("\n" + "="*50)
print("方法3：预训练模型（BERT + LoRA）训练中...")

# 加载BERT分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# 数据编码
def encode_text(texts):
    return tokenizer(
        texts, padding="max_length", truncation=True, 
        max_length=MAX_LEN, return_tensors="pt"
    )

train_enc = encode_text(train_texts)
test_enc = encode_text(test_texts)
train_input_ids = train_enc["input_ids"].to(DEVICE)
train_attention_mask = train_enc["attention_mask"].to(DEVICE)
test_input_ids = test_enc["input_ids"].to(DEVICE)
test_attention_mask = test_enc["attention_mask"].to(DEVICE)

# 加载BERT + LoRA配置
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.05
)
bert_model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
bert_model = get_peft_model(bert_model, lora_config).to(DEVICE)
bert_model.print_trainable_parameters()  # 仅训练0.2%参数！

# 训练
optimizer = optim.AdamW(bert_model.parameters(), lr=2e-5)
bert_model.train()
for epoch in range(EPOCHS):
    loss_sum = 0
    for i in range(0, len(train_input_ids), BATCH_SIZE):
        input_ids = train_input_ids[i:i+BATCH_SIZE]
        attention_mask = train_attention_mask[i:i+BATCH_SIZE]
        labels = train_y[i:i+BATCH_SIZE]
        optimizer.zero_grad()
        out = bert_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    print(f"BERT+LoRA Epoch {epoch+1}, Loss: {loss_sum:.2f}")

# 评估
bert_model.eval()
with torch.no_grad():
    out = bert_model(input_ids=test_input_ids, attention_mask=test_attention_mask)
    bert_preds = torch.argmax(out.logits, dim=1).cpu().numpy()
bert_acc = accuracy_score(test_labels, bert_preds)
print("BERT+LoRA结果：")
print(classification_report(test_labels, bert_preds, digits=4))

# ===================== 最终效果对比汇总 =====================
print("\n" + "="*60)
print("📊 三种文本分类方法效果对比（测试集）")
print("="*60)
print(f"1. 传统机器学习(TF-IDF+SVM)  准确率：{svm_acc:.4f}")
print(f"2. 浅层深度学习(TextCNN)     准确率：{cnn_acc:.4f}")
print(f"3. 预训练模型(BERT+LoRA)    准确率：{bert_acc:.4f}")
print("="*60)
