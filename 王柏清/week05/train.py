import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import load_jinyong_dataset, build_vocab, get_dataloader
from model import GPT
from config import CONFIG  # 导入全局配置
import os

os.makedirs(os.path.dirname(CONFIG["save_path"]), exist_ok=True)
def train():
    cfg = CONFIG
    print("加载数据集...")
    all_text = load_jinyong_dataset(cfg["data_dir"])
    vocab, char2idx, idx2char = build_vocab(all_text)
    vocab_size = len(vocab)
    
    train_loader = get_dataloader(
        all_text, char2idx,
        seq_len=cfg["seq_len"],
        batch_size=cfg["batch_size"]
    )
    
    model = GPT(
        vocab_size=vocab_size,
        embed_dim=cfg["embed_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        ff_dim=cfg["ff_dim"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"]
    ).to(cfg["device"])
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-5)
    
    best_ppl = float('inf')
    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0
        
        for input_ids, labels in train_loader:
            input_ids = input_ids.to(cfg["device"])
            labels = labels.to(cfg["device"])
            
            optimizer.zero_grad()
            logits, loss = model(input_ids, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | PPL: {ppl:.4f}")
        
        if ppl < best_ppl:
            best_ppl = ppl
            torch.save({
                'model': model.state_dict(),
                'vocab': vocab,
                'char2idx': char2idx,
                'idx2char': idx2char,
                'config': cfg
            }, cfg["save_path"])
            
        scheduler.step()
        
    print("训练完成，最佳困惑度：", best_ppl)

if __name__ == "__main__":
    train()