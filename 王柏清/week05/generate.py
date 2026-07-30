import torch
import torch.nn.functional as F
from model import GPT
from data import detokenize
from config import CONFIG

def load_model(path=CONFIG["save_path"], device=CONFIG["device"]):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = GPT(
        vocab_size=len(ckpt["vocab"]),
        embed_dim=cfg["embed_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        ff_dim=cfg["ff_dim"],
        max_seq_len=cfg["max_seq_len"],
        dropout=0
    ).to(device)
    
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["char2idx"], ckpt["idx2char"], device

#top-p采样：根据概率分布累积概率，保留前top_p的token，其他token概率置0
@torch.no_grad()
def generate_top_p(model, char2idx, idx2char, prompt, max_new=100, top_p=0.9, temp=0.8, device='cuda'):
    input_ids = [char2idx.get(c, char2idx['<unk>']) for c in prompt]
    input_ids = torch.tensor([input_ids], device=device)
    
    for _ in range(max_new):
        input_ids = input_ids[:, -model.max_seq_len:]
        logits, _ = model(input_ids)
        logits = logits[:, -1, :] / temp
        
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > top_p
        remove[:, 0] = 0
        remove_idx = sorted_idx[remove]
        logits[:, remove_idx] = -float('inf')
        
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, 1)
        input_ids = torch.cat([input_ids, next_tok], dim=-1)
        
    return detokenize(input_ids[0].tolist(), idx2char)

#top-k采样：保留概率最高的top_k个token，其他token概率置0
@torch.no_grad()
def generate_top_k(model, char2idx, idx2char, prompt, max_new=100, top_k=10, temp=0.8, device='cuda'):
    input_ids = [char2idx.get(c, char2idx['<unk>']) for c in prompt]
    input_ids = torch.tensor([input_ids], device=device)

    for _ in range(max_new):
        input_ids = input_ids[:, -model.max_seq_len:]
        logits, _ = model(input_ids)
        logits = logits[:, -1, :] / temp
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        not_topk_idx = sorted_idx[:, top_k:]
        logits[:, not_topk_idx] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_tok], dim=-1)

    return detokenize(input_ids[0].tolist(), idx2char)

if __name__ == "__main__":
    model, c2i, i2c, device = load_model()
    prompt = "张无忌对乔峰说道"
    print(generate_top_p(model, c2i, i2c, prompt, device=device))
    print(generate_top_k(model, c2i, i2c, prompt, device=device))