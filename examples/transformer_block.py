"""Bloque Transformer decoder-only (pre-LN) escrito a mano + mini LM de caracteres."""
import math
import torch
from torch import nn

torch.manual_seed(0)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.h, self.dk).transpose(1, 2) for t in (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)  # (B, h, T, T)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))  # máscara causal
        out = scores.softmax(-1) @ v
        return self.proj(out.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, vocab, d_model=64, n_heads=4, n_layers=2, max_len=64):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.Sequential(*[Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, idx):
        pos = torch.arange(idx.shape[1])
        return self.head(self.ln(self.blocks(self.tok(idx) + self.pos(pos))))


text = "hola mundo. " * 50
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])

ctx = 16
starts = torch.arange(0, len(data) - ctx - 1)
x = torch.stack([data[i:i + ctx] for i in starts])
y = torch.stack([data[i + 1:i + ctx + 1] for i in starts])

model = TinyLM(len(chars))
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
for step in range(200):
    logits = model(x)
    loss = nn.functional.cross_entropy(logits.view(-1, len(chars)), y.view(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()

print(f"loss final={loss.item():.3f}")
assert logits.shape == (len(x), ctx, len(chars))
assert loss.item() < 0.5
