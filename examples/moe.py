"""Capa Mixture-of-Experts con router top-k y pérdida de balanceo de carga."""
import torch
from torch import nn

torch.manual_seed(0)


class MoE(nn.Module):
    def __init__(self, d_model=32, d_hidden=64, n_experts=4, k=2):
        super().__init__()
        self.k, self.n_experts = k, n_experts
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_model))
            for _ in range(n_experts)
        )

    def forward(self, x):                                   # x: (tokens, d_model)
        probs = self.router(x).softmax(-1)                  # (tokens, E)
        topw, topi = probs.topk(self.k, dim=-1)             # solo k expertos por token
        topw = topw / topw.sum(-1, keepdim=True)
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            token_idx, slot = (topi == e).nonzero(as_tuple=True)
            if token_idx.numel():                           # solo cómputo para tokens enrutados
                out[token_idx] += topw[token_idx, slot, None] * expert(x[token_idx])
        # balanceo (Switch): fracción de tokens por experto × prob. media del router
        frac = torch.bincount(topi[:, 0], minlength=self.n_experts).float() / len(x)
        aux = self.n_experts * (frac * probs.mean(0)).sum()
        return out, aux


moe = MoE()
x = torch.randn(256, 32)
y, aux = moe(x)
print(f"salida {tuple(y.shape)}  aux_loss={aux.item():.3f} (≈1.0 = balanceado)")
assert y.shape == x.shape
(y.pow(2).mean() + 0.01 * aux).backward()
