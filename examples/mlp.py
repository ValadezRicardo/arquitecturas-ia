"""MLP: clasifica dos espirales entrelazadas (no separables linealmente)."""
import math
import torch
from torch import nn

torch.manual_seed(0)


def make_spirals(n=500):
    t = torch.linspace(0.25, 3 * math.pi, n)
    a = torch.stack([t * torch.cos(t), t * torch.sin(t)], dim=1)
    b = -a
    x = torch.cat([a, b]) / 10 + 0.02 * torch.randn(2 * n, 2)
    y = torch.cat([torch.zeros(n), torch.ones(n)]).long()
    return x, y


x, y = make_spirals()
model = nn.Sequential(
    nn.Linear(2, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 2),
)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

for step in range(500):
    loss = nn.functional.cross_entropy(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()

acc = (model(x).argmax(1) == y).float().mean().item()
print(f"loss={loss.item():.3f} acc={acc:.2%}")
assert acc > 0.9
