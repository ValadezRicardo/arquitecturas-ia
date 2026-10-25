"""GAN mínima: aprende una gaussiana 1D N(4, 1.25)."""
import torch
from torch import nn

torch.manual_seed(0)

G = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1))
D = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))
opt_g = torch.optim.Adam(G.parameters(), lr=1e-3)
opt_d = torch.optim.Adam(D.parameters(), lr=1e-3)
bce = nn.BCEWithLogitsLoss()

for step in range(3000):
    real = torch.randn(128, 1) * 1.25 + 4
    fake = G(torch.randn(128, 8))

    # 1) Discriminador: real -> 1, falso -> 0
    d_loss = bce(D(real), torch.ones(128, 1)) + bce(D(fake.detach()), torch.zeros(128, 1))
    opt_d.zero_grad()
    d_loss.backward()
    opt_d.step()

    # 2) Generador: engañar al discriminador (pérdida no saturante)
    g_loss = bce(D(fake), torch.ones(128, 1))
    opt_g.zero_grad()
    g_loss.backward()
    opt_g.step()

with torch.no_grad():
    s = G(torch.randn(5000, 8))
print(f"media={s.mean():.2f} (objetivo 4.00)  std={s.std():.2f} (objetivo 1.25)")
assert abs(s.mean().item() - 4) < 0.7
