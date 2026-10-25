"""Difusión DDPM en 1D: aprende a generar muestras de una mezcla de dos gaussianas."""
import torch
from torch import nn

torch.manual_seed(0)
T = 100
betas = torch.linspace(1e-4, 0.05, T)
alphas = 1 - betas
abar = torch.cumprod(alphas, 0)


def sample_data(n):
    sign = (torch.rand(n, 1) < 0.5).float() * 2 - 1
    return sign * 2 + 0.2 * torch.randn(n, 1)  # modos en -2 y +2


# la red predice el ruido ε dado (x_t, t)
net = nn.Sequential(nn.Linear(2, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 1))
opt = torch.optim.Adam(net.parameters(), lr=2e-3)

for step in range(4000):
    x0 = sample_data(256)
    t = torch.randint(0, T, (256,))
    eps = torch.randn_like(x0)
    a = abar[t].unsqueeze(1)
    xt = a.sqrt() * x0 + (1 - a).sqrt() * eps  # proceso hacia adelante en un paso
    pred = net(torch.cat([xt, t.unsqueeze(1).float() / T], dim=1))
    loss = nn.functional.mse_loss(pred, eps)
    opt.zero_grad()
    loss.backward()
    opt.step()

# muestreo ancestral (reverse process)
with torch.no_grad():
    x = torch.randn(2000, 1)
    for t in reversed(range(T)):
        tt = torch.full((2000, 1), t / T)
        eps_hat = net(torch.cat([x, tt], dim=1))
        x = (x - betas[t] / (1 - abar[t]).sqrt() * eps_hat) / alphas[t].sqrt()
        if t > 0:
            x = x + betas[t].sqrt() * torch.randn_like(x)

frac_pos = (x > 0).float().mean().item()
print(f"loss={loss.item():.3f}  fracción en modo +2: {frac_pos:.2f} (esperado ~0.5)")
assert 0.3 < frac_pos < 0.7
