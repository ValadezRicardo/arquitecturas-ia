"""VAE sobre datos 2D sintéticos (dos clusters) con truco de reparametrización."""
import torch
from torch import nn

torch.manual_seed(0)
x = torch.cat([torch.randn(500, 2) * 0.3 + 2, torch.randn(500, 2) * 0.3 - 2])


class VAE(nn.Module):
    def __init__(self, d_in=2, d_z=2, h=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, h), nn.ReLU())
        self.mu, self.logvar = nn.Linear(h, d_z), nn.Linear(h, d_z)
        self.dec = nn.Sequential(nn.Linear(d_z, h), nn.ReLU(), nn.Linear(h, d_in))

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()  # reparametrización
        return self.dec(z), mu, logvar


model = VAE()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
for _ in range(500):
    xh, mu, logvar = model(x)
    recon = nn.functional.mse_loss(xh, x, reduction="sum") / len(x)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(x)
    loss = recon + kl  # -ELBO
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    samples = model.dec(torch.randn(1000, 2))  # generar desde el prior N(0, I)
print(f"recon={recon.item():.3f} kl={kl.item():.3f}")
print("media de muestras generadas:", samples.mean(0).tolist())
assert samples.shape == (1000, 2)
