"""LSTM: predice el siguiente valor de una serie senoidal con ruido."""
import torch
from torch import nn

torch.manual_seed(0)

t = torch.arange(0, 400, dtype=torch.float32)
series = torch.sin(0.1 * t) + 0.05 * torch.randn_like(t)

window = 30
x = torch.stack([series[i:i + window] for i in range(len(series) - window)]).unsqueeze(-1)
y = series[window:].unsqueeze(-1)
split = int(0.8 * len(x))


class Forecaster(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.out(h[:, -1])  # último estado oculto -> predicción


model = Forecaster()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for _ in range(150):
    loss = nn.functional.mse_loss(model(x[:split]), y[:split])
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    test_mse = nn.functional.mse_loss(model(x[split:]), y[split:]).item()
print(f"test MSE={test_mse:.4f}")
assert test_mse < 0.05
