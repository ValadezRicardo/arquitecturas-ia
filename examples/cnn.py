"""CNN con bloque residual; tarea sintética: ¿el cuadrado está a la izquierda o a la derecha?"""
import torch
from torch import nn

torch.manual_seed(0)


def make_data(n=512, size=16):
    x = torch.zeros(n, 1, size, size)
    y = torch.randint(0, 2, (n,))
    for i in range(n):
        r = torch.randint(0, size - 4, (1,)).item()
        c = torch.randint(0, size // 2 - 4, (1,)).item() + (size // 2) * y[i].item()
        x[i, 0, r:r + 4, c:c + 4] = 1.0
    return x + 0.1 * torch.randn_like(x), y


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return torch.relu(x + self.f(x))  # conexión residual


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.res = ResBlock(16)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 2))

    def forward(self, x):
        return self.head(self.res(self.stem(x)))


x, y = make_data()
model = SmallCNN()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
for _ in range(60):
    loss = nn.functional.cross_entropy(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()

model.eval()
acc = (model(x).argmax(1) == y).float().mean().item()
print(f"loss={loss.item():.3f} acc={acc:.2%}")
assert acc > 0.8
