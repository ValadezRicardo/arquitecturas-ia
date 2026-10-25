"""Capa GCN en PyTorch puro: clasificación de nodos en un grafo con dos comunidades."""
import torch
from torch import nn

torch.manual_seed(0)

# Grafo sintético: 2 comunidades de 30 nodos, conexiones densas dentro y escasas entre ellas
n = 60
labels = torch.tensor([0] * 30 + [1] * 30)
same = labels[:, None] == labels[None, :]
prob = torch.where(same, torch.tensor(0.3), torch.tensor(0.02))
A = (torch.rand(n, n) < prob).float()
A = torch.triu(A, 1)
A = A + A.T

# Features débiles: poco informativas por sí solas; la estructura hace el trabajo
X = labels.float().unsqueeze(1) * 0.3 + torch.randn(n, 8)


class GCNLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, x, a):
        a_hat = a + torch.eye(a.shape[0])           # self-loops
        d = a_hat.sum(1).pow(-0.5)
        a_norm = d[:, None] * a_hat * d[None, :]     # D^-1/2 (A+I) D^-1/2
        return a_norm @ self.lin(x)                  # agregación de vecinos


class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1, self.l2 = GCNLayer(8, 16), GCNLayer(16, 2)

    def forward(self, x, a):
        return self.l2(torch.relu(self.l1(x, a)), a)


train_mask = torch.zeros(n, dtype=torch.bool)
train_mask[[0, 1, 2, 30, 31, 32]] = True  # solo 6 nodos etiquetados (semi-supervisado)

model = GCN()
opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
for _ in range(200):
    out = model(X, A)
    loss = nn.functional.cross_entropy(out[train_mask], labels[train_mask])
    opt.zero_grad()
    loss.backward()
    opt.step()

acc = (model(X, A).argmax(1)[~train_mask] == labels[~train_mask]).float().mean().item()
print(f"accuracy en nodos sin etiqueta: {acc:.2%}")
assert acc > 0.8
