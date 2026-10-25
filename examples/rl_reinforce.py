"""REINFORCE con baseline en un entorno mínimo (corredor de 5 casillas). Solo usa torch."""
import torch
from torch import nn

torch.manual_seed(0)
N = 5  # casillas 0..4; el agente empieza en 0; la meta es la casilla 4


def step(s, a):  # a: 0 = izquierda, 1 = derecha
    s2 = max(0, min(N - 1, s + (1 if a == 1 else -1)))
    done = s2 == N - 1
    return s2, (1.0 if done else -0.01), done


policy = nn.Sequential(nn.Linear(N, 16), nn.Tanh(), nn.Linear(16, 2))
opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
one_hot = lambda s: torch.eye(N)[s]
baseline, gamma = 0.0, 0.99

for episode in range(400):
    s, logps, rewards = 0, [], []
    for _ in range(30):
        dist = torch.distributions.Categorical(logits=policy(one_hot(s)))
        a = dist.sample()
        logps.append(dist.log_prob(a))
        s, r, done = step(s, a.item())
        rewards.append(r)
        if done:
            break
    G, returns = 0.0, []
    for r in reversed(rewards):          # retorno descontado
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    baseline = 0.95 * baseline + 0.05 * returns.mean().item()
    loss = -(torch.stack(logps) * (returns - baseline)).sum()   # ∇J = E[∇logπ · (G − b)]
    opt.zero_grad()
    loss.backward()
    opt.step()

p_right = torch.softmax(policy(one_hot(0)), -1)[1].item()
print(f"P(derecha | estado 0) = {p_right:.2f}")
assert p_right > 0.8
