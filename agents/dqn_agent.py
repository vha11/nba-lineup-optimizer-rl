import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


# ===============================
# Red neuronal
# ===============================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ===============================
# Replay Buffer
# ===============================
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)

        return (
            np.array(s),
            np.array(a),
            np.array(r, dtype=np.float32),
            np.array(s2),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ===============================
# DQN Agent
# ===============================
class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.97,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        target_update=200,
        device="cpu",
    ):
        self.action_dim = action_dim
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device

        # redes
        self.q_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.memory = ReplayBuffer()
        self.step_count = 0

    # ----------------------------------
    def select_action(self, state, available_actions):
        if random.random() < self.epsilon:
            return random.choice(available_actions)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_net(state_t).detach().cpu().numpy()[0]

        valid_q = [(a, q_values[a]) for a in available_actions]
        best = max(valid_q, key=lambda x: x[1])[0]

        return best

    # ----------------------------------
    def store(self, s, a, r, s2, done):
        self.memory.push(s, a, r, s2, done)

    # ----------------------------------
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        s, a, r, s2, d = self.memory.sample(self.batch_size)

        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        q_values = self.q_net(s).gather(1, a)

        with torch.no_grad():
            next_q = self.target_net(s2).max(1, keepdim=True)[0]
            target = r + self.gamma * (1 - d) * next_q

        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # actualizar target network
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    # ----------------------------------
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ----------------------------------
    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))