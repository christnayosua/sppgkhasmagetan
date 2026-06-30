"""
train_maze_dqn.py — Training agen CNN-DQN untuk MiniWorld-Maze-v0

Perbedaan utama dengan train_dqn.py (GridWorld):
- Observasi berupa gambar RGB (60x80x3), bukan 4 angka koordinat
- Menggunakan CNN (Convolutional Neural Network) sebagai feature extractor
- Preprocessing: normalisasi pixel [0,255] -> [0,1], transpose ke format (C, H, W)

Referensi:
- MiniWorld Maze: https://miniworld.farama.org/environments/maze/
- DQN dengan CNN: https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import gymnasium as gym
import miniworld

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import random
from collections import namedtuple, deque
import time
import os

# ==================== Replay Memory ====================
# Sama seperti di train_dqn.py — menyimpan transisi pengalaman agen
# agar proses belajar bisa mengambil sampel acak (mengurangi korelasi temporal)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminated'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Simpan satu transisi ke dalam buffer."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# ==================== Arsitektur CNN-DQN ====================
# Berbeda dengan DQN di train_dqn.py yang hanya memakai Linear layers,
# di sini kita HARUS menggunakan Convolutional layers karena input-nya
# adalah gambar (image), bukan angka flat.
#
# Alur data:
#   Gambar RGB (3, 60, 80)
#     -> Conv2d Layer 1 (32 filter, kernel 8x8, stride 4) -> ReLU
#     -> Conv2d Layer 2 (64 filter, kernel 4x4, stride 2) -> ReLU
#     -> Conv2d Layer 3 (64 filter, kernel 3x3, stride 1) -> ReLU
#     -> Flatten menjadi vektor 1D
#     -> Linear Layer (512 neuron) -> ReLU
#     -> Linear Layer Output (3 neuron = 3 aksi)

class CnnDQN(nn.Module):
    """
    CNN-DQN: Convolutional Deep Q-Network.

    Arsitektur ini terinspirasi dari paper DeepMind "Playing Atari with
    Deep Reinforcement Learning" (2013), yang pertama kali menunjukkan
    bahwa neural network bisa belajar langsung dari pixel mentah.

    Mengapa CNN?
    - Gambar memiliki struktur spasial (piksel yang berdekatan saling terkait)
    - CNN mampu mengenali pola visual (dinding, koridor, objek merah)
    - Jauh lebih efisien daripada MLP untuk data gambar
    """

    def __init__(self, input_channels, n_actions):
        super(CnnDQN, self).__init__()

        # Bagian Konvolusi — mengekstrak fitur visual dari gambar
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Hitung ukuran output setelah melewati semua conv layers
        # Input: (3, 60, 80) -> conv1: (32, 14, 19) -> conv2: (64, 6, 8) -> conv3: (64, 4, 6)
        conv_output_size = 64 * 4 * 6  # = 1536

        # Bagian Fully Connected — mengambil keputusan berdasarkan fitur visual
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Ratakan menjadi vektor 1D sebelum masuk ke linear layers
        x = x.reshape(x.size(0), -1)

        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ==================== Preprocessing Observasi ====================

def preprocess_observation(obs):
    """
    Mengubah observasi mentah dari MiniWorld menjadi format tensor PyTorch.

    Langkah-langkah:
    1. Konversi dari uint8 [0, 255] ke float32 [0.0, 1.0] — normalisasi
    2. Transpose dari (H, W, C) ke (C, H, W) — format yang diharapkan PyTorch
    3. Tambah dimensi batch di depan -> (1, C, H, W)

    Mengapa perlu preprocessing?
    - PyTorch Conv2d mengharapkan format (batch, channel, height, width)
    - Normalisasi membantu training lebih stabil (gradien tidak meledak)
    """
    # (60, 80, 3) -> float32 [0, 1]
    obs = obs.astype(np.float32) / 255.0
    # (60, 80, 3) -> (3, 60, 80) — PyTorch mau channel-first
    obs = np.transpose(obs, (2, 0, 1))
    # Tambah dimensi batch -> (1, 3, 60, 80)
    return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)


# ==================== Fungsi Utama Training ====================

def train():
    """
    Melatih agen CNN-DQN untuk menyelesaikan labirin 3D MiniWorld-Maze-v0.

    Konsep utama (sama seperti DQN di GridWorld, tapi dengan CNN):
    1. Agen melihat gambar dari sudut pandangnya (first-person view)
    2. CNN mengekstrak fitur visual (dinding, koridor, kotak merah)
    3. Berdasarkan fitur tersebut, network memprediksi Q-value untuk setiap aksi
    4. Agen memilih aksi terbaik (atau acak saat eksplorasi)
    5. Pengalaman disimpan di Replay Memory, lalu di-sample untuk training
    """

    # ---- Inisialisasi Environment ----
    env = gym.make("MiniWorld-Maze-v0", render_mode="rgb_array")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan perangkat: {device}")

    # ---- Hyperparameters ----
    BATCH_SIZE = 64         # Jumlah sampel per update (lebih kecil dari GridWorld karena data lebih besar)
    GAMMA = 0.99            # Faktor diskon — seberapa penting reward masa depan
    EPS_START = 1.0         # Epsilon awal — awalnya 100% eksplorasi acak
    EPS_END = 0.05          # Epsilon minimum — di akhir, 5% masih acak
    EPS_DECAY = 30000       # Kecepatan penurunan epsilon (lebih lambat karena maze lebih kompleks)
    TAU = 0.005             # Kecepatan soft-update target network
    LR = 1e-4               # Learning rate optimizer
    MEMORY_CAPACITY = 50000 # Kapasitas replay buffer

    # ---- Setup Networks ----
    n_actions = env.action_space.n  # 3 aksi: belok kiri, belok kanan, maju
    input_channels = 3              # RGB = 3 channel warna

    policy_net = CnnDQN(input_channels, n_actions).to(device)
    target_net = CnnDQN(input_channels, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Target network tidak perlu gradien

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(MEMORY_CAPACITY)

    steps_done = 0

    # ---- Fungsi Pemilihan Aksi (Epsilon-Greedy) ----
    def select_action(state_tensor):
        """
        Pilih aksi berdasarkan strategi epsilon-greedy:
        - Dengan probabilitas epsilon: aksi acak (eksplorasi)
        - Dengan probabilitas (1 - epsilon): aksi terbaik menurut CNN (eksploitasi)
        """
        nonlocal steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1.0 * steps_done / EPS_DECAY)
        steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state_tensor).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    # ---- Fungsi Optimisasi (Belajar dari Pengalaman) ----
    def optimize_model():
        """
        Ambil batch transisi dari memory, hitung loss, lalu update bobot network.
        Ini adalah inti dari proses "belajar" agen.
        """
        if len(memory) < BATCH_SIZE:
            return

        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Buat mask: mana saja state yang bukan terminal (permainan belum selesai)
        non_final_mask = torch.tensor(
            tuple(not t for t in batch.terminated),
            device=device, dtype=torch.bool
        )
        non_final_next_states = torch.cat(
            [s for s, t in zip(batch.next_state, batch.terminated) if not t]
        ).to(device)

        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)

        # Q(s, a) — nilai Q untuk aksi yang benar-benar diambil
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # V(s') — nilai state berikutnya menurut target network
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            if non_final_mask.any():
                next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

        # Bellman equation: Q_target = reward + gamma * V(s')
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Huber Loss — lebih robust terhadap outlier dibanding MSE
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    # ---- Loop Training Utama ----
    num_episodes = 500
    max_steps_per_episode = 500  # Batas langkah per episode agar agen punya cukup waktu

    print(f"\nMemulai training CNN-DQN untuk MiniWorld Maze...")
    print(f"   Episodes     : {num_episodes}")
    print(f"   Max steps    : {max_steps_per_episode}")
    print(f"   Action space : {n_actions} (Kiri, Kanan, Maju)")
    print(f"   Observation  : Gambar RGB 60x80 pixel\n")

    start_time = time.time()
    total_rewards = []
    best_reward = float('-inf')

    for i_episode in range(num_episodes):
        obs, info = env.reset()
        state = preprocess_observation(obs).to(device)
        episode_reward = 0.0

        for t in range(max_steps_per_episode):
            # Pilih dan jalankan aksi
            action = select_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            episode_reward += reward

            reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)
            done = terminated or truncated

            if terminated:
                next_state = torch.zeros_like(state)
            else:
                next_state = preprocess_observation(next_obs).to(device)

            # Simpan pengalaman
            memory.push(state, action, next_state, reward_tensor, terminated)
            state = next_state

            # Belajar dari pengalaman
            optimize_model()

            # Soft update target network
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = (
                    policy_net_state_dict[key] * TAU +
                    target_net_state_dict[key] * (1 - TAU)
                )
            target_net.load_state_dict(target_net_state_dict)

            if done:
                break

        total_rewards.append(episode_reward)

        # Simpan model terbaik
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(policy_net.state_dict(), "maze_dqn_model.pth")

        # Progress bar
        if (i_episode + 1) % 5 == 0 or (i_episode + 1) == num_episodes:
            progress = (i_episode + 1) / num_episodes
            bar_length = 30
            filled = int(bar_length * progress)
            bar = '=' * filled + '-' * (bar_length - filled)

            avg_reward = np.mean(total_rewards[-50:])  # Rata-rata 50 episode terakhir
            eps_current = EPS_END + (EPS_START - EPS_END) * np.exp(-1.0 * steps_done / EPS_DECAY)

            print(
                f"\r[{bar}] "
                f"Ep {i_episode+1:4d}/{num_episodes} | "
                f"Reward: {episode_reward:7.2f} | "
                f"Avg(50): {avg_reward:7.2f} | "
                f"eps: {eps_current:.3f}",
                end="", flush=True
            )

    elapsed = time.time() - start_time
    print(f"\n\nTraining selesai dalam {elapsed:.1f} detik.")
    print(f"Model terbaik disimpan sebagai 'maze_dqn_model.pth'")
    print(f"Reward terbaik: {best_reward:.2f}\n")

    env.close()


if __name__ == "__main__":
    train()
