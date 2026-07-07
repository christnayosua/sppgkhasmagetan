"""
run_maze.py — Menjalankan demo MiniWorld-Maze-v0

Fitur:
- Jika model 'maze_dqn_model.pth' ditemukan: gunakan agen CNN-DQN terlatih
- Jika tidak ada model: gunakan aksi acak (random exploration)
- Simpan screenshot sudut pandang agen sebagai 'maze_screenshot.png'
- Tambahkan flag --visual untuk membuka jendela 3D interaktif

Penggunaan:
  python run_maze.py            # Screenshot saja (tanpa jendela)
  python run_maze.py --visual   # Buka jendela 3D untuk melihat labirin
"""

import gymnasium as gym
import miniworld
from PIL import Image
import numpy as np
import os
import sys
import time

# Coba import PyTorch dan model CNN — jika tidak tersedia, tetap bisa jalan dengan mode acak
try:
    import torch
    from train_maze_dqn import CnnDQN, preprocess_observation
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def load_trained_model(n_actions, device):
    """
    Muat model CNN-DQN yang sudah dilatih (jika ada).
    Mengembalikan model atau None jika file tidak ditemukan.
    """
    model_path = "maze_dqn_model.pth"
    if not os.path.exists(model_path):
        return None

    model = CnnDQN(input_channels=3, n_actions=n_actions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Model terlatih berhasil dimuat dari '{model_path}'")
    return model


def main():
    # ---- Cek apakah user mau lihat visualisasi ----
    visual_mode = "--visual" in sys.argv
    render_mode = "human" if visual_mode else "rgb_array"

    if visual_mode:
        print("Mode Visual: Jendela 3D akan terbuka.")
    
    # ---- Inisialisasi Environment ----
    env = gym.make("MiniWorld-Maze-v0", render_mode=render_mode)
    obs, info = env.reset()
    print("Environment MiniWorld-Maze-v0 berhasil diinisialisasi.")

    # ---- Cek apakah ada model terlatih ----
    use_model = False
    model = None
    device = None

    if TORCH_AVAILABLE:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_trained_model(n_actions=env.action_space.n, device=device)
        if model is not None:
            use_model = True
            print("Mode: Agen CNN-DQN Terlatih")
        else:
            print("Mode: Eksplorasi Acak (model belum dilatih)")
    else:
        print("Mode: Eksplorasi Acak (PyTorch tidak tersedia)")

    # ---- Jalankan beberapa langkah di dalam labirin ----
    total_steps = 100 if visual_mode else (50 if use_model else 20)
    total_reward = 0.0

    for step in range(total_steps):
        if use_model:
            # Gunakan model CNN-DQN untuk memilih aksi
            # Untuk mode human, kita perlu ambil observasi dari env langsung
            state_tensor = preprocess_observation(obs).to(device)
            with torch.no_grad():
                action = model(state_tensor).max(1)[1].item()
        else:
            # Pilih aksi acak
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if visual_mode:
            time.sleep(0.05)  # Sedikit jeda agar visualisasi bisa diikuti

        if terminated or truncated:
            print(f"   Episode selesai di langkah {step + 1} (reward: {total_reward:.2f})")
            obs, info = env.reset()
            total_reward = 0.0

    # ---- Simpan screenshot (hanya di mode rgb_array) ----
    if not visual_mode:
        img_array = env.render()
        img = Image.fromarray(img_array)
        img.save("maze_screenshot.png")
        print(f"\nScreenshot berhasil disimpan sebagai 'maze_screenshot.png'")

    env.close()
    print("Selesai.")


if __name__ == "__main__":
    main()

