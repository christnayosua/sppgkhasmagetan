"""
run_basic.py — Menjalankan demo MLAgentsBasic-v0

Fitur:
- Jika model 'basic_dqn_model.pth' ditemukan: gunakan agen DQN terlatih.
- Jika tidak ada model: gunakan aksi acak (random exploration).
- Simpan screenshot lintasan/visualisasi sebagai 'basic_screenshot.png'.
- Tambahkan flag --visual untuk membuka jendela visualisasi interaktif PyGame.

Penggunaan:
  python run_basic.py            # Menyimpan screenshot saja
  python run_basic.py --visual   # Membuka jendela pygame dan menjalankan demonstrasi agen
"""

import gymnasium as gym
import gymnasium_env
from PIL import Image
import numpy as np
import os
import sys
import time

try:
    import torch
    from train_basic_dqn import BasicDQN
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def load_trained_model(n_actions, device):
    model_path = "basic_dqn_model.pth"
    if not os.path.exists(model_path):
        return None

    # Observasi space dari Basic environment hanya 1 float (posisi agen)
    model = BasicDQN(n_observations=1, n_actions=n_actions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Model terlatih berhasil dimuat dari '{model_path}'")
    return model

def main():
    visual_mode = "--visual" in sys.argv
    render_mode = "human" if visual_mode else "rgb_array"

    if visual_mode:
        print("Mode Visual: Jendela simulasi PyGame akan terbuka.")
    
    # Inisialisasi Environment (Gunakan randomize_goals=False agar mencocokkan default ML-Agents)
    env = gym.make("gymnasium_env/MLAgentsBasic-v0", render_mode=render_mode, randomize_goals=False)
    obs, info = env.reset()
    print("Environment MLAgentsBasic-v0 berhasil diinisialisasi.")

    use_model = False
    model = None
    device = None

    if TORCH_AVAILABLE:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_trained_model(n_actions=env.action_space.n, device=device)
        if model is not None:
            use_model = True
            print("Mode: Agen DQN Terlatih")
        else:
            print("Mode: Eksplorasi Acak (model belum dilatih)")
    else:
        print("Mode: Eksplorasi Acak (PyTorch tidak tersedia)")

    # Jalankan beberapa langkah simulasi
    total_steps = 100 if visual_mode else 1
    total_reward = 0.0

    for step in range(total_steps):
        if use_model:
            # Gunakan model DQN untuk memprediksi aksi
            state_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = model(state_tensor).max(1)[1].item()
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if visual_mode:
            time.sleep(0.3)  # Jeda agar pergerakan agen di layar terlihat jelas
            
        if terminated or truncated:
            print(f"   Episode selesai di langkah {step + 1} dengan total reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0.0
            if not visual_mode:
                break

    # Simpan screenshot hanya di mode rgb_array
    if not visual_mode:
        img_array = env.render()
        img = Image.fromarray(img_array)
        img.save("basic_screenshot.png")
        print("\nScreenshot visualisasi berhasil disimpan sebagai 'basic_screenshot.png'")

    env.close()
    print("Selesai.")

if __name__ == "__main__":
    main()
