import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import FlattenObservation
import torch
import time
import os
import matplotlib.pyplot as plt

# Import arsitektur DQN yang sama dari train_dqn.py
from train_dqn import DQN

def save_screenshot():
    print("Mengambil screenshot...")
    env = gym.make('gymnasium_env/GridWorld-v0', render_mode="rgb_array")
    env = FlattenObservation(env)
    env.reset()
    
    frame = env.render()
    if frame is not None:
        os.makedirs("video", exist_ok=True)
        plt.imshow(frame)
        plt.axis('off')
        plt.savefig("video/screenshot.png", bbox_inches='tight', pad_inches=0)
        print("Screenshot berhasil disimpan ke video/screenshot.png")
    env.close()

def run_demo():
    print("Memulai demonstrasi (20 detik)...")
    
    env = gym.make('gymnasium_env/GridWorld-v0', render_mode="human")
    env = FlattenObservation(env)
    
    n_actions = env.action_space.n
    n_observations = env.observation_space.shape[0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(n_observations, n_actions).to(device)
    
    if os.path.exists("dqn_model.pth"):
        model.load_state_dict(torch.load("dqn_model.pth", map_location=device, weights_only=True))
        print("Berhasil memuat model yang telah dilatih (dqn_model.pth).")
    else:
        print("Peringatan: dqn_model.pth tidak ditemukan. Menggunakan bobot model acak (untrained).")
        print("Harap jalankan 'python train_dqn.py' terlebih dahulu.")
        
    model.eval()

    start_time = time.time()
    episodes = 0
    
    # Jalankan terus menerus selama sedikit di atas 20 detik agar pengguna sempat merekam layarnya
    while time.time() - start_time < 20:
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        
        while not done:
            with torch.no_grad():
                action = model(state).max(1)[1].view(1, 1).item()
                
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Jika sudah mencapai 25 detik, kita bisa hentikan loop bagian dalam juga
            if time.time() - start_time >= 25:
                break
        
        episodes += 1
        # Beri jeda sejenak di antara episode agar transisi terlihat jelas
        time.sleep(0.5)

    print(f"Demonstrasi selesai. Total episode yang dimainkan: {episodes}")
    env.close()

if __name__ == "__main__":
    save_screenshot()
    run_demo()
