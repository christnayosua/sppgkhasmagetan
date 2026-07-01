"""
train_maze_ppo.py

Implementasi algoritma PPO (Proximal Policy Optimization) murni menggunakan PyTorch.
PPO adalah algoritma state-of-the-art yang lebih stabil dari DQN, cocok untuk lingkungan 3D.
"""

import gymnasium as gym
import miniworld

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

import numpy as np
import time

# ==================== Hyperparameters ====================
NUM_ENVS = 1              # Karena kita implementasi basic, kita pakai 1 environment
NUM_STEPS = 512           # Jumlah langkah yang dikumpulkan sebelum update (Rollout length)
TOTAL_TIMESTEPS = 500000  # Total langkah keseluruhan
BATCH_SIZE = 64           # Ukuran minibatch saat PPO update
N_EPOCHS = 4              # Jumlah epoch per update
GAMMA = 0.99              # Faktor diskon (Discount factor)
GAE_LAMBDA = 0.95         # Parameter GAE (Generalized Advantage Estimation)
CLIP_COEF = 0.2           # Batas (clip) untuk fungsi objektif PPO
ENTROPY_COEF = 0.01       # Bobot untuk mendorong eksplorasi (Entropy bonus)
VF_COEF = 0.5             # Bobot untuk Value Loss
LR = 3e-4                 # Learning rate (PPO biasanya lebih stabil dengan LR tetap)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== Preprocessing ====================
def preprocess_observation(obs):
    """Mengubah gambar RGB (H, W, C) numpy ke tensor PyTorch (C, H, W) yang dinormalisasi."""
    # MiniWorld-Maze-v0 output: (60, 80, 3)
    obs = np.transpose(obs, (2, 0, 1))  # Menjadi (3, 60, 80)
    obs = obs / 255.0                   # Normalisasi ke [0, 1]
    return torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

# ==================== Model Jaringan Saraf ====================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCriticCnn(nn.Module):
    def __init__(self, input_channels, n_actions):
        super(ActorCriticCnn, self).__init__()
        
        # Fitur Ekstraktor CNN (Sama dengan DQN)
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 4 * 7, 512)),
            nn.ReLU(),
        )
        
        # Actor: Memilih aksi
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        
        # Critic: Menilai state (V-value)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

# ==================== Rollout Buffer ====================
class RolloutBuffer:
    def __init__(self, num_steps, obs_shape):
        self.states = torch.zeros((num_steps, *obs_shape)).to(device)
        self.actions = torch.zeros((num_steps,)).to(device)
        self.logprobs = torch.zeros((num_steps,)).to(device)
        self.rewards = torch.zeros((num_steps,)).to(device)
        self.dones = torch.zeros((num_steps,)).to(device)
        self.values = torch.zeros((num_steps,)).to(device)
        self.step = 0

    def add(self, state, action, logprob, reward, done, value):
        self.states[self.step] = state
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value
        self.step += 1

    def clear(self):
        self.step = 0

# ==================== Loop Training PPO ====================
def train():
    env = gym.make("MiniWorld-Maze-v0", render_mode="rgb_array")
    n_actions = env.action_space.n
    obs_shape = (3, 60, 80)
    
    print(f"\nMemulai training PPO untuk MiniWorld Maze...")
    print(f"Perangkat: {device}")
    
    agent = ActorCriticCnn(input_channels=3, n_actions=n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LR, eps=1e-5)
    
    buffer = RolloutBuffer(NUM_STEPS, obs_shape)
    
    global_step = 0
    start_time = time.time()
    
    obs, _ = env.reset()
    state = preprocess_observation(obs).squeeze(0)  # Hapus dimensi batch sementara
    
    episode_reward = 0
    recent_rewards = []
    
    while global_step < TOTAL_TIMESTEPS:
        # 1. KUMPULKAN DATA (ROLLOUT)
        for step in range(NUM_STEPS):
            global_step += 1
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(state.unsqueeze(0))
                
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            # Reward shaping ringan
            if not terminated:
                reward -= 0.01
                if action.item() in [0, 1]: # Penalti berputar
                    reward -= 0.005
                    
            episode_reward += reward
            
            # Simpan ke buffer
            buffer.add(state, action.item(), logprob, reward, done, value.flatten())
            
            if done:
                recent_rewards.append(episode_reward)
                if len(recent_rewards) > 50:
                    recent_rewards.pop(0)
                
                obs, _ = env.reset()
                state = preprocess_observation(obs).squeeze(0)
                episode_reward = 0
            else:
                state = preprocess_observation(next_obs).squeeze(0)
                
        # 2. HITUNG ADVANTAGE MENGGUNAKAN GAE
        with torch.no_grad():
            next_value = agent.get_value(state.unsqueeze(0)).flatten()
            advantages = torch.zeros_like(buffer.rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - int(done) # status done dari step terakhir
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - buffer.dones[t + 1]
                    nextvalues = buffer.values[t + 1]
                    
                delta = buffer.rewards[t] + GAMMA * nextvalues * nextnonterminal - buffer.values[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                
            returns = advantages + buffer.values

        # 3. UPDATE NETWORK (PPO EPOCHS)
        b_states = buffer.states
        b_actions = buffer.actions
        b_logprobs = buffer.logprobs
        b_advantages = advantages
        b_returns = returns
        b_values = buffer.values

        b_inds = np.arange(NUM_STEPS)
        
        for epoch in range(N_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, NUM_STEPS, BATCH_SIZE):
                end = start + BATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_states[mb_inds], b_actions[mb_inds].long())
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                # Normalisasi advantage per minibatch
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss (Clipped Surrogate Objective)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy loss (mendorong eksplorasi)
                entropy_loss = entropy.mean()

                # Total loss
                loss = pg_loss - ENTROPY_COEF * entropy_loss + v_loss * VF_COEF

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        buffer.clear()

        # Log Progress (Setiap kali selesai 1 update = NUM_STEPS langkah)
        avg_reward = np.mean(recent_rewards) if len(recent_rewards) > 0 else 0
        elapsed = time.time() - start_time
        print(f"Step: {global_step:6d}/{TOTAL_TIMESTEPS} | Avg(50) Reward: {avg_reward:7.2f} | Time: {elapsed:.1f}s")
        
        # Simpan setiap beberapa langkah
        if global_step % 50000 == 0 or global_step >= TOTAL_TIMESTEPS:
            torch.save(agent.state_dict(), "maze_ppo_model.pth")

    print("\nTraining PPO Selesai!")
    env.close()

if __name__ == "__main__":
    train()
