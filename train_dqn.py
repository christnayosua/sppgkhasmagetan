import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import FlattenObservation

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import random
from collections import namedtuple, deque
import time

# Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminated'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQN Architecture
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

def train():
    env = gym.make('gymnasium_env/GridWorld-v0')
    env = FlattenObservation(env)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan perangkat: {device}")

    # Hyperparameters
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 50000
    TAU = 0.005
    LR = 1e-4

    n_actions = env.action_space.n
    n_observations = env.observation_space.shape[0]

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    steps_done = 0

    def select_action(state):
        nonlocal steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(not s for s in batch.terminated), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s, t in zip(batch.next_state, batch.terminated) if not t])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
            
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    num_episodes = 5000 if torch.cuda.is_available() else 3000
    
    print("\nMemulai training agen Deep Q-Learning...\n")
    start_time = time.time()
    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        for t in range(env.spec.max_episode_steps if env.spec.max_episode_steps else 300):
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device, dtype=torch.float32)
            
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward, terminated)

            state = next_state

            optimize_model()

            # Soft update target network
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                break
                
        if (i_episode + 1) % 5 == 0 or (i_episode + 1) == num_episodes:
            progress = (i_episode + 1) / num_episodes
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            # Progress bar sederhana
            rocket_pos = min(int(bar_length * progress), bar_length - 1)
            rocket_track = ['-'] * bar_length
            rocket_track[rocket_pos] = '>'
            track_str = "".join(rocket_track)
            
            print(f"\r[{bar}] Episode {i_episode+1:4d}/{num_episodes} | {track_str}", end="", flush=True)

    print(f"\n\nSelesai! Agen berhasil dilatih dalam {time.time() - start_time:.2f} detik.")
    
    torch.save(policy_net.state_dict(), "dqn_model.pth")
    print("Model disimpan sebagai 'dqn_model.pth'\n")
    env.close()

if __name__ == "__main__":
    train()
