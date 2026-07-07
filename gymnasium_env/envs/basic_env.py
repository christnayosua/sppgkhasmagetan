import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class MLAgentsBasicEnv(gym.Env):
    """
    Kustom Gymnasium Environment yang meniru 'Basic' environment dari Unity ML-Agents.
    
    Tugas: Agen (posisi awal di tengah lintasan 1D, indeks 10) harus belajar
    untuk menavigasi ke Large Goal (indeks 17) untuk mendapatkan reward optimal +1.0,
    sambil menghindari/mengabaikan Small Goal (indeks 3) yang hanya memberikan +0.1.
    Setiap langkah dikenakan penalty -0.01.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, randomize_goals=False):
        super(MLAgentsBasicEnv, self).__init__()
        
        self.render_mode = render_mode
        self.randomize_goals = randomize_goals
        
        # Dimensi lintasan linear
        self.min_position = 0
        self.max_position = 20
        self.track_length = self.max_position - self.min_position + 1  # 21 sel
        
        # State
        self.position = 10
        self.small_goal_pos = 3
        self.large_goal_pos = 17
        
        # Action space: 0 (diam), 1 (kiri), 2 (kanan)
        self.action_space = spaces.Discrete(3)
        
        # Observation space:
        # Jika posisi gol tetap: Hanya posisi agen ternormalisasi (shape 1)
        # Jika posisi gol diacak: Posisi agen, small goal, dan large goal ternormalisasi (shape 3)
        if self.randomize_goals:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(3,), dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            )
            
        # PyGame render variables
        self.window_width = 640
        self.window_height = 180
        self.window = None
        self.clock = None
        self.cumulative_reward = 0.0
        self.step_count = 0

    def _get_obs(self):
        normalized_pos = float(self.position) / self.max_position
        if self.randomize_goals:
            normalized_small = float(self.small_goal_pos) / self.max_position
            normalized_large = float(self.large_goal_pos) / self.max_position
            return np.array([normalized_pos, normalized_small, normalized_large], dtype=np.float32)
        else:
            return np.array([normalized_pos], dtype=np.float32)

    def _get_info(self):
        return {
            "position": self.position,
            "small_goal": self.small_goal_pos,
            "large_goal": self.large_goal_pos,
            "steps": self.step_count,
            "cumulative_reward": self.cumulative_reward
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset state
        self.position = 10
        self.step_count = 0
        self.cumulative_reward = 0.0
        
        if self.randomize_goals:
            # Acak penempatan gol di posisi 3 atau 17
            if self.np_random.random() > 0.5:
                self.small_goal_pos = 3
                self.large_goal_pos = 17
            else:
                self.small_goal_pos = 17
                self.large_goal_pos = 3
        else:
            self.small_goal_pos = 3
            self.large_goal_pos = 17
            
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, info

    def step(self, action):
        # Gerakan berdasarkan action
        if action == 1:    # Kiri
            self.position -= 1
        elif action == 2:  # Kanan
            self.position += 1
        # action == 0: diam
        
        # Batasi posisi agen dalam lintasan
        self.position = int(np.clip(self.position, self.min_position, self.max_position))
        self.step_count += 1
        
        # Hitung reward dan status selelesai
        reward = -0.01  # Penalty langkah
        terminated = False
        
        if self.position == self.large_goal_pos:
            reward = 1.0
            terminated = True
        elif self.position == self.small_goal_pos:
            reward = 0.1
            terminated = True
            
        self.cumulative_reward += reward
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("ML-Agents Basic Environment")
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((240, 244, 248))  # Sleek light blue-gray background

        # Parameter gambar grid
        margin_x = 40
        grid_width = self.window_width - (margin_x * 2)
        cell_width = grid_width / self.track_length
        grid_y = 60
        grid_height = 40

        # Menggambar 21 sel lintasan
        for i in range(self.track_length):
            x = margin_x + (i * cell_width)
            rect = pygame.Rect(x, grid_y, cell_width, grid_height)
            
            # Tentukan warna sel
            if i == self.large_goal_pos:
                color = (40, 167, 69)  # Premium Green untuk Large Goal
            elif i == self.small_goal_pos:
                color = (255, 193, 7)  # Premium Yellow untuk Small Goal
            else:
                color = (255, 255, 255)  # Putih biasa untuk lintasan kosong
                
            pygame.draw.rect(canvas, color, rect)
            pygame.draw.rect(canvas, (180, 190, 200), rect, 1)  # Border sel

        # Menggambar agen (lingkaran biru)
        agent_x = margin_x + (self.position * cell_width) + (cell_width / 2)
        agent_y = grid_y + (grid_height / 2)
        pygame.draw.circle(canvas, (0, 123, 255), (int(agent_x), int(agent_y)), int(cell_width * 0.35))
        pygame.draw.circle(canvas, (0, 75, 160), (int(agent_x), int(agent_y)), int(cell_width * 0.35), 2)  # Border agen

        # Informasi Teks (Menggunakan system font gratis)
        pygame.font.init()
        font = pygame.font.SysFont("Outfit", 20)
        if font is None:
            font = pygame.font.Font(None, 24)

        pos_text = font.render(f"Posisi Agen: {self.position} / 20", True, (33, 37, 41))
        reward_text = font.render(f"Reward Kumulatif: {self.cumulative_reward:.2f}", True, (33, 37, 41))
        step_text = font.render(f"Langkah: {self.step_count}", True, (33, 37, 41))
        
        # Legend/Keterangan
        legend_font = pygame.font.SysFont("Outfit", 14)
        if legend_font is None:
            legend_font = pygame.font.Font(None, 18)
            
        l_text1 = legend_font.render("Kuning: Small Goal (+0.1)", True, (130, 95, 0))
        l_text2 = legend_font.render("Hijau: Large Goal (+1.0)", True, (20, 90, 30))
        l_text3 = legend_font.render("Biru: Agen", True, (0, 60, 130))

        # Gambar teks ke canvas
        canvas.blit(pos_text, (margin_x, 15))
        canvas.blit(reward_text, (margin_x + 180, 15))
        canvas.blit(step_text, (margin_x + 450, 15))
        
        canvas.blit(l_text1, (margin_x, 120))
        canvas.blit(l_text2, (margin_x + 200, 120))
        canvas.blit(l_text3, (margin_x + 400, 120))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
