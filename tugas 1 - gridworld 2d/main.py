import gymnasium as gym
import gymnasium_env
import time

env = gym.make('gymnasium_env/GridWorld-v0', render_mode="human")
observation, info = env.reset()

for _ in range(50):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
        
    time.sleep(0.5)

env.close()
