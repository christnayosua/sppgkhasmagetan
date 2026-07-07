from gymnasium.envs.registration import register

register(
    id="gymnasium_env/GridWorld-v0",
    entry_point="gymnasium_env.envs:GridWorldEnv",
    max_episode_steps=1000,
)

register(
    id="gymnasium_env/MLAgentsBasic-v0",
    entry_point="gymnasium_env.envs:MLAgentsBasicEnv",
    max_episode_steps=200,
)

