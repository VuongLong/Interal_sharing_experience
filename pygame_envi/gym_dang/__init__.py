from gym.envs.registration import register

register(
    id='dang-v1',
    entry_point='gym_dang.envs:DangEnv',
)
