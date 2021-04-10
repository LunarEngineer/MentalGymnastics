from gym.envs.registration import register

register(
    id='mental-v0',
    entry_point='mentalgym.envs:MentalEnv'
)