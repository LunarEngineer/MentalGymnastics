from .functionbank import FunctionBank
from gym.envs.registration import register

__all__ = [
    'utils',
    'FunctionBank'
]

register(
    id='mental-v0',
    entry_point='mentalgym.envs:MentalEnv'
)
