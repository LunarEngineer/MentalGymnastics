import constants
import types
import utils
from functionbank import FunctionBank
from gym.envs.registration import register

register(
    id='mental-v0',
    entry_point='mentalgym.envs:MentalEnv'
)
