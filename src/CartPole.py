#import pyvirtualdisplay
#_display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
#_ = _display.start()

import gym
import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer
from ray.tune.registry import register_env

ray.shutdown()
#ray.init(include_webui=False, ignore_reinit_error=True)
ray.init(ignore_reinit_error=True)

#ENV = 'CartPoleBulletEnv-v1'
ENV = 'CartPole-v0'
def make_env(env_config):
#    import pybullet_envs
    return gym.make(ENV)
register_env(ENV, make_env)
TARGET_REWARD = 190
TRAINER = DQNTrainer

#print("Here!")
tune.run(
    TRAINER,
    stop={"episode_reward_mean": TARGET_REWARD},
    config={
        "env": ENV,
        "num_workers": 1,
        "num_gpus": 0,
        "monitor": True,
        "evaluation_num_episodes": 50,
        "log_level": "DEBUG",
        "framework": "torch"
    },
#    resources_per_trial={"cpu": 4, "gpu": 1}
)
