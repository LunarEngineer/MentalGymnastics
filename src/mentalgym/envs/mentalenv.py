import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from typing import Tuple

class MentalEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		self.experiment_space_min = -100
		self.experiment_space_max = 100
		self.radius_min = 1
		self.radius_max = self.experiment_space_max - self.experiment_space_min
		self.number_functions = 8
		self.number_metrics = 2
		self.step_length = 10
		self._step = 0
		self.state = {
			"experiment_space": (self.number_functions, ())
		}

		self.action_space = gym.spaces.Dict({
			'function_id': spaces.Discrete(self.number_functions),
			'location': spaces.Box(
				low=self.experiment_space_min, 
				high=self.experiment_space_max, 
				shape=(2,)
			),
			'radius': spaces.Box(
				low=self.radius_min, 
				high=self.radius_max, 
				shape=(1,)
			)
		})
		
		self.observation_space = gym.spaces.Dict({
			'experiment_space': spaces.Tuple((
				spaces.Discrete(self.number_functions), 
				spaces.Box(
					low=self.experiment_space_min, 
					high=self.experiment_space_max, 
					shape=(2,)
				)
			)),
			'function_space': [0 for _ in range(self.number_functions)],
			'function_metrics': np.ndarray((self.number_functions, self.number_metrics)) # populate with the actual actions
		})
	def step(self, action:Tuple[int, Tuple[float, float]]):		

		done = False
		# create the state
		# figure out the next state

		# if action == 2:
		# 	reward = 1
		# else:
		# 	reward = -1

		self.state[self._step] = action
		
		if self._step == self.step_length:
			done = True

		self._step += 1
		info = {}

		return self.state, reward, done, info
	def reset(self):
		state = 0
		return state
	def render(self, mode='human'):
		pass
	def close(self):
		pass
