import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from typing import Tuple
from mentalgym.utils.reward import monotonic_reward, connection_reward, completion_reward

class MentalEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		self.experiment_space_min = -100
		self.experiment_space_max = 100
		self.radius_min = 1
		self.radius_max = self.experiment_space_max - self.experiment_space_min
		self.number_functions = 8
		self.number_metrics = 2
		self.max_steps = 10
		self._step = 0

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
			'experiment_space': spaces.Dict({
				# This needs to be an iterable as long as the episode length 
				# where each element can have a value from 0 to the number of functions in the function bank
				'function_ids': spaces.Discrete(self.max_steps), 
				# This is the location
				'function_locations': spaces.Box(
					low=self.experiment_space_min, 
					high=self.experiment_space_max, 
					shape=(self.max_steps,)
				),
				# This is 1 if a connection is made
				'function_connections': spaces.MultiBinary(1) 
			}),
			'function_space': [0 for _ in range(self.number_functions)],
			'function_metrics': np.ndarray((self.number_functions, self.number_metrics)) # populate with the actual actions
		})
	def step(self, action:Tuple[int, Tuple[float, float, float]]):		
		done = False
		
		# get all source and sink nodes from the function bank
		# get the function representation from the function bank

		# determine if that is a atomic or composed function
		# if atomic
		#	determine all input notes using radius and other nodes locations
		# else
		#	just recreate as it was placed

		# When building the net, don't hook on output layer until done
		# net is build at this point
		
		# if reached maximum steps (when done) or if connected to the sink
		# 	run and measure
		
		# calculate reward based on current observation space and 

		# monotonic_reward(self.observation_space["experiment_space"], action, )

		self.observation_space['experiment_space']
		
		if self._step == self.max_steps:
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