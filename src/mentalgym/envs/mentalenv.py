import gym

import numpy as np
from gym import (
    # error,
    spaces,
    # utils
)

# from gym.utils import seeding
# from typing import Tuple, Any

# from mentalgym.utils.reward import (
# monotonic_reward,
# connection_reward,
# completion_reward,
# )


class MentalEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(MentalEnv, self).__init__()

        self.experiment_space_min = -100
        self.experiment_space_max = 100
        self.radius_min = 1
        self.radius_max = self.experiment_space_max - self.experiment_space_min
        self.number_functions = 8
        self.number_metrics = 2
        self.max_steps = 10
        self._step = 0
        self.state = self.reset()

        self.action_space = gym.spaces.Box(4,)
        # self.action_space = gym.spaces.Dict(
            # {
                # "function_id": spaces.Discrete(self.number_functions),
                # "location": spaces.Box(
                    # low=self.experiment_space_min,
                    # high=self.experiment_space_max,
                    # shape=(2,),
                # ),
                # "radius": spaces.Box(
                    # low=self.radius_min, high=self.radius_max, shape=(1,)
                # ),
            # }
        # )
        # self.action_space = spaces.Tuple([spaces.Discrete(1)])

        # Create iterable for function IDs
#        fn_ids_tuple = ()
#        for i in range(self.max_steps):
#            fn_ids_tuple += (spaces.Discrete(self.number_functions),)

        # Fn ID, locX, locY, FnConnection
        self.observation_space = gym.spaces.Box(4,self.max_steps)


        # self.observation_space = gym.spaces.Dict(
            # {



                # "experiment_space": spaces.Dict(
                    # {
                        # # Functions currently placed in the experiment space
                        # "function_ids": spaces.Tuple(fn_ids_tuple),
                        # # Locations of those functions
                        # "function_locations": spaces.Box(
                            # low=self.experiment_space_min,
                            # high=self.experiment_space_max,
                            # shape=(self.max_steps,),
                        # ),
                        # # This is 1 if a connection is made
                        # "function_connection": spaces.MultiBinary(1),
                    # }
                # ),
                # #  "function_metrics": np.ndarray(
                # #      (self.number_functions, self.number_metrics)
                # #  ),  # populate with the actual actions
            # }
        # )
        # self.observation_space = spaces.Tuple([spaces.Discrete(1)])

    #   def step(self, action:Dict[int, Any]):
    def step(self, action):
        done = False

        # get all source and sink nodes from the function bank
        # get the function representation from the function bank

        # determine if that is a atomic or composed function
        # if atomic
        #   determine all input notes using radius and other nodes locations
        # else
        #   just recreate as it was placed

        # When building the net, don't hook on output layer until done
        # net is build at this point

        # if reached maximum steps (when done) or if connected to the sink
        #   run and measure

        # calculate reward based on current observation space and

        print("Action:", action)
        self._step += 1

        if self._step == self.max_steps:
            done = True

        info = {}

        # TODO: Check if new function connects to ANY node
#        for

#        self.state["experiment_space"]["function_ids"] += (action["function_id"],)
#        self.state["experiment_space"]["function_locations"] += (action["location"],)

#        if not self.function_connection:
#            if distance(input, action["location"]
#            if action['radius']

        reward = 0

        return self.state, reward, done, info

    def reset(self):
        self._step = 0
        self.function_connection = False

        # TODO: Initialize state to Input (get ID for this input to query in .step())
        state = {
            "experiment_space": {
                "function_ids": tuple([0] * self.max_steps),
                "function_locations": [0] * self.max_steps,
                "function_connection": 0,
            }
        }
        state = np.zeros(3)
        return state

    def render(self, mode="human"):
        pass

    def close(self):
        pass
