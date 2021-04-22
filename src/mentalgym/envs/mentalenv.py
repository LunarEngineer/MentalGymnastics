import gym

import numpy as np
from gym import (
    # error,
    spaces,
    # utils
)

from mentalgym.utils.data import function_bank
from mentalgym.utils.spaces import refresh_experiment_container

# from gym.utils import seeding
# from typing import Tuple, Any

from mentalgym.utils.reward import connection_reward, completion_reward


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

        self._experiment_space = refresh_experiment_container(function_bank=function_bank, min_loc=np.array([0, 0]), max_loc=np.array([10, 10]))
        print(self._experiment_space)

        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
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

        # Fn ID, locX, locY, FnConnection
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4, self.max_steps))


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


        # TODO: Check if new function connects to ANY node
        # check if the new function, connects to other actions that have been dropped
        # 
        switch(action["type"]):
            case "composed": 
                # radius doesn't matter, it is put into place
                # add to experiment space
            case "atomic":
                # radius does matter
                # make tree from experiment space location query(action.r)
                """
                >>> from scipy.spatial import cKDTree

                >>> points_ref = np.array([(1, 1), (3, 3), (4, 4), (5, 4), (6, 6)])
                >>> tree = cKDTree(points_ref)

                >>> idx = tree.query_ball_point((4, 4), action.r)
                >>> points_ref[idx]
                # array([[3, 3], [4, 4], [5, 4]])
                """
                # if it is a good action (that connects)
                #   go get the actual atomic function and instantiate it
                #   add to experiment space, not the function bank YET.
        
        # reward = connection_reward()

        print("Action:", action)
        self._step += 1

        if self._step == self.max_steps:
            # run _build_net()
            # reward = linear_completion_reward()
            done = True

        info = {}
        
        reward = 0

        # self.state: all the function ids, all the locations, connection

        return self.state, reward, done, info

    def _build_net(self):
        pass

    def reset(self):
        self._step = 0
        self.function_connection = False
        # TODO: Initialize state to Input (get ID for this input to query in .step())
        state = np.zeros((4, self.max_steps))
        return state

    def render(self, mode="human"):
        pass

    def close(self):
        pass
