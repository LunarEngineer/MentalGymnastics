"""Reward functions for Mental Gym.

This file contains basic reward functions used in the Mental Gym environment.
"""
import numpy as np
from mentalgym.utils.types import ExperimentSpace


def basic_reward(
    experiment_space: ExperimentSpace,
    function_space: FunctionSpace
) -> float:
    """Calculates simple reward for an experiment.

    Parameters
    ----------
    experiment_space: ExperimentSpace
        This is the 

    This reward function provides a simple reward that approaches
    one as the distance between the sink and the closest node
    approaches zero.
    """
    # TODO: Calculate distance.
    d = None
    # Reward calculation
    r = np.exp(-d)
    return r
