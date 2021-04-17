"""Reward functions for Mental Gym.

This file contains basic reward functions used in the Mental Gym environment.
It contains a small utility to make it easier to build rewards programatically.
"""
import numpy as np
from mentalgym.utils.types import ExperimentSpace, FunctionSpace
from typing import Iterable, Union, Callable, Optional


def build_reward_function(
    experiment_space: ExperimentSpace,
    function_space: FunctionSpace,
    reward_set: Iterable[Union[str, Callable]] = ['monotonic'],
    score: float = 0
) -> :
    """Creates composite reward.

    Uses passed callables or strings representing default
    functions to calculate reward from an experiment space
    or a function space.

    Parameters
    ----------
    experiment_space: ExperimentSpace
        This is an experiment space maintained by the gym.
    function_space: FunctionSpace
        This is a function space maintained by the gym.
    reward_set: Optional[Iterable[Union[str, Callable]]]
        The set of rewards to build. Each of these will be
        superimposed. Defaults to ['monotonic']
    score: float
        The output of the scoring function.

    Returns
    -------
    reward: float
        The reward generated from the current state.

    Examples
    --------
    >>> # Fill out examples when experiment space structure 
    """
    # Give default rewards of monotonic if unspecified
    if reward is None:
        reward = ['monotonic']
    # Initialize reward to zero
    reward = 0
    # Loop through the iterable of strings / functions
    reward_dict = {
        "monotonic": monotonic_reward,
        "connection": connection_reward,
        "completion": linear_completion_reward
    }
    # Helper function to trigger for completion reward
    def requires_score(x: str) -> bool:
        return x == "completion"

    for r in reward_set:
        # If it's a string in the reward dict, use *that* function.
        if r in reward_dict:
            # If the reward requires a score
            if requires_score(r):
                # Pass the score
                reward += reward_dict[r](
                    experiment_space,
                    function_space,
                    score
                )
            else:
                # Otherwise only the spaces
                reward += reward_dict[r](
                    experiment_space,
                    function_space
                )
        # If it's a function, use *that* function.
        else if isinstance(r, Callable):
            reward += r(
                experiment_space,
                function_space
            )
        # Otherwise *blow up*! Kapow!
        raise Exception(f"{r} is not a valid reward.")
    # Then, return reward
    return reward

def monotonic_reward(
    experiment_space: ExperimentSpace,
    function_space: FunctionSpace
) -> float:
    """Calculates simple reward for an experiment.

    This reward function provides a simple reward that approaches
    one as the distance between the sink and the closest node
    approaches zero.
    """
    # TODO: Calculate distance.
    d = None
    # Reward calculation
    r = np.exp(-d)
    return r

def connection_reward(
    experiment_space: ExperimentSpace,
    function_space: FunctionSpace
) -> float:
    """Calculates connection reward for an experiment.

    This reward function provides a small constant reward if any
    nodes in the net are composed functions.
    """
    # TODO: Query function space for composed actions.
    # If there are composed functions, return a reward of 10.
    connected = False
    return connected * 10.

def linear_completion_reward(
    experiment_space: ExperimentSpace,
    function_space: FunctionSpace,
    score: float = 0
) -> float:
    """Calculates completed net reward for an experiment.

    This reward is of the form $mx + b$ where x is the output of
    the scoring function. (Think accuracy from 0 to 1.) m is a
    scalar multiplier and b is a small bias constant reward.

    Parameters
    ----------
    score: float
        The output of the scoring function, used here as x.
    """
    x = score
    m = 100.
    b = 20.
    return m*x + b
