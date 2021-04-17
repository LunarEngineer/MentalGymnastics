"""Reward functions for Mental Gym.

This file contains basic reward functions used in the Mental Gym environment.
It contains a small utility to make it easier to build rewards programatically.
"""
import numpy as np
from mentalgym.types import ExperimentSpace, FunctionSet
from scipy.spatial import cKDTree
from typing import Iterable, Union, Callable, Optional

# This is shared by all the reward functions
__param_str__ = """
    Parameters
    ----------
    experiment_space_container: pd.DataFrame
        This has columns of:
        * id,
        * type,
        * input,
        * exp_loc_0,
        * exp_loc_1
    function_set: FunctionSet
        This is an iterable of functions used in the current run.
    reward_set: Optional[Iterable[Union[str, Callable]]]
        The set of rewards to build. Each of these will be
        superimposed. Defaults to ['monotonic']
    score: float
        The output of the scoring function.

    Returns
    -------
    reward: ArrayLike
        The reward generated from the current state.
        This is a Numpy array of floats.
"""

def build_reward_function(
    experiment_space_container: pd.DataFrame,
    function_set: FunctionSet,
    reward_set: Iterable[Union[str, Callable]] = ['monotonic'],
    score: float = 0
) -> :
    f"""Creates composite reward.

    Uses passed callables or strings representing default
    functions to calculate reward from an experiment space
    or a function space.

    {__param_str__}

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
    experiment_space_container: pd.DataFrame,
    function_set: FunctionSet,
) -> float:
    f"""Calculates simple reward for an experiment.

    This reward function provides a simple reward that approaches
    one as the distance between the sink and the closest node
    approaches zero. This only rewards for composed actions.

    {__param_str__}
    Examples
    --------
    >>> # Load in a testing function bank
    >>> from mentalgym.utils.data import function_bank
    >>> # Grab the utilities to create some experiment data.
    >>> from mentalgym.utils.spaces import refresh_experiment_container
    >>> from mentalgym.utils.spaces import append_to_experiment
    >>> # Bring in the reward function
    >>> from mentalgym.utils.reward import monotonic_reward
    >>> container = refresh_experiment_container(function_bank)
    >>> # Now, take the composed functions and add them to the experiment
    >>> composed_funcs = function_bank.query('type=="composed"')
    >>> composed_iter = [
    ...    row.to_dict() for
    ...    ind, row in composed_funcs.iterrows()
    ... ]
    >>> # This adds two nodes to the container.
    >>> extended_container = append_to_experiment(
    ...     container,
    ...     composed_iter,
    ...     function_bank
    ... )
    >>> # Here you can see the closest composed node has a
    >>> #   distance of sqrt(25**2 + 50**2)
    >>> extended_container
             id      type                 input  exp_loc_0  exp_loc_1
    0  column_0    source                  None        0.0        0.0
    1  column_1    source                  None       50.0        0.0
    2  column_2    source                  None      100.0        0.0
    3    output      sink                  None        0.0      100.0
    0     steve  composed            [column_0]       25.0       50.0
    1       bob  composed  [column_0, column_1]       50.0       75.0
    >>> monotonic_reward(extended_container, composed_iter)
    array([5.27473208e-25])
    """
    # Get the names of all the location fields in the experiment
    #   space.
    location_fields = [
        _ for
        _ in experiment_space_container.columns
        if _.startswith("exp_loc")
    ]
    # Separate out the *sink* nodes.
    sink_locations = experiment_space_container.query(
        'type=="sink"'
    )[location_fields]
    # Separate out the *non-sink* nodes
    non_sink_locations = experiment_space_container.query(
        'type not in ["sink", "source"]'
    )[location_fields]
    # Build a KD-tree from the non-sink nodes.
    # TODO: Consider caring about the arguments for this.
    # Allowing control over this tree would require more effort
    #   than it's likely worth.
    non_sink_tree = cKDTree(non_sink_locations)
    # Query the sink node. This returns the distance and iloc.
    d, _ = non_sink_tree.query(sink_locations)
    # Reward calculation
    r = np.exp(-d)
    return r


def connection_reward(
    experiment_space_container: pd.DataFrame,
    function_set: FunctionSet,
) -> float:
    f"""Calculates connection reward for an experiment.

    This reward function provides a small constant reward if any
    nodes in the net are composed functions.

    {__param_str__}

    """
    # TODO: Query function space for composed actions.
    # If there are composed functions, return a reward of 10.
    connected = False
    return connected * 10.

def linear_completion_reward(
    experiment_space_container: pd.DataFrame,
    function_set: FunctionSet,
    score: float = 0
) -> float:
    f"""Calculates completed net reward for an experiment.

    This reward is of the form $mx + b$ where x is the output of
    the scoring function. (Think accuracy from 0 to 1.) m is a
    scalar multiplier and b is a small bias constant reward.

    {__param_str__}

    Parameters
    ----------
    score: float
        The output of the scoring function, used here as x.
    """
    x = score
    m = 100.
    b = 20.
    return m*x + b
