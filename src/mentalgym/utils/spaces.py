"""Contains functions for working with spaces.

This has functions for working with the experiment space, function
space, and the built in Gym spaces.
"""
from numpy.typing import ArrayLike

def make_experiment_space(
    function_bank: FunctionBank,
    min_loc: ArrayLike[float] = np.array([0, 0]),
    max_loc: ArrayLike[float] = np.array([100, 100]),
    buffer: float = .01
):
    """Create default experiment space from a function bank.

    This creates a default experiment space with nothing but the
    input and output nodes.

    function_bank: FunctionBank
        This is an object which can be queried for function information
    min_loc: ArrayLike[float] = np.array([0, 0])
        This is an array of 'minimum' locations for every axis in the
        experiment space. This (in a 2d space) would be [x_min, y_min]
    max_loc: ArrayLike[float] = np.array([100, 100])
        This is an array of 'maximum' locations for every axis in the
        experiment space. This (in a 2d space) would be [x_max, y_max]
    buffer: float = .01
        This is a float buffer which pads the nodes away from the
        edges of experiment space.
    """
    # These are Pandas DataFrames
    input_nodes = function_bank._query('type=="input"')
    output_nodes = function_bank._query('type=="output"')
    # We're going to create locations for input.
    # They're going to be uniformly distributed over the first axis.
    # A 'buffer' is added to pad the nodes away from the edge.
    num_inputs = input_nodes.shape[0]
    input_locs = np.linspace(min_loc[0])