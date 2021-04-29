"""Typing for Mental Environment concepts and variables.

The types below are used throughout the environment.
"""
import pandas as pd
# from mentalgym.functions import (
#     AtomicFunction,
#     ComposedFunction
# )
# from mentalgym.functions.composed import ComposedFunction
from numpy.typing import ArrayLike
from typing import Any, Dict, Iterable, Type, Union
####################################################################
#                          Function Typing                         #
####################################################################
# These types are used to describe a Function data structure.
# Functions are keyed by a string unique identifier
FunctionID = str
# A Function is a dictionary which has *at minimum* keys required
#   to create a composed function
# Those keys include: id, type, input, location
Function = Dict[str, Any]
# A Function Set is a container, with actions, that can be iterated
#   over.
FunctionSet = Iterable[Dict[str, Function]]
# A Function Bank is a Pandas DataFrame, or a FunctionSet.
FunctionBank = Union[pd.DataFrame, FunctionSet]
# A Function object is the uninstantiated class of a Function.
# This is commented out below to prevent circular imports, but is
#   a concept that is carried through the environment.
# FunctionObject = Union[Type[AtomicFunction], Type[ComposedFunction]]
####################################################################
#                   Observation Space Typing                       #
####################################################################
# The ExperimentSpace is a component of the observation space
# It contains a dictionary of
ExperimentSpace = Union[Dict[str, ArrayLike], pd.DataFrame]
ExperimentSpaceContainer = Dict[str, ArrayLike]
####################################################################
#                    Function Metric Typing                        #
####################################################################
# Functions, when used, log their results to the Function Bank
# They record the ExperimentID (random string), the EnvironmentID
#   (proxy for AgentID)
ExperimentID = str