"""Typing for Mental Environment concepts and variables.

The types below are used throughout the environment.
"""
from ray import ObjectId
from numpy.typing import ArrayLike
from typing import Any, Dict, Iterable, Union
from functionbank import FunctionBank

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

####################################################################
#                   Observation Space Typing                       #
####################################################################
# The ExperimentSpace is a component of the observation space
# It contains a dictionary of 
ExperimentSpace = Dict[str, ArrayLike]
ExperimentSpaceContainer = Dict[str, ArrayLike]
####################################################################
#                    Function Metric Typing                        #
####################################################################
# Functions, when used, log their results to the Function Bank
# They record the ExperimentID (random string), the EnvironmentID
#   (proxy for AgentID)
ExperimentID = str