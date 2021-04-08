from ray import ObjectId
from typing import Dict, Iterable, Union

# These types are used throughout the action bank
ActionID = str
Action = Dict[str, Any]
ActionSet = Dict[str, Action]

ExperimentID = str
Experiment = Union[ObjectId,ExperimentID]
AgentID = str
Agent = Union[ObjectId,AgentID]
