from ray import ObjectId
from typing import Any, Dict, Iterable, Union

# These types are used throughout the function bank
FunctionID = str
Function = Dict[str, Any]
FunctionSet = Iterable[Dict[str, Function]]

ExperimentID = str
Experiment = Union[ObjectId, ExperimentID]
AgentID = str
Agent = Union[ObjectId, AgentID]
