from ray import ObjectId
from typing import Union, Iterable

ActionID = str
Action = Union[ObjectId,ActionID] (inherited either from Ray or PyArrow, shouldn't matter)
ActionSet = Iterable[Action]
ExperimentID = str
Experiment = Union[ObjectId,ExperimentID]
ExperimentCoordinate = Iterable[float]
ActionBank: Iterable[Action]
AgentID = str
Agent = Union[ObjectId,AgentID]
