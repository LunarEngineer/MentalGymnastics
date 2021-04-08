"""Contains the action bank class and supporting code."""
import pandas as pd
from __futures__ import annotation
from utils.types import Action, ActionSet

class ActionBank():
    """Builds and tracks actions and their history.

    This is a class, used by the Mental Gymnastics gym environment
    which is tracking the state of actions created by controller
    agents operating in the environment.

    It has methods for creating, storing, updating, and querying
    actions.

    The actions are stored locally in a dictionary of dictionaries.
    These dictionaries are created via the .build function, are
    updated by the .update function, are pruned by the .prune

    These dictionaries are mirrored to local disk when calling
    _save_bank() as a json (check)

    Parameters
    ----------
    action_bank_directory: str = '.action_bank'
        The directory to store / load the action bank from.

    Methods
    -------
    query()
        Returns information for queried actions
    _query()
        Calls .query on the Pandas representation of the actions.
    sample()
        Returns a sample of actions
    prune()
        Removes actions from the action bank.
    build()
        Returns a composed action from input.
    update()
        Update an action.
    _build_bank()
        Builds an action bank from a local directory.
    _save_bank()
        Stores the action bank information locally.
    """

    def __init__(
        self,
        action_bank_directory: str = ".action_bank"
    ) -> ActionBank:
        self._action_bank_directory = action_bank_directory
        self._action_manifest = self._build_bank()
        raise NotImplementedError

    def _build_bank(self) -> ActionSet:
        """Build action bank from local directory.

        Read a json document to build the action manifest.

        This function can get smarter, but this works as a stopgap.

        Returns
        -------
        action_manifest: ActionSet
            This is an action manifest
        """
	bank = None
        action_dir = self._action_bank_query
	raise NotImplementedError
	return bank

    def _save_bank(self) -> None:
	"""Save action bank to local directory.

	Dump the action manifest to disk.
	"""
	raise NotImplementedError

    def query(self, action_id: str) -> Action:
        """Return action information.

        This returns the action identified by the action id. 

	Todo: Should this contain a secondary parameter to subset the action?

	Parameters
	----------
	action_id: str
	    The string identifier for the action.

	Returns
	-------
	action: Action

	Examples
	--------
	>>> steve = ActionBank()
	>>> steve.query('actionid')
	Representationofaction
        """
	return self._action_manifest[action_id]

    def _query(self, query_string: Optional[str] = None) -> pd.DataFrame:
	"""Return filtered action set.

	This constructs a Pandas DataFrame from the action manifest,
	filters it according to the query_str parameter, and returns
	the subset.

	Parameters
	----------
	query_str: Optional[str] = None
	    An empty string by default, this is a query string
	    that meets the format used by pandas .query method.

	Returns
	-------
	filtered_action_set: pandas.DataFrame
	    This is the action manifest, filtered by a query string

	Examples
	--------
	>>> # Here is code that makes a default manifest which
	>>> #   has an 'accuracy' feature in the dataset and
	>>> #   contains an action with an id of 'steve'.
	>>> # There are *three* actions in this set, two of
	>>> #   which have accuracy over 50%
	>>> ab = ActionBank()
	>>> act = ab._query('id=="steve"')
	>>> isinstance(act,pd.DataFrame)
	True
	>>> act.nrow
	1
	>>> act_set = ab._query('accuracy>0.5')
	>>> act_set.nrow
	2
	"""
	action_frame = pd.DataFrame.from_dict(self._action_manifest)
	return action_frame.query(query_str)
