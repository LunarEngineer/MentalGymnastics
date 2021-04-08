"""Contains the action bank class and supporting code."""

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
    ):
        self._action_bank_directory = action_bank_directory
        self._build_bank(action_bank_directory)
        raise NotImplementedError

    def _build_bank(self):
        """Build action bank from local directory.

        Read a json document to build the action_manifest dict.

        This function can get smarter, but this works as a stopgap.

        Returns
        -------
        action_manifest: Dict[str,Dict[str,Any]]
            This is an action manifest
        """

    def query(self):
        """Return action information.

        This runs .query on a Pandas DataFrame representation of the
        actions in the action bank. 

        """