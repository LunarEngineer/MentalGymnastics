"""Contains the function bank class and supporting code."""
from __future__ import annotations

import json
import os
import pandas as pd

from mentalgym.types import Function, FunctionSet

# This import will be used to fill the atomic function set
# This will change, dependent on the code that's introduced
#   for the atomic functions and composed functions.


class FunctionBank():
    """Builds and tracks Functions and their history.

    This is a class, used by the Mental Gymnastics gym environment
    which is tracking the state of functions created by controller
    agents operating in the environment.

    It has methods for creating, storing, updating, and querying
    functions.

    The functions are stored locally in a dictionary of dictionaries.
    These dictionaries are created via the .build function, are
    updated by the .update function, are pruned by the .prune

    These dictionaries are mirrored to local disk when calling
    _save_bank() as a json (check)

    Parameters
    ----------
    function_bank_directory: str = '.function_bank'
        The directory to store / load the function bank from.
    sampling_function: Optional[Callable] = None
        The default sampling function. This is any function
        that takes in a Pandas dataframe and returns a sampled
        dataframe. This function is used when calling .sample().
        This defaults to uniform random the size of the function
        manifest.
    Methods
    -------
    query(): Needs testing, finished documentation
        Returns information for queried functions
    _query(): Needs testing, finished documentation
        Calls .query on the Pandas representation of the functions.
    sample()
        Returns a sample of functions.
        Using this to build the set of functions is equivalent to shaping
        function space.
    prune()
        Removes functions from the function bank.
        This is used to prune dead-end elements of the search space.
    build()
        Returns a composed function from input.
    update()
        Update an function.
    _build_bank()
        Builds an function bank from a local directory.
    _save_bank()
        Stores the function bank information locally.
    """

    def __init__(
        self,
        function_bank_directory: str = ".function_bank"
    ):
        self._function_bank_directory = function_bank_directory
        self._function_manifest = self._build_bank()
        raise NotImplementedError

    def _build_bank(self) -> FunctionSet:
        """Build function bank from local directory.

        Read a json document to build the function manifest.

        This function can get smarter, but this works as a stopgap.

        Returns
        -------
        function_manifest: FunctionSet
            This is a validated function manifest
        """
        # TODO: Get some default atomics up for tests
        atomic_functions = {
            None
        }
        # First, load in the manifest
        manifest_file = os.path.join(
            self._function_bank_directory,
            '.manifest'
        )
        with open(manifest_file,'r') as f:
            function_manifest = json.loads(f.readlines())
        # Then, do any validation necessary for those functions
        def validate_functions(function_manifest:FunctionSet):
            # This will be abstracted
            return function_manifest
        return validate_functions(function_manifest)

    def _save_bank(self) -> None:
        """Save function bank to local directory.

        Dump the function manifest to disk.
        """
        manifest_file = os.path.join(
            self._function_bank_directory,
            '.manifest'
        )
        # This writes self._function_manifest to json
        with open('.manifest','w') as f:
            f.write(json.dumps(self._function_manifest))

    def query(
        self,
        function_id: str,
        **kwargs
    ) -> Function:
        """Return function information.

        This returns the function identified by the function id.

        Todo: Should this contain a secondary parameter to subset the function?

        Parameters
        ----------
        function_id: str
            The string identifier for the function.
        **kwargs
            Key word arguments

        Returns
        -------
        function: Function

        Examples
        --------
        >>> steve = FunctionBank()
        >>> steve.query('functionid')
        Representationoffunction
        """
        return self._function_manifest[function_id]

    def _query(self, query_string: Optional[str] = None) -> pd.DataFrame:
        """Return filtered function set.

        This constructs a Pandas DataFrame from the function manifest,
        filters it according to the query_str parameter, and returns
        the subset.

        Parameters
        ----------
        query_str: Optional[str] = None
            An empty string by default, this is a query string
            that meets the format used by pandas .query method.

        Returns
        -------
        filtered_function_set: pandas.DataFrame
            This is the function manifest, filtered by a query string

        Examples
        --------
        >>> # Here is code that makes a default manifest which
        >>> #   has an 'accuracy' feature in the dataset and
        >>> #   contains an function with an id of 'steve'.
        >>> # There are *three* functions in this set, two of
        >>> #   which have accuracy over 50%
        >>> ab = FunctionBank()
        >>> act = ab._query('id=="steve"')
        >>> isinstance(act,pd.DataFrame)
        True
        >>> act.nrow
        1
        >>> act_set = ab._query('accuracy>0.5')
        >>> act_set.nrow
        2
        """
        function_frame = pd.DataFrame.from_dict(self._function_manifest)
        return function_frame.query(query_str)
