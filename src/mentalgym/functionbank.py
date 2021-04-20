"""Contains the function bank class and supporting code."""
from __future__ import annotations

import json
import os
import pandas as pd

# TODO: Uncomment / change this when atomic function information more fleshed out.
# from mentalgym.atomic_functions import base_atomic
base_atomic = {}
from mentalgym.types import Function, FunctionSet
from mentalgym.utils.data import dataset_to_functions
from mentalgym.utils.sampling import softmax_score_sample
from mentalgym.utils.spaces import prune_function_set
from mentalgym.utils.validation import validate_function_bank
from typing import Callable, Optional

#TODO Track function integer index as well with function bank.

class FunctionBank():
    """Builds and tracks Functions and their history.

    This is a class, used by the Mental Gymnastics gym environment,
    which is tracking the state of functions created by controller
    agents operating in the environment.

    It has methods for creating, storing, updating, and querying
    functions.

    The functions are stored internally as an iterable container of
    function representations. This data structure is mirrored to
    disk when calling ._save_bank and refreshed from disk when
    calling ._load_bank.

    When this function bank is instantiated it will check to see if
    a function bank exists at the location specified. If there is a
    function bank this will read that in, elsewise it will create a
    default function bank.

    Function banks store information about input, output, composed,
    and atomic functions. A single function bank represents a single
    modeling dataset. Composed functions within the function space
    are descendants of the input features, as each composed function
    added must be composed from functions already within the space.

    The 'only a single dataset' restriction is a soft restriction.
    Any dataset with the same *features* can use the same function
    bank. Concept drift is likely to render specific functions (and
    descendants) defunct.

    Any functions stored within the bank that require persistent
    artifacts (weights, hyperparameters, etc...) are persisted
    in the function bank directory. The names of the subdirectories
    are an exact match for the id and are case insensitive.

    Function Bank Directory Structure:
    ${function_bank_directory}\
        ${composed_function_1}\
            artifact_{1_1}
            ...
            artifact_{n_1}
        ...
        ${composed_function_2}\
            artifact_{1_2}
            ...
            artifact_{n_2}

    Parameters
    ----------
    modeling_data: pd.DataFrame
        This is the modeling dataset. This is used both to store
        information on the modeling data and to ensure that the
        function bank isn't used with another dataset.
    function_bank_directory: str = '.function_bank'
        The directory to store / load the function bank from.
    dataset_scraper_function: Optional[Callable] = None
        This is the default function which will scrape the modeling
        dataset. Replace this to add extra input nodes, such as
        statistics, into the modeling input dataset.
    sampling_function: Optional[Callable] = None
        The default sampling function. This is any function
        that takes in a Pandas DataFrame and returns a sampled
        DataFrame. This function is used when calling .sample().
        This defaults to uniform random the size of the function
        manifest.
    pruning_function: Optional[Callable] = None
        The default pruning function. This is any function that
        takes in a Pandas DataFrame and returns a boolean mask
        as long as that dataset.
    population_size: Optional[int] = 1000
        This is the number of functions kept 'alive' at any time.
        Living functions can be drawn via the sampling function.
        This has the effect of truncating evolutionary deadends.

    Methods
    -------
    query(): Needs testing, finished documentation
        Returns information for queried functions
    _query(): Needs testing, finished documentation
        Calls .query on the Pandas representation of the functions.
    sample()
        Returns a sample of functions.
        Using this to build the set of functions is equivalent to shaping
        function space. This is how you define your exploration strategy.
    prune()
        Removes functions from the function bank.
        This is used to prune dead-end elements of the search space.
        All this does is 'disable' actions. A disabled action cannot
        be drawn during an episode.
    build()
        Returns a composed function from input.
    score()
        This takes an Experiment Space and a score, updating statistics
        for those actions. If this score (i.e. accuracy) doesn't
        already exist then it is added to the dataset. Default
        functionality assumes that scores are better if higher.
        This stores current hyperparameters for actions when they are
        created.
    _build_bank()
        Builds an function bank from a local directory.
    _save_bank()
        Stores the function bank information locally.
    """
    def __init__(
        self,
        modeling_data: pd.DataFrame,
        function_bank_directory: str = ".function_bank",
        dataset_scraper_function: Optional[Callable] = None,
        sampling_function: Optional[Callable] = None,
        pruning_function: Optional[Callable] = None,
        population_size: Optional[int] = 1000
    ):
        ############################################################
        # This sets up the engine of the function bank using user  #
        #   passed functionality if appropriate and default if not.#
        ############################################################
        # This is the function that will make input/output nodes.
        if dataset_scraper_function is not None:
            self._dataset_scraper_function = dataset_scraper_function
        else:
            self._dataset_scraper_function = dataset_to_functions
        # This is the default sampling function used to fill the
        #   agent's action bank. This can be considered an exploration
        #   strategy favoring BFS or DFS. This could be Beam search.
        # Choose your own adventure. The default is a quasi-probabilistic
        #   DFS with a softmax weighted probability.
        if sampling_function is not None:
            self._sampling_function = sampling_function
        else:
            self._sampling_function = softmax_score_sample
        # This is the default pruning function used to mark nodes
        #   as *dead*. These nodes will not be drawn again
        if pruning_function is not None:
            self._pruning_function = pruning_function
        else:
            self._pruning_function = prune_function_set
        ############################################################
        #           This section stores hyperparameters.           #
        ############################################################
        self._function_bank_directory = function_bank_directory
        self._population_size = population_size
        self._data = modeling_data
        ############################################################
        #              This section starts the engine.             #
        ############################################################
        # In this step the dataset is scraped to build input/output
        #   nodes and atomic and composed functions are read from
        #   disk if present.
        self._function_manifest = self._build_bank()
        raise NotImplementedError

    ################################################################
    # These following functions are used to persist the function   #
    #   bank. These are used to read / write from the storage      #
    #   location, which by default is the local directory where    #
    #   the function bank was instantiated.                        #
    ################################################################
    def _build_bank(
        self,
    ) -> FunctionSet:
        """Build function bank.

        Read a json document to build the function manifest.
        If one does not exist a default manifest will be created.
        The passed dataset ensures that input / output are created
        and validated.

        This does some basic validation to ensure the modeling data
        matches existing dataset information and that all the
        functions appear to be valid functions.

        Returns
        -------
        function_manifest: FunctionSet
            This is a validated function manifest which the bank
            will use to build / return functions.
        """
        # The filepath for the function manifest.
        manifest_file = os.path.join(
            self._function_bank_directory,
            '.manifest'
        )
        # This function will build a default json document if one
        #   does not exist
        if not os.path.exists(manifest_file):
            build_default_function_bank(self._data)
            self._save_bank()
        # Read in the manifest
        with open(manifest_file,'r') as f:
            function_manifest = json.load(f)
        # Then, do any validation necessary for those functions
        assert validate_function_bank(function_manifest)
        # Finally, give up the goods.
        return function_manifest

    def _save_bank(self) -> None:
        """Save function bank to local directory.

        Dump the function manifest to disk.
        """
        manifest_file = os.path.join(
            self._function_bank_directory,
            '.manifest'
        )
        # This writes self._function_manifest to json
        with open(manifest_file, 'w') as f:
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
        return function_frame.query(query_string)

    def sample(self, n, include_base: bool = False):
        """Return n favorable actions.

        If `include_base` is set to True this will also return
        input, output, and atomic actions as counted among the n.
        This should be called at the beginning of episodes.
        """
        # TODO: Call the self._sampling_func on the function manifest
        raise NotImplementedError
def build_default_function_bank():
    """Creates a Function Set composed of atomic functions.
    """
    raise NotImplementedError