"""Contains the function bank class and supporting code."""
from __future__ import annotations
from collections import deque

import numpy as np
import os
import pandas as pd
import pickle

from mentalgym.types import Function, FunctionSet
from mentalgym.utils.function import dataset_to_functions, make_function
from mentalgym.utils.sampling import softmax_score_sample
from mentalgym.utils.spaces import (
    prune_function_set,
    build_default_function_space
)
from mentalgym.utils.validation import validate_function_set
from numbers import Number
from pandas.core.algorithms import isin
from typing import Callable, Iterable, Optional, Union


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
    calling ._load_bank. Any *objects* in the function space are
    persisted in folders collocated with the function bank.

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

    API usage for the *function objects* is still up in the air,
    but they should be able to be instantiated and called in order
    to:

    * 'Call' a function
    * Save and load weights of a function.

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

    Function Representation keys and values:
    
    * i: This is an ascending integer id for the functions.
    * id: This is a string representation of the function.
    * type: This is a string representation of the type of the
        function. This falls into the set of values ['input',
        'output', 'atomic', 'composed']
    * input: This is a iterable containing ids of nodes which this
        function depends on.
    * living: This is a boolean which specifies whether this
        function can be drawn at sample time.
    * forward: This is the callable representation of the
        function, and each type of function has a different
        representation:
        * atomic: This callable is a factory function which can take
            an arbitrary number of nodes as input to create a
            composed function and return the composed callable.
        * composed: This callable reproduces the atomic action which
            produced it. This will load and save weights
            appropriately in the forward.
        * input: This callable returns the values of the column of
            the *current sample* of the internal dataset.
        * output: This callable returns the 

    The function representation in the function bank *also* carries
        a set of fields for every scoring function. This means that
        if you 'score' with a function called 'x' then there will be
        fields 'scores', which contains an array with
        `buffer_length` values.

    Parameters
    ----------
    modeling_data: pd.DataFrame
        This is the modeling dataset. This is used both to store
        information on the modeling data and to ensure that the
        function bank isn't used with another dataset.
    target: Optional[str] = None
        This is the column name 
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
        This defaults to a weighted sampling that favors more
        effective actions.
    pruning_function: Optional[Callable] = None
        The default pruning function. This is any function that
        takes in a Pandas DataFrame and returns a boolean mask
        as long as that dataset.
    population_size: Optional[int] = 1000
        This is the number of functions kept 'alive' at any time.
        Living functions can be drawn via the sampling function.
        This has the effect of truncating evolutionary deadends.
    verbose: bool = False
        This tells the function bank to produce verbose output.
    force_refresh: bool = False
        This is an optional flag that you may turn on to force the
        function bank to purge the existing function bank directory.
        This will delete everything in that directory and start a
        fresh function bank.

    Methods
    -------
    query(query_str) -> pd.DataFrame
        Overloads Pandas .query on the internal function bank.
    append()
    sample(n, include_base, random_state) -> FunctionSet:
        Returns a sample of functions.
    prune(random_state): 
        Removes functions from the function bank.
        This is used to prune dead-end elements of the search space.
        All this does is 'disable' composed actions. A disabled action
        cannot be drawn during an episode.
    score(function_set, score, score_name)
        This takes an iterable of Functions (various options) and
        updates the scoring information for those functions.
    idxmax() -> int:
        Returns the maximum index from the function bank.
    to_df() -> pandas.DataFrame:
        This returns the function manifest as a DataFrame.
    _build_bank()
        Builds an function bank from a local directory.
    _save_bank()
        Stores the function bank information locally.
    """
    def __init__(
        self,
        modeling_data: pd.DataFrame,
        target: Optional[str] = None,
        function_bank_directory: str = ".function_bank",
        dataset_scraper_function: Optional[Callable] = None,
        sampling_function: Optional[Callable] = None,
        pruning_function: Optional[Callable] = None,
        population_size: Optional[int] = 1000,
        verbose: bool = False,
        force_refresh: bool = False
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
        #             This section stores parameters.              #
        ############################################################
        self._function_bank_directory = function_bank_directory
        self._population_size = population_size
        self._data = modeling_data
        self._target = target
        self._verbose = verbose
        self._force_refresh = force_refresh

        ############################################################
        #              This section starts the engine.             #
        ############################################################
        # In this step the dataset is scraped to build input/output
        #   nodes and atomic and composed functions are read from
        #   disk if present.
        self._function_manifest = self._build_bank()

    ################################################################
    #                   Experimental Analysis                      #
    ################################################################
    def analysis(self):
        # TODO
        # 1. Persist a static function bank to a dataset somewhere.
        # 2. Compute an aggregate dataset that showcases living
        #   functions, their type, complexity, and performance over
        #   the last hundred episodes.
        #   This is then added to the experiment space.
        #   Persist this aggregate dataset.
        raise("Baked in analysis routine")

    def function_statistics(
        self,
        ids: Optional[Union[str, Iterable[str]]] = None,
        extended: bool = False
    ) -> pd.DataFrame:
        """Compute statistics for functions.

        This will compute summary statistics for functions in the
        function bank and will return the summary statistics for
        every scoring function used.

        If requested this will also return the score buffers for
        each action indicating 'in the last `buffer length` episodes
        they were used these were the scores'. Note that two functions
        values for 'buffer 1' are not necessarily correlated in time.

        This will fill buffers that haven't been used with NaN. When
        computing statistics this will ignore NaN.

        Parameters
        ----------
        ids: Optional[Union[str, Iterable[str]]]
            This is an optional id or set of ids to limit
            computation.
        extended: bool = False
            If False this will only return summary statistics for
            each score type for each function:
            * mean
            * median

        Returns
        -------
        function_statistics: pd.DataFrame
            A DataFrame with as many rows as there are functions
            queried for (all if no ids passed) containing aggregate
        """
        df = self.to_df()
        if ids is not None:
            df = df.query(f'id == {ids}').reset_index(drop = True)
        score_ids = df[['id']]
        score_cols = [_ for _ in df if _.startswith('score')]
        score_dfs = []
        for score_col in score_cols:
            # This assumes that all the deques have the same length.
            # This should be a good assumption.
            # This gets the maximum length for the buffers
            score_len = df[score_col].iloc[0].maxlen
            # This creates column names that long.
            score_names = [f'{score_col}_{_}' for _ in range(score_len)]
            # This, then, creates an empty dataset just as long
            score_arr = np.empty((df.shape[0], score_len))
            score_arr[:] = np.nan
            score_df = pd.DataFrame(
                score_arr,
                columns = score_names
            )
            # This pulls out the buffers
            scores = pd.DataFrame.from_records(df[score_col])
            #   and finds out how long each of the buffers is.
            scores_len = scores.shape[1]
            # Then shove the real data on top.
            score_df.iloc[:,:scores_len] = scores
            # Finally, stuff this in the scoring list.
            score_dfs.append(score_df)
        # score_dfs is now a list as long as the number of scoring
        #   functions with *non-aggregate* data. These are aggregated
        #   to make the summary statistics.
        # We want the ids to be up front.
        output_dfs = [score_ids]
        # Now, walk down the scoring datasets, summarize them, append
        #   the summary to the main dataset, and then append the
        #   non-aggregate data if requested.
        for score_col, score_df in zip(score_cols, score_dfs):
            # This transposes the dataset so that the statistics are
            #   summarized by function
            summary_df = score_df.T.describe().T
            # This then adds the *name* of the scoring function to
            #   the aggregate column names.
            summary_df.columns = [
                f'{score_col}_{_}' for _ in summary_df.columns
            ]
            # This if/then branch decides what to keep and appends
            #   the data to the primary dataset.
            if extended:
                output_dfs += [summary_df, score_df]
            else:
                output_dfs += [summary_df]
        # This takes everything and jams it together.
        _output = pd.concat(output_dfs,axis=1)
        return _output

    ################################################################
    #                     Mechanics Definitions                    #
    #                                                              #
    # The prune, sample, append, and query methods are used to     #
    #   excise, sample, add, and filter functions in the function  #
    #   bank. The score function is used to update scores in the   #
    #   bank.                                                      #
    ################################################################
    def prune(
        self,
        random_state: Optional[int] = None
    ):
        """Prune the function bank.

        This uses an overloadable function, set at init, which can
        return sampled ids from the function bank DataFrame
        representation.

        Those functions are set to living and the function manifest
        updated.

        Parameters
        ----------
        random_state: Optional[int] = None
            This is an optional random seed used for repeatable
            results.

        Examples
        --------
        >>> # Go look in tests/functionbank/prune_tester
        >>> # There's an example in there.
        """
        # This returns a *keep* style sample of ids from the composed
        #   ids.
        x = self.to_df()
        living_ids: pd.Series = self._sampling_function(
            x = x.query('type=="composed" and living==True'),
            n = self._population_size,
            random_state = random_state
        )
        # This is a boolean mask with True for 'keep this composed'
        prune_id_mask = x.id.isin(living_ids)
        # This is a boolean mask with True for 'this is io or atomic'
        io_mask = x.type.isin(['source', 'sink', 'atomic'])
        # This makes an 'or' from the two above and then negates it.
        # This has the effect of 'masking' only the composed we wish
        #   to prune.
        x.loc[~(prune_id_mask | io_mask),'living'] = False
        status_message = f"""Function Bank Prune:

        Current Function Bank
        ---------------------\n{self.to_df()}

        Population Size
        ---------------\n{self._population_size}

        Sampled IDs to Persist
        ----------------------\n{living_ids}

        Pruning Mask
        ------------\n{prune_id_mask}

        Input/Output/Atomic Mask
        ------------------------\n{io_mask}

        Function Bank Posterior to Prune
        --------------------------------\n{x}
        """
        if self._verbose:
            print(status_message)
        self._function_manifest = x.to_dict(orient = 'records')

    def sample(
        self,
        n: int = False,
        include_base: bool = False,
        random_state: Optional[int] = None
    ) -> FunctionSet:
        """Return up to n favorable actions.

        If `include_base` is set to True this will also start with
        the base input, output, and atomic actions prior to sampling
        actions, meaning that if your input dataset has 100 input
        features and you request `n = 3` you will get 103 Function
        representations returned.

        You can only sample as many composed functions as *exist*
        and so if you only have two composed functions ('steve' and
        'bob') in the function bank and you pass an n of 5 you will
        get 2 functions returned.

        Parameters
        ----------
        n: int
            The maximum number of actions to sample.
        include_base: bool
            Whether or not to frontload the inputs / outputs / etc...
        random_state: Optional[int] = None
            Pass a seed for repeatability.

        Returns
        -------
        sampled_functions: FunctionSet
            This is an iterable of Function representations

        Examples
        --------
        >>> import pandas as pd
        >>> from mentalgym.utils.data import testing_df
        >>> from mentalgym.functionbank import FunctionBank
        >>> from mentalgym.utils.function import make_function
        >>> function_bank = FunctionBank(testing_df)
        >>> composed_functions = [
        ...     make_function(
        ...         function_index = -3,
        ...         function_id = 'steve',
        ...         function_inputs = ['1'],
        ...         function_type = 'intermediate',
        ...         function_location = [1, 1]
        ...     ),
        ...     make_function(
        ...         function_index = -4,
        ...         function_id = 'bob',
        ...         function_inputs = ['1', '2'],
        ...         function_type = 'intermediate',
        ...         function_location = [1, 1, 2]
        ...     )
        ... ]
        >>> function_bank.append(composed_functions)
        >>> pd.DataFrame(function_bank.sample(n=1,random_state=0))
           i   id      type   input  living score_default object hyperparameters
        0  4  bob  composed  [1, 2]    True           [0]   None              {}
        >>> pd.DataFrame(function_bank.sample(n=1,include_base=True,random_state=0))
           i       id      type   input  living score_default                                             object hyperparameters
        0  0   Linear    atomic     NaN    True           [0]  <class 'mentalgym.functions.atomic.Linear.Line...              {}
        1  1     ReLU    atomic     NaN    True           [0]     <class 'mentalgym.functions.atomic.relu.ReLU'>              {}
        2  2  Dropout    atomic     NaN    True           [0]  <class 'mentalgym.functions.atomic.Dropout.Dro...              {}
        3  4      bob  composed  [1, 2]    True           [0]                                               None              {}
        """
        # 0. Checking
        assert n > 0, "Must request a positive number of samples."
        # 1. Turn the function manifest into a DataFrame.
        function_bank = pd.DataFrame(
            self._function_manifest
        )
        # 2. Query the dataset.
        if include_base:
            # If we're including the base set then this gets input,
            #   output, and atomic.
            base_set = [
                'input',
                'output',
                'atomic'
            ]
        else:
            # If we're not including the base set then this gets
            #   nothing.
            base_set = []
        # This will query for all functions which match the expected
        #   types in the base_set list.
        base_functions = function_bank.query(
            f'type in {base_set}'
        )
        # 3. How many functions were requested?
        # Get that many composed functions if they exist.
        # Because all functions come with score_default this will
        #   automatically work.
        composed_ids = self._sampling_function(
            x = function_bank.query('type == "composed"'),
            n = n,
            random_state = random_state
        )
        # 4. Pull that data out of the bank.
        composed_functions = function_bank.query(
            f'id == {composed_ids}'
        )
        # 5. Turn both those into iterables and *smoosh* them.
        base_set = base_functions.to_dict(orient = 'records')
        composed_set = composed_functions.to_dict(orient = 'records')
        # 6. Return the concatenated array.
        return base_set + composed_set

    def append(self, function: Union[FunctionSet, Function]):
        """Appends a function or functions to the bank.

        This will append either a single function representation or
        an iterable of function representations to the Function Bank.

        Parameters
        ----------
        function: Union[FunctionSet, Function]
            This is either a function or iterable of functions.

        Examples
        --------
        >>> import pandas as pd
        >>> from mentalgym.utils.data import testing_df
        >>> from mentalgym.functionbank import FunctionBank
        >>> from mentalgym.utils.function import make_function
        >>> function_bank = FunctionBank(testing_df)
        >>> composed_functions = [
        ...     make_function(
        ...         function_index = -3,
        ...         function_id = 'steve',
        ...         function_inputs = ['1'],
        ...         function_type = 'intermediate',
        ...         function_location = [1, 1]
        ...     ),
        ...     make_function(
        ...         function_index = -4,
        ...         function_id = 'bob',
        ...         function_inputs = ['1', '2'],
        ...         function_type = 'intermediate',
        ...         function_location = [1, 1, 2]
        ...     )
        ... ]
        >>> function_bank.to_df()
           i        id    type  input  living score_default                                             object hyperparameters
        0 -1  column_0  source    NaN    True           [0]                                                NaN            None
        1 -1  column_1  source    NaN    True           [0]                                                NaN            None
        2 -1  column_2  source    NaN    True           [0]                                                NaN            None
        3 -1    output    sink    NaN    True           [0]                                                NaN            None
        4  0    Linear  atomic    NaN    True           [0]  <class 'mentalgym.functions.atomic.Linear.Line...              {}
        5  1      ReLU  atomic    NaN    True           [0]     <class 'mentalgym.functions.atomic.relu.ReLU'>              {}
        6  2   Dropout  atomic    NaN    True           [0]  <class 'mentalgym.functions.atomic.Dropout.Dro...              {}
        >>> function_bank.append(composed_functions)
        >>> function_bank.to_df()
          i        id      type   input  living score_default                                             object hyperparameters
        0 -1  column_0    source     NaN    True           [0]                                                NaN            None
        1 -1  column_1    source     NaN    True           [0]                                                NaN            None
        2 -1  column_2    source     NaN    True           [0]                                                NaN            None
        3 -1    output      sink     NaN    True           [0]                                                NaN            None
        4  0    Linear    atomic     NaN    True           [0]  <class 'mentalgym.functions.atomic.Linear.Line...              {}
        5  1      ReLU    atomic     NaN    True           [0]     <class 'mentalgym.functions.atomic.relu.ReLU'>              {}
        6  2   Dropout    atomic     NaN    True           [0]  <class 'mentalgym.functions.atomic.Dropout.Dro...              {}
        7  3     steve  composed     [1]    True           [0]                                               None              {}
        8  4       bob  composed  [1, 2]    True           [0]                                               None              {}
        """
        # What doctoring needs to be done to the function?
        if isinstance(function, dict):
            function_set = [function]
        else:
            function_set = function
        # 1) Assert that every function *is* a function
        validate_function_set(function_set)
        # 2) *Make* all the functions
        curated_function_set = []
        for i, func in enumerate(function_set):
            if 'id' in func:
                f_id = func['id']
            else:
                f_id = None
            curated_function_set.append(
                make_function(
                    function_index = self.idxmax() + i + 1,
                    function_id = f_id,
                    function_object = func['object'],
                    function_hyperparameters = func['hyperparameters'],
                    function_type = 'composed',
                    function_inputs = func['input'],
                    max_score_len = 100
                )
            )
            i += 1

        self._function_manifest = pd.DataFrame(
            self._function_manifest
        ).append(
            curated_function_set
        ).to_dict(orient='records')

    def query(self, query_string: Optional[str] = None) -> pd.DataFrame:
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
        >>> from mentalgym.utils.data import testing_df
        >>> from mentalgym.functionbank import FunctionBank
        >>> from mentalgym.utils.function import make_function
        >>> function_bank = FunctionBank(testing_df)
        >>> function_bank.query('type=="atomic"')
           i       id    type  input  living score_default                                             object hyperparameters
        4  0   Linear  atomic    NaN    True           [0]  <class 'mentalgym.functions.atomic.Linear.Line...              {}
        5  1     ReLU  atomic    NaN    True           [0]     <class 'mentalgym.functions.atomic.relu.ReLU'>              {}
        6  2  Dropout  atomic    NaN    True           [0]  <class 'mentalgym.functions.atomic.Dropout.Dro...              {}
        >>> function_bank.query('i==-1')
           i        id    type  input  living score_default object hyperparameters
        0 -1  column_0  source    NaN    True           [0]    NaN            None
        1 -1  column_1  source    NaN    True           [0]    NaN            None
        2 -1  column_2  source    NaN    True           [0]    NaN            None
        3 -1    output    sink    NaN    True           [0]    NaN            None
        """
        function_frame = pd.DataFrame.from_dict(self._function_manifest)
        return function_frame.query(query_string)

    def score(
        self,
        function_set: Union[FunctionSet, pd.DataFrame, Iterable[str]],
        score: Union[Number, Iterable[Number]],
        score_name: Union[str, Iterable[str]] = 'default',
    ):
        """Add scoring information to the bank.

        This adds scoring information to the function bank.
        Every function in the bank has a *score buffer* for every
        scoring function which it is measured against.

        Parameters
        ----------
        function_set: Union[FunctionSet, pd.DataFrame]
            This is either an iterable of function representations
            or a DataFrame built from that iterable. It can also
            simply be an iterable of strings representing the
            function ids.
        score: Union[Number, List[Number]]
            This is a *bigger is better* style score. The current
            method of scoring involves taking the arithmetic mean
            of scores for the different scoring functions.
            That means that if a function 'bob' had scores of
            * score_accuracy: [.9, .7, .6]
            * score_complexity: [5, -2,]
        score_name: Union[str, Iterable[str]] = 'default'
            These are the names associated with the scores. If more
            than one score is passed in an iterable this should be
            an iterable as long as the array of scores.

        Examples
        --------
        >>> import pandas as pd
        >>> from mentalgym.utils.data import testing_df
        >>> from mentalgym.functionbank import FunctionBank
        >>> from mentalgym.utils.function import make_function
        >>> function_bank = FunctionBank(testing_df)
        >>> composed_functions = [
        ...     make_function(
        ...         function_index = -3,
        ...         function_id = 'steve',
        ...         function_inputs = ['1'],
        ...         function_type = 'intermediate',
        ...         function_location = [1, 1]
        ...     ),
        ...     make_function(
        ...         function_index = -4,
        ...         function_id = 'bob',
        ...         function_inputs = ['1', '2'],
        ...         function_type = 'intermediate',
        ...         function_location = [1, 1, 2]
        ...     )
        ... ]
        >>> function_bank.append(composed_functions)
        >>> function_bank.to_df()
           i        id      type   input  living score_default                                             object hyperparameters
        0 -1  column_0    source     NaN    True           [0]                                                NaN            None
        1 -1  column_1    source     NaN    True           [0]                                                NaN            None
        2 -1  column_2    source     NaN    True           [0]                                                NaN            None
        3 -1    output      sink     NaN    True           [0]                                                NaN            None
        4  0    Linear    atomic     NaN    True           [0]  <class 'mentalgym.functions.atomic.Linear.Line...              {}
        5  1      ReLU    atomic     NaN    True           [0]     <class 'mentalgym.functions.atomic.relu.ReLU'>              {}
        6  2   Dropout    atomic     NaN    True           [0]  <class 'mentalgym.functions.atomic.Dropout.Dro...              {}
        7  3     steve  composed     [1]    True           [0]                                               None              {}
        8  4       bob  composed  [1, 2]    True           [0]                                               None              {}
        >>> function_bank.score(composed_functions, .6)
        """
        # 0.1) Pretreat the scores:
        if not isinstance(score, Iterable):
            _score = [score]
        else:
            _score = score
        # 0.2) Pretreat the score names:
        if isinstance(score_name, str):
            _score_name = [score_name]
        else:
            _score_name = score_name
        # 0.3) Ensure they are equal length.
        err_msg = f"""Function Scoring Error:

        There is an inconsistency between the number of scores and
        the number of score names.

        Scores: {_score}
        Score Names: {_score_name}
        """
        assert len(_score) == len(_score_name), err_msg
        # 0.4) Now add 'score_' in front of the score names.
        _score_name = [f'score_{_}' for _ in _score_name]
        # 0.5) Make sure the data structure plays nicely.
        # Basically one of a few things can happen.
        # # 1) It has an id property available.
        # This can happen when a FunctionBank or ExperimentSpace are passed.
        try:
            # Assume the id property is an id in the bank.
            function_ids = function_set.id.to_list()
            function_df = self.query(f'id=={function_ids}')
        except AttributeError:
        # # 2) It is an *iterable* of Functions
            try:
                # Assume the id property is an id in the bank.
                function_ids = pd.DataFrame(function_set).id.to_list()
                function_df = self.query(f'id=={function_ids}')
            except AttributeError:
        # # 3) Or it is a string or list of strings
                try:
        # # 4) In which case it had better return a set of results
        #        as long as the query set.
                    # Assume these are ids
                    if isinstance(function_set, str):
                        function_ids = [function_set]
                        n_func = 1
                    else:
                        function_ids = list(np.unique(function_set))
                        n_func = len(function_ids)
                    function_df = self.query(f'id in {function_ids}')
                    # And if you got a set of results that's not the
                    #   same length as the unique indices, blow up.
                    err_msg = f"""FunctionBank KeyError:

                    The queried keys: {function_ids}
                    The number of functions: {n_func}
                    The available keys: {self.to_df().id.to_list()}
                    function_df:\n{function_df}
                    """
                    if function_df.shape[0] != n_func:
                        raise Exception(err_msg)
                except KeyError:
        # # 5) Otherwise it needs to blow up, scratching its head.
                    err_msg = """Function Scoring Error:

                        The function_set argument (the functions to score) can
                        be a Pandas DataFrame (i.e. Experiment Space) with an id
                        column, an iterable of Function representations with an id
                        field, or simply a string or iterable of strings
                        representing the ids to query for from the function bank.

                        You passed:\n{function_set}
                        """
                    raise Exception
        functionbank = self.to_df()
        # This gets a list of all the scoring columns from the
        #   function bank.
        score_cols = [
            _ for _ in functionbank.columns
            if _.startswith('score_')
        ]
        # This gets the maximum positional index for the column
        #   a.k.a. column number `n`
        max_col_ind = int(np.max(
            [functionbank.columns.get_loc(c) for c in score_cols]
        ))
        # This, then, walks through the scoring columns passed.
        # If the scoring column *already* exists then it is updated
        #   in place.
        # This is used to make the deques.
        n = functionbank.shape[0]
        for score_col, score in zip(_score_name, _score):
            # Does the scoring column exist?
            if score_col not in score_cols:
                # If not, create it with empty deques.
                functionbank.insert(
                    loc = max_col_ind + 1,
                    column = score_col,
                    value = [deque([0], maxlen=100) for _ in range(n)]
                )
                max_col_ind += 1
            # This is an n x m DataFrame where n is the number of
            # functions being scored and m is the number of scoring
            # functions that already exist. This then pulls the
            # scoring column out, as a series, and calls apply on it
            # to update the deques.
            functionbank.query(
                f'id == {function_ids}'
            )[score_col].apply(
                lambda x: x.append(score)
            )
        functionbank.reset_index(inplace=True, drop = True)
        # Now, update the manifest.
        self._function_manifest = functionbank.to_dict(
            orient='records'
        )

    ################################################################
    #                       Helper Functions                       #
    #                                                              #
    # These helper functions provide general helpful utility:      #
    #   idxmax: Return the highest function index                  #
    #   to_df: Return the function manifest cast to dataframe.     #
    ################################################################
    def idxmax(self) -> int:
        """Return the max index.

        This returns the maximum index of the functions in the bank.

        Returns
        -------
        idxmax: int
            The maximum integer index for composed functions in the
            function bank.

        Examples
        --------
        >>> import pandas as pd
        >>> from mentalgym.utils.data import testing_df
        >>> from mentalgym.functionbank import FunctionBank
        >>> from mentalgym.utils.function import make_function
        >>> function_bank = FunctionBank(testing_df)
        >>> composed_functions = [
        ...     make_function(
        ...         function_index = -3,
        ...         function_id = 'steve',
        ...         function_inputs = ['1'],
        ...         function_type = 'intermediate',
        ...         function_location = [1, 1]
        ...     ),
        ...     make_function(
        ...         function_index = -4,
        ...         function_id = 'bob',
        ...         function_inputs = ['1', '2'],
        ...         function_type = 'intermediate',
        ...         function_location = [1, 1, 2]
        ...     )
        ... ]
        >>> function_bank.idxmax()
        2
        >>> function_bank.append(composed_functions)
        >>> function_bank.idxmax()
        4
        """
        return self.to_df().i.max()

    def to_df(self):
        """Convenience function to return function df.

        This wraps the internal manifest in a dataframe and returns
        it. This is just a convenience function.

        Returns
        -------
        function_bank_df: pd.DataFrame
            A DataFrame representation of the function bank.
        """
        return pd.DataFrame(self._function_manifest)
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

        If one does not exist a default manifest will be created.
        The passed dataset ensures that input / output are created
        and validated correctly.

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
        # This function will build a default manifest of functions
        #   and save it locally, if it does not exist.
        # Anything that's not JSON compatible gets pickled in and out.
        # The force refresh allows for stomping on data, even if it
        #   already exists.
        if (not os.path.exists(manifest_file)) or self._force_refresh:
            self._function_manifest = build_default_function_space(
                self._data,
                self._target
            )
            self._save_bank()
        # Read in the manifest
        function_manifest = pd.read_json(
            manifest_file
        )
        # TODO: If the column order ever gets whack; use this
        # fm_columns = function_manifest.columns
        # Here we are going to load any pickled things.
        # Identify the scoring fields.
        score_col = [
            _ for _ in function_manifest.columns
            if _.startswith('score')
        ]
        # This then pulls out all the pickleable fields.
        pickle_fields = [
            'object', *[_ for _ in score_col]
        ]
        # This turns the pickleable fields into bytes and saves
        #   them out to disk. This is because they cannot be
        #   represented in JSON.
        def load_pickles(x: pd.Series):
            """Load pickleable items"""
            # Turn the Series into a dictionary
            pickle_items = x.to_dict()
            # Use the popped ID to check for the dir
            function_folder = os.path.join(
                self._function_bank_directory,
                pickle_items.pop('id')
            )
            # Then walk through the dictionary and save
            #   all the items that can't be dumped to JSON.
            update_dict = {}
            for k in pickle_items.keys():
                flname = os.path.join(
                    function_folder,
                    f'{k}.pickle'
                )
                update_dict[k] = pickle.load(
                    open(flname,'rb')
                )
            return update_dict

        # Use the loading function to get all the pickled data
        #   back into a list of dicts.
        f_list = function_manifest[
            ['id'] + pickle_fields
        ].apply(load_pickles, 1).to_list()
        # Dataframify it
        f_frame = pd.DataFrame(f_list)
        # Replace the data in the manifest with the unpickled data.
        function_manifest.loc[:, pickle_fields] = f_frame
        # Turn it into a Function Set.
        function_set = function_manifest.to_dict(
            orient = 'records'
        )
        # Then, do any validation necessary for those functions
        validate_function_set(function_set)
        # Finally, give up the goods.
        return function_set

    def _save_bank(self) -> None:
        """Save function bank to local directory.

        This dumps the manifest out to disk; any fields that cannot
        be represented as JSON will be pickled out. The pickled
        fields will be changed 'in-situ' to the string 'pickle' to
        represent that they can be loaded from disk.
        """
        # This defines the JSON file which holds the manifest.
        manifest_file = os.path.join(
            self._function_bank_directory,
            '.manifest'
        )
        # This turns the list of functions into something that's
        #   a little easier to slice and dice.
        _writable = self.to_df()
        # This pulls out the score columns (which are deques)
        score_col = [
            _ for _ in _writable.columns
            if _.startswith('score')
        ]
        # This then pulls out all the pickleable fields.
        pickle_fields = ['object', *[_ for _ in score_col]]
        # This turns the pickleable fields into bytes and saves
        #   them out to disk. This is because they cannot be
        #   represented in JSON.
        def save_pickles(x: pd.Series):
            """Save pickleable items"""
            # Turn the Series into a dictionary
            pickle_items = x.to_dict()
            # Use the popped ID to check for the dir
            function_folder = os.path.join(
                self._function_bank_directory,
                pickle_items.pop('id')
            )
            # Make it if it doesn't exist
            os.makedirs(function_folder, exist_ok = True)
            # Then walk through the dictionary and save
            #   all the items that can't be dumped to JSON.
            for k, v in pickle_items.items():
                flname = os.path.join(
                    function_folder,
                    f'{k}.pickle'
                )
                pickle.dump(v, open(flname, "wb"))
                # with open(flname, 'wb') as f:
                #     pickle.dump(v, f)
        # Save all the pickleable items out
        _writable.loc[
            :, ['id'] + pickle_fields
        ].apply(save_pickles, 1)
        # Change all the pickleable fields to 'pickle' so they
        #   don't lose their position in the dataframe.
        _writable.loc[
            :, pickle_fields
        ] = 'pickle'
        _writable.to_json(
            manifest_file,
            orient = 'records'
        )