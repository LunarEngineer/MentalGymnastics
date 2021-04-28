"""Contains defined sampling functions compliant with the gym.

This file contains a few sampling functions which demonstrate the
input and output structure for the sampling functionality in the
gym.

The data structure of the function bank includes fields:
* id
* type
* score_${score_function_1_name}
* score_${score_function_..._name}
* score_${score_function_n_name}

These sampling functions return an iterable of ids compatible with
those in the experiment space and function bank.
"""

import numpy as np
import pandas as pd
from numpy.random import default_rng
from numpy.typing import ArrayLike
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Optional


def softmax_score_sample(
    x: pd.DataFrame,
    n: int,
    random_state: Optional[int] = None
) -> ArrayLike:
    """Returns ids for sampled functions.

    This function sends all scoring columns into a standard scaler,
    takes the geometric mean across scoring columns afterwards, then
    calculates softmax for the resulting vector to ensure all the
    values are positive and less than or equal to one.

    NaN values are filled with the mean prior to running through the
    standard scaling.

    The softmax values are used as probabilistic weights and a
    'choice' sampling of `n` items without replacement is conducted.

    Parameters
    ----------
    x: pd.DataFrame
        This is the Function Bank in DataFrame format.
    n: int
        The number of ids to sample without replacement.
        If this number is greater than the number of ids then
        all ids are returned.
    random_state: Optional[int] = None
        An optional randomness value. This is used to init Numpy's
        default_rng, so there's a reasonable range of values.
        https://numpy.org/doc/stable/reference/random/generator.html

    Examples
    --------
    >>> import pandas as pd
    >>> from mentalgym.utils.sampling import softmax_score_sample
    >>> scoring_output = pd.DataFrame(
    ...     data = {
    ...         'id': ['bob','janice','dilly','dally','beans'],
    ...         'meaningless': ['a','b','c','d','e'],
    ...         'extra': ['a','b','c','d','e'],
    ...         'information': ['a','b','c','d','e'],
    ...         'score_accuracy': [[0.95], [0.7], [0.6], [0.5], [0.6]],
    ...         'score_complexity': [[0.01, 0.05], [100], np.nan, [20], [50]]
    ...     }
    ... )
    >>> softmax_score_sample(scoring_output, 4, 0)
    array(['janice', 'bob', 'beans', 'dally'], dtype=object)
    """
    # Do a quick check on the number of unique ids.
    # If it's less than the desired sample just hand it all back.
    if x.id.count() < n:
        return x.id
    # Make the default rng repeatable.
    rng = default_rng(random_state)
    # Pull out the scoring dataset column names.
    score_col = [_ for _ in x.columns if _.startswith('score_')]
    # Downselect to the keep columns.
    scoring_data = x[score_col].applymap(np.mean)
    # Fill the NaN scaled scores with the mean
    imputer = SimpleImputer()
    imputed_scores = imputer.fit_transform(scoring_data)
    # Standardize each column.
    col_scaler = StandardScaler()
    scaled_scores = col_scaler.fit_transform(imputed_scores)
    # Now take the mean across rows.
    scaled_and_averaged_scores = np.mean(
        scaled_scores,
        axis=1
    )
    # Compute softmax for the treated scores.
    s = softmax(scaled_and_averaged_scores)
    # Then draw some samples using the softmax output as a quasi
    #   probabilistic vector. The output of this is an n-length
    #   vector of ids.
    sampled_ids = rng.choice(
        x.id,
        size = n,
        replace = False,
        p = s
    )
    return list(sampled_ids)