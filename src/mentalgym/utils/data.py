"""These are functions for working with data."""
import pandas as pd
from typing import Any, Dict, Iterable, Optional

def dataset_to_actions(
    dataset: pd.DataFrame,
    target: Optional[str] = None
) -> Iterable[Dict[str, Any]]:
    """Convert Pandas data to input nodes.

    Parameters
    ----------
    df: pandas.DataFrame
        A modeling dataset.
    target: Optional[str] = None
        If left blank this will assume the final column is the target.

    Returns
    -------
    input_actions: Iterable[Dict[str,Any]]
        An iterable of dictionaries which represent input nodes.

    Examples
    --------
    >>> import pandas as pd
    >>> static_df = pd.DataFrame(
    >>>     data = {
    >>>         'A': [1, 2, 3],
    >>>         'B': [4, 5, 6],
    >>>         'C': [7, 8, 9]
    >>>     }
    >>> )
    >>> dataset_to_actions(static_df, target = 'A')
    [{'id': 'A',
      'type': 'sink',
      'input': None,
      'values': 0    1
      1    2
      2    3
      Name: A, dtype: int64},
     {'id': 'B',
      'type': 'source',
      'input': None,
      'values': 0    4
      1    5
      2    6
      Name: B, dtype: int64},
     {'id': 'C',
      'type': 'source',
      'input': None,
      'values': 0    7
      1    8
      2    9
      Name: C, dtype: int64}]
    """
    # Set a default target if none available
    if target is None:
        target = static_df.columns[-1]
    # Create an empty list
    output = []
    # Now, walk through the columns
    for col, vals in static_df.iteritems():
        # Create an action
        col_dict = {
            'id': col,
            'type': 'sink' if col == target else 'source',
            'input': None,
            'values': vals
        }
        # Add it to the output list
        output.append(col_dict)
    return output
