"""Contains data and data utilities for the environment."""
import tempfile
import pandas as pd
from mentalgym.functionbank import FunctionBank
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import gin

####################################################################
#                   Create simple testing data                     #
####################################################################
# This dataset will be used as a testing dataset for the Gym.
X, y = make_classification(
    n_samples=10,
    n_features=3,
    n_informative=2,
    n_redundant=1,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=2,
    flip_y=0.01,
    class_sep=1.0,
    shift=0.0,
    scale=1.0,
    shuffle=True,
    random_state=42
)
testing_df = pd.DataFrame(
    X,
    columns=[
        "column_0",
        "column_1",
        "column_2"
    ]
).assign(output=y)

@gin.configurable
def make_dataset(name: str):

    dataset = None

    if name == 'iris':
        from sklearn.datasets import load_iris
        X, y = load_iris(return_X_y=True, as_frame=True)
        dataset = X.assign(output=y)

    return dataset

def make_sk2c():
    X, y = make_classification(
        n_samples = 100000,
        n_features = 100,
        n_informative = 30
    )
    dataset = pd.DataFrame(
        X,
        columns = [f'INPUT_{_}' for _ in range(X.shape[1])]
    ).assign(output=y)
    dataset, testset = train_test_split(
        dataset,
        test_size=.2,
        random_state=0
    )
    dataset, valset = train_test_split(
        dataset,
        test_size=.3,
        random_state=1
    )
    
    return dataset, valset, testset
####################################################################
#                Create simple Experiment Space                    #
####################################################################
# This base container has the input and output elements of the
# testing dataset within it.

# base_container = refresh_experiment_container(
#     pd.DataFrame(dataset_to_functions(testing_df))
# )
class Empty():
    pass

####################################################################
#                Simple Composed Function Example                  #
####################################################################
# This base container has the input and output elements of the
#   testing dataset within it. These functions are added to extend
#   the space. These represent actions which would be made by agents
#   during the course of an episode.

function_composed_one = {
    'i': 2,
    'id': 'steve',
    'type': 'composed',
    'input': ['column_0'],
    'object': Empty,
    'exp_loc_0': 25.,
    'exp_loc_1': 50.,
}
function_composed_two = {
    'i': 3,
    'id': 'bob',
    'type': 'composed',
    'object': Empty,
    'input': ['column_0', 'column_1'],
    'exp_loc_0': 50.,
    'exp_loc_1': 75.,
}
function_composed_three = {
    'i': 4,
    'id': 'carl',
    'type': 'composed',
    'object': Empty,
    'input': ['steve', 'bob', 'column_1'],
    'exp_loc_0': 50.,
    'exp_loc_1': 75.,
}


function_set = [
    function_composed_one,
    function_composed_two,
    function_composed_three
]


####################################################################
#                  Create simple Function Bank                     #
####################################################################
# This default function bank has information for the input, output,
#   atomic, and composed functions. It reads the atomic from the
#   mentalgym.atomic module, reads the composed from the directory
#   within which it's instantiated, and reads the input and output
#   from the refreshed experiment container. Using a temp directory
#   helps with cleanup afterwards.
with tempfile.TemporaryDirectory() as d:
    function_bank = FunctionBank(
        modeling_data = testing_df,
        function_bank_directory = d
    )
