"""Contains data for testing."""
from sklearn.datasets import make_classification

####################################################################
#                   Create simple testing data                     #
####################################################################
# This dataset will be used as a testing dataset for the Gym.
X, y = make_classification(
    n_samples = 100000,
    n_features = 4,
    n_informative = 2,
    n_redundant = 2,
    n_repeated = 0,
    n_classes = 2,
    n_clusters_per_class = 2,
    flip_y=0.01,
    class_sep=1.0,
    shift=0.0,
    scale=1.0,
    shuffle=True,
    random_state=42
)

testing_df = pd.DataFrame(
    X,
    columns = [
        "A",
        "B",
        "C",
        "D"
    ]
).assign(Y=y)

