"""Contains data for testing."""
from sklearn.datasets import make_classification

####################################################################
#                   Create simple testing data                     #
####################################################################

####################################################################
#                Create simple Experiment Space                    #
####################################################################
simple_experiment_space = {
    "id": ["A", "B", "C", "SINK"],
    "location": np.array([[0, 0], [0, 1], [0, 2], [1, 1]])
}