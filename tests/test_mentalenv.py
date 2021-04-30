import pytest
import gym
import pandas as pd
import numpy as np
from mentalgym.envs import MentalEnv
from sklearn.datasets import make_classification


def test__build_atomic_function():
    action_index = 0
    action_location = (0, 0)
    connected_df = pd.DataFrame(
        columns=[
            "i",
            "id",
            "type",
            "input",
            "object",
            "hyperparameters",
            "exp_loc_0",
            "exp_loc_1",
        ]
    )
    MentalEnv._build_atomic_function(
        self, action_index, action_location, connected_df
    )


if __name__ == "__main__":
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
        random_state=42,
    )

    dataset = (
        pd.DataFrame(X)
        .assign(y=y)
        .rename(columns={_: str(_) for _ in range(X.shape[1])}),
    )

    test_env = MentalEnv(
        dataset=dataset,
        experiment_space_min=np.array([0.0, 0.0]),
        experiment_space_max=np.array([100.0, 100.0]),
        number_functions=8,
        max_steps=4,
        epochs=5,
        net_lr=0.0001,
        net_batch_size=128,
        seed=None,
        verbose=False,
        **kwargs,
    )
