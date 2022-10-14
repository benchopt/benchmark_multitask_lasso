import numpy as np

from benchopt import BaseDataset


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_samples, n_features, n_tasks': [
            (10, 30, 5),
            (100, 500, 100)]
    }

    def __init__(self, n_samples=10, n_features=50, n_tasks=30, random_state=27):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_tasks = n_tasks
        self.random_state = random_state

    def get_data(self):

        rng = np.random.RandomState(self.random_state)
        X = rng.randn(self.n_samples, self.n_features)
        Y = rng.randn(self.n_samples, self.n_tasks)

        # `data` holds the keyword arguments for the `set_data` method of the
        # objective.
        # They are customizable.
        return dict(X=X, Y=Y)
