from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm


class Objective(BaseObjective):
    name = "Multitask Lasso"
    parameters = {
        "reg": [1, 0.5, 0.1, 0.01],
    }

    def __init__(self, reg=0.1):
        self.reg = reg

    def set_data(self, X, Y):
        self.X, self.Y = X, Y
        # todo normalize by n_samples?
        self.alpha_max = np.max(norm(X.T @ Y, axis=1))
        self.lmbd = self.reg * self.alpha_max

    def compute(self, W):
        R = self.Y - self.X @ W
        p_obj = 0.5 * norm(R, ord="fro") ** 2 + self.lmbd * norm(
            W, axis=1).sum()
        nnz = np.sum(norm(W, axis=1) != 0)
        return dict(value=p_obj, sparsity=nnz)

    def to_dict(self):
        return dict(X=self.X, Y=self.Y, lmbd=self.lmbd)
