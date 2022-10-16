from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import warnings
    import numpy as np
    from celer import MultiTaskLasso
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = "Celer"
    stopping_strategy = 'iteration'

    install_cmd = 'conda'
    requirements = ['pip:celer']
    references = [
        'M. Massias, A. Gramfort and J. Salmon, ICML, '
        '"Celer: a Fast Solver for the Lasso with Dual Extrapolation", '
        'vol. 80, pp. 3321-3330 (2018)'
    ]

    def set_objective(self, X, Y, lmbd):
        self.X, self.Y, self.lmbd = X, Y, lmbd
        self.clf = MultiTaskLasso(
            alpha=lmbd / len(Y), tol=1e-8 / (Y ** 2).sum(), fit_intercept=False,
            verbose=0, prune=True)

    def run(self, n_iter):
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        if n_iter == 0:
            self.clf.coef_ = np.zeros((self.Y.shape[1], self.X.shape[1]))
        else:
            self.clf.max_iter = n_iter
            self.clf.fit(self.X, self.Y)

    @staticmethod
    def get_next(n_iter):
        return n_iter + 1

    def get_result(self):
        return self.clf.coef_.T
