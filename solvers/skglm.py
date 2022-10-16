from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from skglm.estimators import MultiTaskLasso


class Solver(BaseSolver):
    name = "skglm"
    stopping_strategy = "iteration"

    install_cmd = "conda"
    requirements = ["skglm"]
    references = [
        'Q. Bertrand and Q. Klopfenstein and P.-A. Bannier and G. Gidel '
        'and M. Massias '
        '"Beyond L1: Faster and Better Sparse Models with skglm", NeurIPS 2022 '
        'https://arxiv.org/abs/2204.07826'
    ]

    def set_objective(self, X, Y, lmbd):
        self.X, self.Y = X, Y
        self.lmbd = lmbd
        self.tol = 1e-8
        self.clf = MultiTaskLasso(
            alpha=self.lmbd / self.X.shape[0], tol=self.tol / (Y ** 2).sum(),
            fit_intercept=False)

        # Cache Numba compilation
        self.run(1)

    def run(self, n_iter):
        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.Y)

    def get_result(self):
        return self.clf.coef_.T
