"""
Variational inference for the Dirichlet Process Gaussian Mixture Model.

The implementation now lives in dpgm.py (consolidated). This module is a thin
compatibility shim that re-exports it, with ``VariationalDPGM`` kept as an alias
for the class's earlier name.
"""

from dpgm import (
    VariationalInference,
    fit_dpgm_vi,
    predict_dpgm_vi,
    _log_wishart_normalizer,
)

# backwards-compatible alias for the original class name
VariationalDPGM = VariationalInference

__all__ = [
    "VariationalInference",
    "VariationalDPGM",
    "fit_dpgm_vi",
    "predict_dpgm_vi",
]


if __name__ == '__main__':
    # self-check on synthetic data: ELBO should increase monotonically and recover 3 clusters
    import numpy

    numpy.random.seed(42)
    n = 200
    c1 = numpy.random.randn(n, 2) * 0.5
    c2 = numpy.random.randn(n, 2) * 0.5 + numpy.array([6, 6])
    c3 = numpy.random.randn(n, 2) * 0.5 + numpy.array([6, 0])
    X = numpy.vstack([c1, c2, c3])
    X = X[numpy.random.permutation(len(X))]

    print("Fitting variational DP-GMM on {} synthetic points...".format(len(X)))
    result = fit_dpgm_vi(X, alpha=1.0, truncation=20, max_iter=200, verbose=True)
    print("\nEffective clusters found: {}".format(result['n_clusters']))
    print("Cluster counts: {}".format(result['cluster_counts']))
    traj = result['model'].elbo_trajectory_
    increases = all(traj[i + 1] >= traj[i] - 1e-6 for i in range(len(traj) - 1))
    print("ELBO monotonically non-decreasing: {} ({} iters, final {:.2f})".format(
        increases, len(traj), traj[-1]))
