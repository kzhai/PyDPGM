"""
Variational inference (mean-field CAVI) for a Dirichlet Process Gaussian Mixture Model.

Companion to dpgm.py (collapsed Gibbs sampler). Where the Gibbs sampler draws samples and
mixes slowly, this performs deterministic coordinate-ascent variational inference (CAVI) and
converges to a local optimum of the ELBO in a handful of iterations.

Model
-----
    - Truncated stick-breaking DP prior (Blei & Jordan, 2006):
        v_t ~ Beta(1, alpha),   pi_t = v_t * prod_{s<t}(1 - v_s),   t = 1..T
      truncated at T components (T is an upper bound; unused components get ~0 weight).
    - Conjugate Normal-Inverse-Wishart prior on each component (Bishop, PRML 10.2):
        Lambda_t ~ Wishart(W_0, nu_0),   mu_t | Lambda_t ~ N(m_0, (beta_0 Lambda_t)^{-1})
    - z_n ~ Cat(pi),   x_n | z_n = t ~ N(mu_t, Lambda_t^{-1})

Variational factors (mean field):
    q(v_t)          = Beta(gamma1_t, gamma2_t)
    q(mu_t,Lam_t)   = NormalInverseWishart(m_t, beta_t, W_t, nu_t)
    q(z_n)          = Categorical(resp_n)

Only depends on numpy / scipy.

Reference: Blei & Jordan (2006), "Variational inference for Dirichlet process mixtures";
Bishop (2006), PRML chapter 10.2.
"""

import numpy
import scipy.special
from scipy.special import digamma, gammaln, logsumexp

numpy.seterr(divide='ignore')


def _log_wishart_normalizer(logdet_W, nu, D):
    """log B(W, nu) from Bishop (B.79), given log|W| and degrees of freedom nu."""
    i = numpy.arange(1, D + 1)
    return -(nu / 2.0) * logdet_W - (nu * D / 2.0) * numpy.log(2.0) \
        - (D * (D - 1) / 4.0) * numpy.log(numpy.pi) \
        - numpy.sum(gammaln((nu + 1 - i) / 2.0))


class VariationalDPGM(object):
    """
    Mean-field variational inference for a DP Gaussian mixture.

    Parameters
    ----------
    truncation : int
        Truncation level T (maximum number of components). Set well above the number of
        clusters you expect; unused components collapse to ~zero weight.
    alpha : float
        DP concentration. Higher -> more effective components.
    max_iter : int
        Maximum CAVI iterations.
    tol : float
        Convergence tolerance on the relative ELBO change.
    beta_0, nu_0, m_0, W_0 : NIW prior hyperparameters (sensible data-driven defaults if None).
    random_state : int
        Seed for the responsibility initialization.
    verbose : bool
    """

    def __init__(self, truncation=50, alpha=1.0, max_iter=200, tol=1e-4,
                 beta_0=1.0, nu_0=None, m_0=None, W_0=None,
                 random_state=0, verbose=True):
        self.truncation = truncation
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self._beta_0 = beta_0
        self._nu_0 = nu_0
        self._m_0 = m_0
        self._W_0 = W_0
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ setup

    def _initialize(self, X):
        self._X = X
        (self._N, self._D) = X.shape
        T, D = self.truncation, self._D

        # ---- prior hyperparameters (data-driven defaults) ----
        self._m0 = numpy.mean(X, axis=0) if self._m_0 is None else numpy.asarray(self._m_0, float)
        self._beta0 = float(self._beta_0)
        self._nu0 = float(D + 2) if self._nu_0 is None else float(self._nu_0)
        if self._W_0 is None:
            # E[Lambda] = nu_0 * W_0 ~ inv(data covariance): match the prior precision to data scale
            data_cov = numpy.cov(X.T) + 1e-6 * numpy.eye(D)
            self._W0 = numpy.linalg.inv(data_cov) / self._nu0
        else:
            self._W0 = numpy.asarray(self._W_0, float)
        self._W0_inv = numpy.linalg.inv(self._W0)
        assert self._m0.shape == (D,) and self._W0.shape == (D, D)

        # ---- initialize responsibilities via a distance-based soft assignment ----
        rng = numpy.random.RandomState(self.random_state)
        seed_means = X[rng.choice(self._N, T, replace=(T > self._N))]
        # squared distances to the T seed means -> softmax responsibilities
        sq_dist = numpy.sum((X[:, None, :] - seed_means[None, :, :]) ** 2, axis=2)
        log_resp = -0.5 * sq_dist
        log_resp -= logsumexp(log_resp, axis=1, keepdims=True)
        self._resp = numpy.exp(log_resp)

    # -------------------------------------------------------------- CAVI steps

    def _compute_statistics(self, resp):
        """Weighted count, mean, and scatter per component (Bishop 10.51-10.53)."""
        T, D = self.truncation, self._D
        Nk = resp.sum(axis=0) + 1e-10                       # (T,)
        xbar = (resp.T @ self._X) / Nk[:, None]             # (T, D)
        S = numpy.zeros((T, D, D))
        for t in range(T):
            diff = self._X - xbar[t]                        # (N, D)
            S[t] = (resp[:, t][:, None] * diff).T @ diff / Nk[t]
        return Nk, xbar, S

    def _update_niw(self, Nk, xbar, S):
        """Posterior NIW parameters per component (Bishop 10.60-10.63)."""
        self._beta = self._beta0 + Nk
        self._nu = self._nu0 + Nk
        self._m = (self._beta0 * self._m0[None, :] + Nk[:, None] * xbar) / self._beta[:, None]
        self._W = numpy.zeros((self.truncation, self._D, self._D))
        for t in range(self.truncation):
            diff = (xbar[t] - self._m0)[:, None]
            W_inv = (self._W0_inv + Nk[t] * S[t]
                     + (self._beta0 * Nk[t] / self._beta[t]) * (diff @ diff.T))
            self._W[t] = numpy.linalg.inv(W_inv)

    def _update_sticks(self, Nk):
        """Beta posteriors for the stick-breaking weights (Blei & Jordan, 2006)."""
        self._gamma1 = 1.0 + Nk
        # sum of counts in components *after* t
        tail = numpy.cumsum(Nk[::-1])[::-1] - Nk
        self._gamma2 = self.alpha + tail

    def _expectations(self):
        """E[ln|Lambda_t|], E[ln pi_t], and Beta digamma terms used by the E-step and ELBO."""
        T, D = self.truncation, self._D
        i = numpy.arange(1, D + 1)

        logdet_W = numpy.linalg.slogdet(self._W)[1]                              # (T,)
        E_logdet_Lambda = (digamma((self._nu[:, None] + 1 - i[None, :]) / 2.0).sum(axis=1)
                           + D * numpy.log(2.0) + logdet_W)                      # (T,)

        digamma_sum = digamma(self._gamma1 + self._gamma2)
        E_log_v = digamma(self._gamma1) - digamma_sum                            # (T,)
        E_log_1mv = digamma(self._gamma2) - digamma_sum                          # (T,)
        # E[ln pi_t] = E[ln v_t] + sum_{s<t} E[ln(1 - v_s)]
        cumulative = numpy.concatenate(([0.0], numpy.cumsum(E_log_1mv)[:-1]))
        E_log_pi = E_log_v + cumulative                                          # (T,)

        return E_logdet_Lambda, logdet_W, E_log_pi, E_log_v, E_log_1mv

    def _e_step(self, E_logdet_Lambda, E_log_pi):
        """Responsibilities from expected log-likelihood + expected log-weights (Bishop 10.46, 10.67)."""
        T, D = self.truncation, self._D
        # expected quadratic form E[(x-mu)^T Lambda (x-mu)] = D/beta_t + nu_t (x-m_t)^T W_t (x-m_t)
        log_rho = numpy.zeros((self._N, T))
        for t in range(T):
            diff = self._X - self._m[t]                     # (N, D)
            maha = numpy.sum((diff @ self._W[t]) * diff, axis=1)
            E_quad = D / self._beta[t] + self._nu[t] * maha
            log_rho[:, t] = (E_log_pi[t] + 0.5 * E_logdet_Lambda[t]
                             - 0.5 * D * numpy.log(2.0 * numpy.pi) - 0.5 * E_quad)
        log_resp = log_rho - logsumexp(log_rho, axis=1, keepdims=True)
        return log_resp, numpy.exp(log_resp)

    # ------------------------------------------------------------------- ELBO

    def _elbo(self, Nk, xbar, S, resp, log_resp,
              E_logdet_Lambda, logdet_W, E_log_pi, E_log_v, E_log_1mv):
        T, D = self.truncation, self._D
        ln2pi = numpy.log(2.0 * numpy.pi)

        # E[ln p(X | Z, mu, Lambda)]  (Bishop 10.71)
        term_x = 0.0
        for t in range(T):
            diff = (xbar[t] - self._m[t])[:, None]
            quad = (self._nu[t] * (diff.T @ self._W[t] @ diff)).item()
            trace_SW = self._nu[t] * numpy.trace(S[t] @ self._W[t])
            term_x += 0.5 * Nk[t] * (E_logdet_Lambda[t] - D / self._beta[t]
                                     - trace_SW - quad - D * ln2pi)

        # E[ln p(Z | pi)]  (Bishop 10.72)
        term_z = numpy.sum(Nk * E_log_pi)

        # E[ln p(v)] with Beta(1, alpha) stick prior
        term_v = numpy.sum(numpy.log(self.alpha) + (self.alpha - 1.0) * E_log_1mv)

        # E[ln p(mu, Lambda)]  (Bishop 10.74)
        term_ml = 0.0
        logB0 = _log_wishart_normalizer(numpy.linalg.slogdet(self._W0)[1], self._nu0, D)
        for t in range(T):
            diff = (self._m[t] - self._m0)[:, None]
            quad = (self._beta0 * self._nu[t] * (diff.T @ self._W[t] @ diff)).item()
            term_ml += 0.5 * (D * numpy.log(self._beta0 / (2.0 * numpy.pi))
                              + E_logdet_Lambda[t] - D * self._beta0 / self._beta[t] - quad)
            term_ml += 0.5 * (self._nu0 - D - 1.0) * E_logdet_Lambda[t]
            term_ml -= 0.5 * self._nu[t] * numpy.trace(self._W0_inv @ self._W[t])
        term_ml += T * logB0

        # -E[ln q(Z)]  (entropy of responsibilities)
        term_qz = -numpy.sum(resp * log_resp)

        # -E[ln q(v)]  (entropy of the Beta factors)
        ln_beta_fn = gammaln(self._gamma1) + gammaln(self._gamma2) - gammaln(self._gamma1 + self._gamma2)
        term_qv = -numpy.sum((self._gamma1 - 1.0) * E_log_v
                             + (self._gamma2 - 1.0) * E_log_1mv - ln_beta_fn)

        # -E[ln q(mu, Lambda)]  (Bishop 10.77 with Wishart entropy B.82)
        term_qml = 0.0
        for t in range(T):
            logB_t = _log_wishart_normalizer(logdet_W[t], self._nu[t], D)
            H_lambda = -logB_t - 0.5 * (self._nu[t] - D - 1.0) * E_logdet_Lambda[t] + 0.5 * self._nu[t] * D
            e_ln_q = (0.5 * E_logdet_Lambda[t] + 0.5 * D * numpy.log(self._beta[t] / (2.0 * numpy.pi))
                      - 0.5 * D - H_lambda)
            term_qml -= e_ln_q

        return term_x + term_z + term_v + term_ml + term_qz + term_qv + term_qml

    # -------------------------------------------------------------------- fit

    def fit(self, X):
        self._initialize(X)
        previous_elbo = None
        self.elbo_trajectory_ = []

        for iteration in range(self.max_iter):
            # M-step: sufficient statistics -> NIW + stick posteriors
            Nk, xbar, S = self._compute_statistics(self._resp)
            self._update_niw(Nk, xbar, S)
            self._update_sticks(Nk)

            # expectations, ELBO, then E-step
            E_logdet_Lambda, logdet_W, E_log_pi, E_log_v, E_log_1mv = self._expectations()
            elbo = self._elbo(Nk, xbar, S, self._resp, numpy.log(self._resp + 1e-300),
                              E_logdet_Lambda, logdet_W, E_log_pi, E_log_v, E_log_1mv)
            self.elbo_trajectory_.append(elbo)

            log_resp, self._resp = self._e_step(E_logdet_Lambda, E_log_pi)

            if self.verbose and (iteration + 1) % max(1, self.max_iter // 20) == 0:
                print("Iteration {}/{}: ELBO = {:.2f}, effective clusters = {}".format(
                    iteration + 1, self.max_iter, elbo, self.n_effective_clusters(Nk)))

            if previous_elbo is not None:
                rel_change = abs(elbo - previous_elbo) / (abs(previous_elbo) + 1e-10)
                if rel_change < self.tol:
                    if self.verbose:
                        print("Converged at iteration {} (relative ELBO change {:.2e})".format(
                            iteration + 1, rel_change))
                    break
            previous_elbo = elbo

        self._Nk = self._resp.sum(axis=0)
        self.converged_elbo_ = self.elbo_trajectory_[-1]
        return self

    def n_effective_clusters(self, Nk=None, threshold=1.0):
        if Nk is None:
            Nk = self._resp.sum(axis=0)
        return int(numpy.sum(Nk > threshold))

    # ---------------------------------------------------------------- predict

    def predict(self, X):
        """Hard cluster assignment (argmax responsibility) for new data."""
        X = numpy.asarray(X, float)
        saved_X, saved_N = self._X, self._N
        self._X, self._N = X, X.shape[0]
        E_logdet_Lambda, _, E_log_pi, _, _ = self._expectations()
        _, resp = self._e_step(E_logdet_Lambda, E_log_pi)
        self._X, self._N = saved_X, saved_N
        return numpy.argmax(resp, axis=1)


def fit_dpgm_vi(data, alpha=1.0, truncation=50, max_iter=200, tol=1e-4,
                random_state=0, verbose=True):
    """
    Fit a DP Gaussian mixture by variational inference. Returns a dict mirroring
    dpgm.fit_dpgm so it drops into the same tooling.

    Cluster labels are remapped to a contiguous 0..K-1 over the effective (non-empty) components.
    """
    data = numpy.asarray(data, float)
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array (N x M)")

    model = VariationalDPGM(truncation=truncation, alpha=alpha, max_iter=max_iter,
                            tol=tol, random_state=random_state, verbose=verbose)
    model.fit(data)

    raw_labels = numpy.argmax(model._resp, axis=1)
    used = numpy.unique(raw_labels)
    remap = {old: new for new, old in enumerate(used)}
    labels = numpy.array([remap[l] for l in raw_labels], dtype=int)

    return {
        'labels': labels,
        'n_clusters': len(used),
        'cluster_means': model._m[used].copy(),
        'cluster_counts': numpy.bincount(labels, minlength=len(used)),
        'log_likelihood': model.converged_elbo_,   # ELBO of the converged fit
        'model': model,
    }


def predict_dpgm_vi(model, data):
    """Predict contiguous-relabeled cluster assignments for new data."""
    if isinstance(model, dict):
        model = model['model']
    return model.predict(numpy.asarray(data, float))


if __name__ == '__main__':
    # self-check on synthetic data: ELBO should increase monotonically and recover 3 clusters
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
