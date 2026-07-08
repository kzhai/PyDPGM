"""
Consolidated Dirichlet Process Gaussian Mixture Model (DPGM) script
Author: Consolidated from PyDPGM repository

This script provides a standalone, independent implementation of DPGM clustering
with no external dependencies (except standard scientific libraries).

Original author: Ke Zhai (zhaike@cs.umd.edu)
Implements collapsed Gibbs sampling for the Dirichlet process Gaussian mixture model (DPGM).
"""

import os
import random
import sys

import numpy
import scipy
import scipy.special

negative_infinity = -1e500
# We will be taking log(0) = -Inf, so turn off this warning
numpy.seterr(divide="ignore")


class MonteCarlo(object):
    """
    @param truncation_level: the maximum number of clusters, used for speeding up the computation
    @param snapshot_interval: the interval for exporting a snapshot of the model
    """

    def __init__(
        self,
        split_merge_heuristics=-1,
        split_proposal=0,
        merge_proposal=0,
        split_merge_iteration=1,
        component_resampling_interval=100,
        restrict_gibbs_sampling_iteration=10,
        # gamma_shape_alpha=1,
        # gamma_scale_beta=1,
        hyper_parameter_interval=-1,
    ):
        self._split_merge_heuristics = split_merge_heuristics
        self._split_proposal = split_proposal
        self._merge_proposal = merge_proposal

        self._split_merge_iteration = split_merge_iteration
        self._component_resampling_interval = component_resampling_interval
        self._restrict_gibbs_sampling_iteration = restrict_gibbs_sampling_iteration

        # self._gamma_shape_alpha = gamma_shape_alpha
        # self._gamma_scale_beta = gamma_scale_beta
        self._hyper_parameter_interval = hyper_parameter_interval

    """
	@param data: a N-by-D numpy array object, defines N points of D dimension
	@param alpha: the concentration parameter of the dirichlet process
	@param kappa_0: initial kappa_0
	@param nu_0: initial nu_0
	@param mu_0: initial cluster center
	@param lambda_0: initial lambda_0
	"""

    def _initialize(
        self,
        data,
        alpha_alpha=1.0,
        # alpha_kappa=1.,
        # alpha_nu=1.,
        alpha_mu=None,
        alpha_sigma=None,
        initial_clusters=0,
    ):
        self._X = data
        self._N, self._D = self._X.shape

        # initialize the initial mean and sigma prior for the cluster
        if alpha_mu == None:
            # self._mu_0 = numpy.zeros((1, self._D))
            self._mu_0 = numpy.mean(self._X, axis=0)[numpy.newaxis, :]
        else:
            self._mu_0 = alpha_mu
        assert self._mu_0.shape == (1, self._D)

        if alpha_sigma == None:
            self._sigma_0 = numpy.eye(self._D)
        # self._sigma_0 = numpy.cov(self._X.T)
        else:
            self._sigma_0 = alpha_sigma
        assert self._sigma_0.shape == (self._D, self._D)

        # use slogdet instead of log(det(...)): at high dimension det() over/underflows to
        # +-inf/0 and log() then yields inf/nan, whereas slogdet returns log|det| directly.
        # slogdet itself can raise spurious FP-flag warnings on some numpy/LAPACK builds even
        # for well-conditioned inputs (the returned value is still correct), so silence locally.
        with numpy.errstate(over="ignore", divide="ignore", invalid="ignore"):
            self._log_sigma_det_0 = numpy.linalg.slogdet(self._sigma_0)[1]
        self._sigma_inv_0 = numpy.linalg.pinv(self._sigma_0)

        # initialize the concentration parameter of the dirichlet distribution
        self._alpha_alpha = alpha_alpha

        """
		# initialize every point to one cluster
		self._K = 1
		self._count = numpy.zeros(1, numpy.uint)
		self._count[0] = self._N
		self._label = numpy.zeros(self._N, numpy.uint)
		
		# compute the sum and square sum of all cluster up to truncation level
		self._sum = numpy.zeros((1, self._D))
		self._sum[0, :] = numpy.sum(self._X, 0)
		
		# initialize the sigma, inv(sigma) and log(det(sigma)) of all cluster up to truncation level
		self._sigma_inv = numpy.zeros((1, self._D, self._D))
		self._log_sigma_det = numpy.zeros(1)
		self._mu = numpy.zeros((1, self._D))
		"""

        self.random_initialization(initial_clusters)

        self._iteration_counter = 0

    def random_initialization(self, number_of_clusters=0):
        assert number_of_clusters <= self._N

        if number_of_clusters == 0:
            # initialize every point to its own cluster (K = N)
            self._label = numpy.arange(self._N)
        elif number_of_clusters == 1:
            # initialize all points to a single cluster
            self._label = numpy.zeros(self._N, dtype="int64")
        else:
            # randomly assign points into `number_of_clusters` clusters; force the first
            # `number_of_clusters` points to cover every label so no cluster starts empty
            self._label = numpy.random.randint(0, number_of_clusters, size=self._N)
            self._label[:number_of_clusters] = numpy.arange(number_of_clusters)

        self._K = len(numpy.unique(self._label))
        self._count = numpy.bincount(self._label)
        assert numpy.all(
            self._count > 0
        ), "initialization contains empty cluster, maybe try to reduce the number of clusters during initialization"
        assert numpy.sum(self._count) == self._N

        # initialize the sigma, inv(sigma) and log(det(sigma)) of all cluster up to truncation level
        self._sigma_inv = numpy.zeros((self._K, self._D, self._D))
        self._log_sigma_det = numpy.zeros(self._K)
        self._mu = numpy.zeros((self._K, self._D))

        # compute the sum and square sum of all cluster up to truncation level
        self._sum = numpy.zeros((self._K, self._D))
        for cluster_index in range(self._K):
            point_indices = numpy.nonzero(self._label == (cluster_index))[0]
            self._sum[cluster_index, :] = numpy.sum(self._X[point_indices, :], 0)

            # update the cluster parameters
            self.update_cluster_parameters(cluster_index)

    def learning(self):
        self._iteration_counter += 1

        self.sample_cgs()

        assert numpy.all(self._count > 0)

        if (
            self._hyper_parameter_interval > 0
            and self._iteration_counter % self._hyper_parameter_interval == 0
        ):
            self.optimize_hyperparameters()

        if self._split_merge_heuristics == 0:
            # self.resample_component()
            if self._iteration_counter % self._component_resampling_interval == 0:
                self.resample_component()
            # self.resample_components()
        elif self._split_merge_heuristics > 0:
            self.split_merge()

        assert numpy.sum(self._count) == self._N
        # compact all the parameters, including removing unused topics and unused tables
        # self.compact_params()

        print(
            "accumulated number of points for each cluster:",
            "[",
            " ".join("%d" % x for x in self._count),
            "]",
        )
        # print("accumulated number of tokens:", numpy.sum(self._n_kv, axis=1)[:, numpy.newaxis].T)

        return self.log_posterior()

    """
	sample the data to train the parameters
	"""

    def sample_cgs(self):
        # sample the total data
        for point_index in numpy.random.permutation(range(self._N)):
            assert self._count.shape == (self._K,)
            assert self._sum.shape == (self._K, self._D)
            assert self._mu.shape == (self._K, self._D)
            assert self._sigma_inv.shape == (self._K, self._D, self._D)
            assert self._log_sigma_det.shape == (self._K,)

            # get the old label of current point
            old_label = self._label[point_index]
            assert old_label < self._K and old_label >= 0, "%d\t%d" % (
                old_label,
                self._K,
            )

            # record down the inv(sigma) and log(det(sigma)) of the old cluster
            old_sigma_inv = self._sigma_inv[old_label, :, :]
            old_log_sigma_det = self._log_sigma_det[old_label]
            old_mu = self._mu[old_label, :]

            # remove the current point from the cluster
            self._count[old_label] -= 1
            self._label[point_index] = -1
            self._sum[old_label, :] -= self._X[point_index, :]

            if self._count[old_label] == 0:
                # if current point is from a singleton cluster, shift the last cluster to current one
                self._count[old_label] = self._count[self._K - 1]
                self._label[numpy.nonzero(self._label == (self._K - 1))] = old_label

                self._sum[old_label, :] = self._sum[self._K - 1, :]
                self._mu[old_label, :] = self._mu[self._K - 1, :]
                self._sigma_inv[old_label, :, :] = self._sigma_inv[self._K - 1, :, :]
                self._log_sigma_det[old_label] = self._log_sigma_det[self._K - 1]

                # remove the very last empty cluster, to remain compact cluster
                self._count = numpy.delete(self._count, [self._K - 1], axis=0)

                self._sum = numpy.delete(self._sum, [self._K - 1], axis=0)
                self._mu = numpy.delete(self._mu, [self._K - 1], axis=0)
                self._sigma_inv = numpy.delete(self._sigma_inv, [self._K - 1], axis=0)
                self._log_sigma_det = numpy.delete(
                    self._log_sigma_det, [self._K - 1], axis=0
                )

                self._K -= 1
                old_label = -1
            else:
                self.update_cluster_parameters(old_label)

            # compute the prior of being in any of the clusters
            cluster_prior = numpy.hstack((self._count[: self._K], self._alpha_alpha))
            cluster_prior = cluster_prior / (self._N - 1.0 + self._alpha_alpha)
            cluster_log_prior = numpy.log(cluster_prior)

            # initialize the likelihood vector for all clusters. Use -inf (probability 0), not 0
            # (probability 1): every slot is overwritten below, but -inf makes any future unfilled
            # slot fail safe -- as a zero-probability cluster -- rather than a phantom likely one.
            cluster_log_likelihood = numpy.full(self._K + 1, -numpy.inf)

            # compute the likelihood for new cluster
            mean_offset = self._X[[point_index], :] - self._mu_0
            assert mean_offset.shape == (1, self._D)

            cluster_log_likelihood[self._K] = -0.5 * self._log_sigma_det_0
            cluster_log_likelihood[self._K] += (
                -0.5
                * numpy.dot(
                    numpy.dot(mean_offset, self._sigma_inv_0), mean_offset.T
                ).item()
            )

            # compute the likelihood for the existing clusters, vectorized across all K clusters
            # instead of a Python loop with a matmul per cluster
            mean_offsets = self._X[point_index, :] - self._mu
            assert mean_offsets.shape == (self._K, self._D)
            # quadratic form off_k . sigma_inv_k . off_k for every cluster k, via batched matmul
            sigma_inv_offsets = numpy.matmul(
                mean_offsets[:, numpy.newaxis, :], self._sigma_inv
            )[:, 0, :]
            quadratic_forms = numpy.sum(sigma_inv_offsets * mean_offsets, axis=1)
            cluster_log_likelihood[: self._K] = (
                -0.5 * self._log_sigma_det - 0.5 * quadratic_forms
            )

            # normalize the posterior distribution
            cluster_log_posterior = cluster_log_prior + cluster_log_likelihood
            cluster_log_posterior -= scipy.special.logsumexp(cluster_log_posterior)
            cluster_posterior = numpy.exp(cluster_log_posterior)

            # sample a new cluster label for current point
            temp_label_probability = numpy.random.multinomial(1, cluster_posterior)[
                numpy.newaxis, :
            ]
            new_label = numpy.nonzero(temp_label_probability == 1)[1][0]
            # cdf = numpy.cumsum(cluster_posterior)
            # new_label = numpy.uint(numpy.nonzero(cdf >= numpy.random.random())[0][0])
            assert new_label >= 0 and new_label <= self._K, (
                cluster_posterior,
                temp_label_probability,
                new_label,
                new_label >= 0,
                new_label <= self._K,
            )

            # if this point starts up a new cluster
            if new_label == self._K:
                self._K += 1
                self._count = numpy.hstack((self._count, numpy.zeros(1)))

                self._sum = numpy.vstack((self._sum, numpy.zeros((1, self._D))))
                self._mu = numpy.vstack((self._mu, numpy.zeros((1, self._D))))
                self._sigma_inv = numpy.vstack(
                    (self._sigma_inv, numpy.zeros((1, self._D, self._D)))
                )
                self._log_sigma_det = numpy.hstack(
                    (self._log_sigma_det, numpy.zeros(1))
                )

            assert self._count.shape == (self._K,)
            assert self._sum.shape == (self._K, self._D)
            assert self._mu.shape == (self._K, self._D)
            assert self._sigma_inv.shape == (self._K, self._D, self._D)
            assert self._log_sigma_det.shape == (self._K,)
            assert new_label >= 0 and new_label < self._K

            self._label[point_index] = new_label
            self._count[new_label] += 1
            self._sum[new_label, :] += self._X[point_index, :]
            # self._square_sum[new_label, :, :] += numpy.dot(self._X[[point_index], :].transpose(), self._X[[point_index], :])

            if new_label == old_label:
                # if the point is allocated to the old cluster, retrieve all previous parameter
                self._sigma_inv[new_label, :, :] = old_sigma_inv
                self._log_sigma_det[new_label] = old_log_sigma_det
                self._mu[new_label, :] = old_mu
            else:
                self.update_cluster_parameters(new_label)

        return

    def inference(self, X_prime):
        N, D = X_prime.shape
        assert D == self._D

        label_prime = numpy.zeros(N) - 1
        log_likelihood_prime = 0

        assert self._count.shape == (self._K,)
        assert self._sum.shape == (self._K, self._D)
        assert self._mu.shape == (self._K, self._D)
        assert self._sigma_inv.shape == (self._K, self._D, self._D)
        assert self._log_sigma_det.shape == (self._K,)

        # compute the prior of being in any of the clusters
        cluster_log_prior = numpy.log(self._count + self._alpha_alpha)
        cluster_log_prior -= scipy.special.logsumexp(cluster_log_prior)

        # sample the entire dataset
        for point_index in numpy.random.permutation(range(N)):
            # initialize the likelihood vector for all clusters
            cluster_log_likelihood = numpy.zeros(self._K)

            # compute the likelihood for the existing clusters, vectorized across all K clusters
            mean_offsets = X_prime[point_index, :] - self._mu
            assert mean_offsets.shape == (self._K, self._D)
            sigma_inv_offsets = numpy.matmul(
                mean_offsets[:, numpy.newaxis, :], self._sigma_inv
            )[:, 0, :]
            quadratic_forms = numpy.sum(sigma_inv_offsets * mean_offsets, axis=1)
            cluster_log_likelihood[:] = (
                -0.5 * self._log_sigma_det - 0.5 * quadratic_forms
            )

            # normalize the posterior distribution
            cluster_log_posterior = cluster_log_prior + cluster_log_likelihood
            cluster_log_posterior -= scipy.special.logsumexp(cluster_log_posterior)
            cluster_posterior = numpy.exp(cluster_log_posterior)

            # sample a new cluster label for current point
            temp_label_probability = numpy.random.multinomial(1, cluster_posterior)[
                numpy.newaxis, :
            ]
            new_label = numpy.nonzero(temp_label_probability == 1)[1][0]
            assert new_label >= 0 and new_label < self._K, (
                cluster_posterior,
                temp_label_probability,
                new_label,
                new_label >= 0,
                new_label < self._K,
            )

            label_prime[point_index] = new_label
            log_likelihood_prime += (
                cluster_log_prior[new_label] + cluster_log_likelihood[new_label]
            )

        assert numpy.all(label_prime >= 0)
        assert numpy.all(label_prime < self._K)

        return label_prime, log_likelihood_prime

    """
	"""

    def optimize_hyperparameters(
        self,
        hyperparameter_samples=10,
        hyperparameter_step_size=1.0,
        hyperparameter_maximum_iteration=10,
    ):
        old_log_alpha_alpha = numpy.log(self._alpha_alpha)

        for ii in range(hyperparameter_samples):
            log_likelihood_old = self.log_posterior()
            log_likelihood_new = numpy.log(numpy.random.random()) + log_likelihood_old
            # print("OLD: %f\tNEW: %f at (%f, %f)" % (log_likelihood_old, log_likelihood_new, self._alpha, self._beta))

            l = old_log_alpha_alpha - numpy.random.random() * hyperparameter_step_size
            r = old_log_alpha_alpha + hyperparameter_step_size

            for jj in range(hyperparameter_maximum_iteration):
                new_log_alpha_alpha = l + numpy.random.random() * (r - l)
                lp_test = self.log_posterior(None, numpy.exp(new_log_alpha_alpha))

                if lp_test > log_likelihood_new:
                    self._alpha_alpha = numpy.exp(new_log_alpha_alpha)
                    old_log_alpha_alpha = numpy.log(self._alpha_alpha)
                    break
                else:
                    if new_log_alpha_alpha < old_log_alpha_alpha:
                        l = new_log_alpha_alpha
                    else:
                        r = new_log_alpha_alpha
                    assert l <= old_log_alpha_alpha
                    assert r >= old_log_alpha_alpha

            print("update hyperparameter to %f" % (numpy.exp(new_log_alpha_alpha)))

    def split_merge(self):
        for iteration in range(self._split_merge_iteration):
            label_probability = 1.0 * self._count / numpy.sum(self._count)

            if self._split_merge_heuristics == 1:
                temp_label_probability = numpy.random.multinomial(1, label_probability)[
                    numpy.newaxis, :
                ]
                random_label_1 = numpy.nonzero(temp_label_probability == 1)[1][0]
                temp_label_probability = numpy.random.multinomial(1, label_probability)[
                    numpy.newaxis, :
                ]
                random_label_2 = numpy.nonzero(temp_label_probability == 1)[1][0]
            elif self._split_merge_heuristics == 2:
                random_label_1 = numpy.random.randint(self._K)
                temp_label_probability = numpy.random.multinomial(1, label_probability)[
                    numpy.newaxis, :
                ]
                random_label_2 = numpy.nonzero(temp_label_probability == 1)[1][0]
            elif self._split_merge_heuristics == 3:
                random_label_1 = numpy.random.randint(self._K)
                random_label_2 = numpy.random.randint(self._K)
            else:
                sys.stderr.write(
                    "error: unrecognized split-merge heuristics %d...\n"
                    % (self._split_merge_heuristics)
                )
                return

            if random_label_1 == random_label_2:
                self.split_metropolis_hastings(random_label_1)
                assert numpy.all(self._count > 0)
            else:
                self.merge_metropolis_hastings(random_label_1, random_label_2)
                assert numpy.all(self._count > 0)

    """
	def split_mh_merge_gs(self):
		if self._split_merge_heuristics==0:
			return
		
		for iteration in range(self._split_merge_iteration):
			label_probability = 1.0 * self._count / numpy.sum(self._count)
			cluster_index = numpy.random.randint(0, self._K)
			
			if self._split_merge_heuristics==1:
				split_probability = label_probability[cluster_index]
			elif self._split_merge_heuristics==2:
				split_probability = 1.0/self._K
			
			if numpy.random.random()<split_probability:
				# perform a split operation
				self.split_metropolis_hastings(cluster_index)
			else:
				# perform a merge operation
				self.resample_component(cluster_index)
			
		return
	"""

    def update_cluster_parameters(self, cluster_id, model_parameter=None):
        if model_parameter == None:
            label = self._label
            K = self._K
            count = self._count
            mu = self._mu
            sum = self._sum
            log_sigma_det = self._log_sigma_det
            sigma_inv = self._sigma_inv
        else:
            label, K, count, mu, sum, log_sigma_det, sigma_inv = model_parameter

        # update the covariance matrix and mu for the old cluster
        if count[cluster_id] == 1:
            # if there is only one point remain in the cluster
            # set its covariance matrix to the hyper cluster
            temp_sigma = self._sigma_0
            temp_sigma_inv = self._sigma_inv_0
        else:
            # if there are more than one point in the cluster
            # adjust its covariance matrix
            points_in_cluster = self._X[numpy.nonzero(label == cluster_id)[0], :]
            assert points_in_cluster.shape == (count[cluster_id], self._D), (
                points_in_cluster.shape,
                count[cluster_id],
            )
            temp_sigma = numpy.cov(points_in_cluster.T)
            # ridge-regularize: collinear/duplicate points, or a cluster with fewer points than
            # dimensions (n < D), make the empirical covariance singular. That destabilizes the
            # pinv below and can drive slogdet(sigma_hat) to -inf, breaking the multinomial sampling.
            if self._D > 1:
                temp_sigma += numpy.eye(self._D) * 1e-6
            else:
                temp_sigma += 1e-6
            temp_sigma_inv = numpy.linalg.pinv(temp_sigma)

        # compute n*\Sigma^{-1}
        temp_a = count[cluster_id] * temp_sigma_inv
        # compute \Sigma_{0}^{-1} + n*\Sigma^{-1}
        temp_b = self._sigma_inv_0 + temp_a
        # compute the posterior covariance of the mean, Sigma_n = (\Sigma_{0}^{-1} + n*\Sigma^{-1})^{-1}
        temp_c = numpy.linalg.pinv(temp_b)
        # posterior-predictive covariance for a new point is Sigma + Sigma_n (data noise + mean
        # uncertainty); use temp_sigma (the cluster's Sigma), NOT self._sigma_0 -- otherwise a large
        # cluster's predictive shape collapses to the global prior and ignores its own spread.
        sigma_hat = temp_sigma + temp_c
        assert sigma_hat.shape == (self._D, self._D)

        sigma_inv[cluster_id, :, :] = numpy.linalg.pinv(sigma_hat)
        # slogdet is numerically stable at high dimension where det() overflows (see _initialize)
        with numpy.errstate(over="ignore", divide="ignore", invalid="ignore"):
            log_sigma_det[cluster_id] = numpy.linalg.slogdet(sigma_hat)[1]

        # compute \Sigma_{0}^{-1} \mu_{0}
        temp_d = numpy.dot(self._sigma_inv_0, self._mu_0.T)
        # compute \Sigma_{0}^{-1} \mu_{0} + \Sigma^{-1} n \bar{x}
        temp_e = temp_d + numpy.dot(temp_sigma_inv, sum[[cluster_id], :].T)
        mu_hat = numpy.dot(temp_c, temp_e).T
        assert mu_hat.shape == (1, self._D)

        mu[cluster_id, :] = mu_hat[0, :]

        if model_parameter == None:
            self._label = label
            self._K = K
            self._count = count
            self._mu = mu
            self._sum = sum
            self._log_sigma_det = log_sigma_det
            self._sigma_inv = sigma_inv
        else:
            model_parameter = (label, K, count, mu, sum, log_sigma_det, sigma_inv)
            return model_parameter

    def split_metropolis_hastings(self, cluster_label):
        # record down the old cluster assignment
        old_log_posterior = self.log_posterior()

        proposed_K = self._K
        proposed_label = numpy.copy(self._label)
        proposed_count = numpy.copy(self._count)
        proposed_mu = numpy.copy(self._mu)
        proposed_sum = numpy.copy(self._sum)
        proposed_sigma_inv = numpy.copy(self._sigma_inv)
        proposed_log_sigma_det = numpy.copy(self._log_sigma_det)

        model_parameter = (
            proposed_label,
            proposed_K,
            proposed_count,
            proposed_mu,
            proposed_sum,
            proposed_log_sigma_det,
            proposed_sigma_inv,
        )

        if self._split_proposal == 0:
            # perform random split for split proposal
            model_parameter = self.random_split(cluster_label, model_parameter)

            if model_parameter == None:
                return

            (
                proposed_label,
                proposed_K,
                proposed_count,
                proposed_mu,
                proposed_sum,
                proposed_log_sigma_det,
                proposed_sigma_inv,
            ) = model_parameter
            assert numpy.all(proposed_count > 0)

            log_proposal_probability = (
                proposed_count[cluster_label] + proposed_count[proposed_K - 1] - 2
            ) * numpy.log(2)
        elif self._split_proposal == 1:
            # perform restricted gibbs sampling for split proposal
            model_parameter = self.random_split(cluster_label, model_parameter)
            # split a singleton cluster
            if model_parameter == None:
                return
            (
                proposed_label,
                proposed_K,
                old_proposed_count,
                proposed_mu,
                proposed_sum,
                proposed_log_sigma_det,
                proposed_sigma_inv,
            ) = model_parameter

            # self.model_assertion(model_parameter)
            model_parameter, transition_log_likelihood = self.restrict_gibbs_sampling(
                cluster_label,
                proposed_K - 1,
                model_parameter,
                self._restrict_gibbs_sampling_iteration + 1,
            )
            (
                proposed_label,
                proposed_K,
                proposed_count,
                proposed_mu,
                proposed_sum,
                proposed_log_sigma_det,
                proposed_sigma_inv,
            ) = model_parameter
            # self.model_assertion(model_parameter)

            if (
                proposed_count[cluster_label] == 0
                or proposed_count[proposed_K - 1] == 0
            ):
                return

            assert numpy.all(proposed_count > 0), (
                proposed_count,
                old_proposed_count,
                cluster_label,
                proposed_K - 1,
            )

            log_proposal_probability = transition_log_likelihood
        elif self._split_proposal == 2:
            # perform sequential allocation gibbs sampling for split proposal
            model_parameter = self.sequential_allocation_split(
                cluster_label, model_parameter
            )

            if model_parameter == None:
                return

            (
                proposed_label,
                proposed_K,
                proposed_count,
                proposed_mu,
                proposed_sum,
                proposed_log_sigma_det,
                proposed_sigma_inv,
            ) = model_parameter
            assert numpy.all(proposed_count > 0)

            log_proposal_probability = (
                proposed_count[cluster_label] + proposed_count[proposed_K - 1] - 2
            ) * numpy.log(2)
        else:
            sys.stderr.write(
                "error: unrecognized split proposal strategy %d...\n"
                % (self._split_proposal)
            )

        # model_parameter = (proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det, proposed_sigma_inv)
        new_log_posterior = self.log_posterior(model_parameter)

        acceptance_log_probability = (
            log_proposal_probability + new_log_posterior - old_log_posterior
        )
        acceptance_log_probability -= scipy.special.logsumexp(
            acceptance_log_probability
        )
        acceptance_probability = numpy.exp(acceptance_log_probability)

        (
            proposed_label,
            proposed_K,
            proposed_count,
            proposed_mu,
            proposed_sum,
            proposed_log_sigma_det,
            proposed_sigma_inv,
        ) = model_parameter

        if numpy.random.random() < acceptance_probability:
            print(
                "split operation granted from %s to %s with acceptance probability %s"
                % (self._count, proposed_count, acceptance_probability)
            )

            self._K = proposed_K
            self._label = proposed_label

            self._count = proposed_count
            self._sum = proposed_sum

            self._mu = proposed_mu
            self._sigma_inv = proposed_sigma_inv
            self._log_sigma_det = proposed_log_sigma_det

        assert self._count.shape == (self._K,), (self._count.shape, self._K)
        assert self._sum.shape == (self._K, self._D)
        assert self._mu.shape == (self._K, self._D)
        assert self._sigma_inv.shape == (self._K, self._D, self._D)
        assert self._log_sigma_det.shape == (self._K,)
        assert numpy.all(self._count > 0), self._count

        return

    def random_split(self, cluster_label, model_parameter):
        # sample the data points set
        (
            proposed_label,
            proposed_K,
            proposed_count,
            proposed_mu,
            proposed_sum,
            proposed_log_sigma_det,
            proposed_sigma_inv,
        ) = model_parameter

        # perform a split operation
        data_point_indices = numpy.nonzero(proposed_label == cluster_label)[0]

        proposed_count = numpy.hstack((proposed_count, numpy.zeros(1)))
        proposed_sum = numpy.vstack((proposed_sum, numpy.zeros((1, self._D))))
        proposed_K += 1
        for data_point_index in data_point_indices:
            # random split the current cluster into two
            if numpy.random.random() < 0.5:
                proposed_label[data_point_index] = proposed_K - 1
                proposed_count[proposed_K - 1] += 1
                proposed_sum[proposed_K - 1, :] += self._X[data_point_index, :]
                proposed_count[cluster_label] -= 1
                proposed_sum[cluster_label, :] -= self._X[data_point_index, :]

        # this is to make sure check we don't split a singleton cluster
        if proposed_count[cluster_label] == 0 or proposed_count[proposed_K - 1] == 0:
            return None

        proposed_mu = numpy.vstack((proposed_mu, numpy.zeros((1, self._D))))
        proposed_sigma_inv = numpy.vstack(
            (proposed_sigma_inv, numpy.zeros((1, self._D, self._D)))
        )
        proposed_log_sigma_det = numpy.hstack((proposed_log_sigma_det, numpy.zeros(1)))

        model_parameter = (
            proposed_label,
            proposed_K,
            proposed_count,
            proposed_mu,
            proposed_sum,
            proposed_log_sigma_det,
            proposed_sigma_inv,
        )
        model_parameter = self.update_cluster_parameters(cluster_label, model_parameter)
        model_parameter = self.update_cluster_parameters(
            proposed_K - 1, model_parameter
        )

        return model_parameter

    def sequential_allocation_split(self, cluster_label, model_parameter):
        # sample the data points set
        (
            proposed_label,
            proposed_K,
            proposed_count,
            proposed_mu,
            proposed_sum,
            proposed_log_sigma_det,
            proposed_sigma_inv,
        ) = model_parameter

        data_point_indices = numpy.nonzero(proposed_label == cluster_label)[0]
        data_point_indices_list = list(data_point_indices)

        if len(data_point_indices_list) < 2:
            return None

        # randomly choose two points and initialize the cluster
        random.shuffle(data_point_indices_list)

        # clear current cluster
        proposed_label[data_point_indices_list] = -1
        proposed_count[cluster_label] = 0
        proposed_sum[cluster_label, :] = 0
        proposed_mu[cluster_label, :] = 0
        proposed_sigma_inv[cluster_label, :, :] = self._sigma_inv_0
        proposed_log_sigma_det[cluster_label] = self._log_sigma_det_0

        # create a new cluster
        proposed_count = numpy.hstack((proposed_count, numpy.zeros(1)))
        proposed_sum = numpy.vstack((proposed_sum, numpy.zeros((1, self._D))))
        proposed_mu = numpy.vstack((proposed_mu, numpy.zeros((1, self._D))))
        proposed_sigma_inv = numpy.vstack(
            (proposed_sigma_inv, numpy.zeros((1, self._D, self._D)))
        )
        proposed_log_sigma_det = numpy.hstack((proposed_log_sigma_det, numpy.zeros(1)))
        proposed_K += 1

        # initialize the existing cluster
        candidate_point_1 = data_point_indices_list.pop()
        proposed_label[candidate_point_1] = cluster_label
        proposed_count[cluster_label] = 1
        proposed_sum[cluster_label, :] = self._X[candidate_point_1, :]

        # initialize the new cluster
        candidate_point_2 = data_point_indices_list.pop()
        proposed_label[candidate_point_2] = proposed_K - 1
        proposed_count[proposed_K - 1] = 1
        proposed_sum[proposed_K - 1, :] = self._X[candidate_point_2, :]

        # update the cluster parameters
        model_parameter = (
            proposed_label,
            proposed_K,
            proposed_count,
            proposed_mu,
            proposed_sum,
            proposed_log_sigma_det,
            proposed_sigma_inv,
        )
        model_parameter = self.update_cluster_parameters(cluster_label, model_parameter)
        model_parameter = self.update_cluster_parameters(
            proposed_K - 1, model_parameter
        )
        (
            proposed_label,
            proposed_K,
            proposed_count,
            proposed_mu,
            proposed_sum,
            proposed_log_sigma_det,
            proposed_sigma_inv,
        ) = model_parameter

        # perform a split operation
        for data_point_index in data_point_indices_list:
            # compute the probability of being in current cluster
            mean_offset = (
                self._X[[data_point_index], :] - proposed_mu[[cluster_label], :]
            )
            assert mean_offset.shape == (1, self._D)
            cluster_log_probability_1 = -0.5 * proposed_log_sigma_det[cluster_label]
            cluster_log_probability_1 += (
                -0.5
                * numpy.dot(
                    numpy.dot(mean_offset, proposed_sigma_inv[cluster_label, :, :]),
                    mean_offset.T,
                ).item()
            )
            cluster_log_probability_1 += numpy.log(proposed_count[cluster_label])

            # compute the probability of being in cluster 2
            mean_offset = (
                self._X[[data_point_index], :] - proposed_mu[[proposed_K - 1], :]
            )
            assert mean_offset.shape == (1, self._D)
            cluster_log_probability_2 = -0.5 * proposed_log_sigma_det[proposed_K - 1]
            cluster_log_probability_2 += (
                -0.5
                * numpy.dot(
                    numpy.dot(mean_offset, proposed_sigma_inv[proposed_K - 1, :, :]),
                    mean_offset.T,
                ).item()
            )
            cluster_log_probability_2 += numpy.log(proposed_count[proposed_K - 1])

            log_ratio_2_over_1 = cluster_log_probability_2 - cluster_log_probability_1
            log_ratio_2_over_1 -= scipy.special.logsumexp(log_ratio_2_over_1)
            ratio_2_over_1 = numpy.exp(log_ratio_2_over_1)

            # sample a new cluster label for current point
            cluster_probability_1 = 1.0 / (1.0 + ratio_2_over_1)
            if numpy.random.random() <= cluster_probability_1:
                new_label = cluster_label
            else:
                new_label = proposed_K - 1

            proposed_label[data_point_index] = new_label
            proposed_count[new_label] += 1
            proposed_sum[new_label, :] += self._X[data_point_index, :]

            model_parameter = (
                proposed_label,
                proposed_K,
                proposed_count,
                proposed_mu,
                proposed_sum,
                proposed_log_sigma_det,
                proposed_sigma_inv,
            )
            model_parameter = self.update_cluster_parameters(new_label, model_parameter)
            (
                proposed_label,
                proposed_K,
                proposed_count,
                proposed_mu,
                proposed_sum,
                proposed_log_sigma_det,
                proposed_sigma_inv,
            ) = model_parameter

            assert proposed_count.shape == (proposed_K,)
            assert proposed_sum.shape == (proposed_K, self._D)
            assert proposed_mu.shape == (proposed_K, self._D)
            assert proposed_sigma_inv.shape == (proposed_K, self._D, self._D)
            assert proposed_log_sigma_det.shape == (proposed_K,)
            assert new_label == cluster_label or new_label == proposed_K - 1

        assert proposed_count[cluster_label] > 0 and proposed_count[proposed_K - 1] > 0

        model_parameter = (
            proposed_label,
            proposed_K,
            proposed_count,
            proposed_mu,
            proposed_sum,
            proposed_log_sigma_det,
            proposed_sigma_inv,
        )
        return model_parameter

    def restrict_gibbs_sampling(
        self,
        cluster_index_1,
        cluster_index_2,
        model_parameter,
        restricted_gibbs_sampling_iteration=1,
    ):
        (
            proposed_label,
            proposed_K,
            proposed_count,
            proposed_mu,
            proposed_sum,
            proposed_log_sigma_det,
            proposed_sigma_inv,
        ) = model_parameter
        data_point_indices = numpy.hstack(
            (
                numpy.nonzero(proposed_label == cluster_index_2)[0],
                numpy.nonzero(proposed_label == cluster_index_1)[0],
            )
        )

        assert (
            len(data_point_indices)
            == proposed_count[cluster_index_1] + proposed_count[cluster_index_2]
        ), (
            len(data_point_indices),
            proposed_count[cluster_index_1],
            proposed_count[cluster_index_2],
        )

        # sample the data points set
        for restrict_gibbs_sampling_iteration_index in range(
            restricted_gibbs_sampling_iteration
        ):
            transition_log_likelihood = 0
            for point_index in data_point_indices:
                # get the old label of current point
                old_label = proposed_label[point_index]
                assert old_label == cluster_index_1 or old_label == cluster_index_2, (
                    old_label,
                    cluster_index_1,
                    cluster_index_2,
                    point_index,
                )

                # record down the inv(sigma) and log(det(sigma)) of the old cluster
                old_sigma_inv = proposed_sigma_inv[old_label, :, :]
                old_log_sigma_det = proposed_log_sigma_det[old_label]
                old_mu = proposed_mu[old_label, :]

                # remove the current point from the cluster
                proposed_count[old_label] -= 1
                proposed_label[point_index] = -1
                proposed_sum[old_label, :] -= self._X[point_index, :]
                assert numpy.all(proposed_count >= 0), proposed_count

                if proposed_count[old_label] == 0:
                    proposed_mu[old_label, :] = self._mu_0[0, :]
                    proposed_log_sigma_det[old_label] = self._log_sigma_det_0
                    proposed_sigma_inv[old_label, :, :] = self._sigma_inv_0
                else:
                    model_parameter = (
                        proposed_label,
                        proposed_K,
                        proposed_count,
                        proposed_mu,
                        proposed_sum,
                        proposed_log_sigma_det,
                        proposed_sigma_inv,
                    )
                    model_parameter = self.update_cluster_parameters(
                        old_label, model_parameter
                    )
                    (
                        proposed_label,
                        proposed_K,
                        proposed_count,
                        proposed_mu,
                        proposed_sum,
                        proposed_log_sigma_det,
                        proposed_sigma_inv,
                    ) = model_parameter

                # compute the probability of being in cluster 1
                mean_offset = (
                    self._X[[point_index], :] - proposed_mu[[cluster_index_1], :]
                )
                assert mean_offset.shape == (1, self._D)
                cluster_log_probability_1 = (
                    -0.5 * proposed_log_sigma_det[cluster_index_1]
                )
                cluster_log_probability_1 += (
                    -0.5
                    * numpy.dot(
                        numpy.dot(
                            mean_offset, proposed_sigma_inv[cluster_index_1, :, :]
                        ),
                        mean_offset.T,
                    ).item()
                )
                if proposed_count[cluster_index_1] == 0:
                    cluster_log_probability_1 += numpy.log(self._alpha_alpha)
                else:
                    cluster_log_probability_1 += numpy.log(
                        proposed_count[cluster_index_1]
                    )

                # compute the probability of being in cluster 2
                mean_offset = (
                    self._X[[point_index], :] - proposed_mu[[cluster_index_2], :]
                )
                assert mean_offset.shape == (1, self._D)
                cluster_log_probability_2 = (
                    -0.5 * proposed_log_sigma_det[cluster_index_2]
                )
                cluster_log_probability_2 += (
                    -0.5
                    * numpy.dot(
                        numpy.dot(
                            mean_offset, proposed_sigma_inv[cluster_index_2, :, :]
                        ),
                        mean_offset.T,
                    ).item()
                )
                if proposed_count[cluster_index_2] == 0:
                    cluster_log_probability_2 += numpy.log(self._alpha_alpha)
                else:
                    cluster_log_probability_2 += numpy.log(
                        proposed_count[cluster_index_2]
                    )

                # sample a new cluster label for current point
                ratio_2_over_1 = numpy.exp(
                    cluster_log_probability_2 - cluster_log_probability_1
                )
                cluster_probability_1 = 1.0 / (1.0 + ratio_2_over_1)
                if numpy.random.random() <= cluster_probability_1:
                    new_label = cluster_index_1
                    transition_log_likelihood += numpy.log(cluster_probability_1)
                else:
                    new_label = cluster_index_2
                    transition_log_likelihood += numpy.log(1 - cluster_probability_1)

                proposed_label[point_index] = new_label
                proposed_count[new_label] += 1
                proposed_sum[new_label, :] += self._X[point_index, :]
                if new_label == old_label:
                    # if the point is allocated to the old cluster, retrieve all previous parameter
                    proposed_sigma_inv[new_label, :, :] = old_sigma_inv
                    proposed_log_sigma_det[new_label] = old_log_sigma_det
                    proposed_mu[new_label, :] = old_mu
                # assert numpy.all(proposed_count>0), (proposed_count, new_label, old_label, cluster_index_1, cluster_index_2)
                else:
                    model_parameter = (
                        proposed_label,
                        proposed_K,
                        proposed_count,
                        proposed_mu,
                        proposed_sum,
                        proposed_log_sigma_det,
                        proposed_sigma_inv,
                    )
                    model_parameter = self.update_cluster_parameters(
                        new_label, model_parameter
                    )
                    (
                        proposed_label,
                        proposed_K,
                        proposed_count,
                        proposed_mu,
                        proposed_sum,
                        proposed_log_sigma_det,
                        proposed_sigma_inv,
                    ) = model_parameter

                assert proposed_count.shape == (proposed_K,)
                assert proposed_sum.shape == (proposed_K, self._D)
                assert proposed_mu.shape == (proposed_K, self._D)
                assert proposed_sigma_inv.shape == (proposed_K, self._D, self._D)
                assert proposed_log_sigma_det.shape == (proposed_K,)
                assert new_label == cluster_index_1 or new_label == cluster_index_2
                assert numpy.all(proposed_count >= 0)

        model_parameter = (
            proposed_label,
            proposed_K,
            proposed_count,
            proposed_mu,
            proposed_sum,
            proposed_log_sigma_det,
            proposed_sigma_inv,
        )
        return model_parameter, transition_log_likelihood

    def merge_metropolis_hastings(self, cluster_label_1, cluster_label_2):
        old_log_posterior = self.log_posterior()

        # this is to switch the label, make sure we always
        if cluster_label_1 > cluster_label_2:
            temp_random_label = cluster_label_1
            cluster_label_1 = cluster_label_2
            cluster_label_2 = temp_random_label

        proposed_K = self._K
        proposed_label = numpy.copy(self._label)
        proposed_count = numpy.copy(self._count)
        proposed_mu = numpy.copy(self._mu)
        proposed_sum = numpy.copy(self._sum)
        proposed_sigma_inv = numpy.copy(self._sigma_inv)
        proposed_log_sigma_det = numpy.copy(self._log_sigma_det)

        assert numpy.sum(proposed_count) == self._N

        model_parameter = (
            proposed_label,
            proposed_K,
            proposed_count,
            proposed_mu,
            proposed_sum,
            proposed_log_sigma_det,
            proposed_sigma_inv,
        )

        if self._merge_proposal == 0:
            # perform random merge for merge proposal
            model_parameter = self.random_merge(
                cluster_label_1, cluster_label_2, model_parameter
            )

            (
                proposed_label,
                proposed_K,
                proposed_count,
                proposed_mu,
                proposed_sum,
                proposed_log_sigma_det,
                proposed_sigma_inv,
            ) = model_parameter
            assert numpy.all(proposed_count > 0)

            log_proposal_probability = -(
                proposed_count[cluster_label_1] - 2
            ) * numpy.log(2)
        elif self._merge_proposal == 1:
            # perform restricted gibbs sampling for merge proposal
            model_parameter, transition_log_probability = self.restrict_gibbs_sampling(
                cluster_label_1,
                cluster_label_2,
                model_parameter,
                self._restrict_gibbs_sampling_iteration + 1,
            )

            (
                proposed_label,
                proposed_K,
                proposed_count,
                proposed_mu,
                proposed_sum,
                proposed_log_sigma_det,
                proposed_sigma_inv,
            ) = model_parameter
            assert numpy.all(proposed_count >= 0)
            assert numpy.sum(proposed_count) == self._N
            assert numpy.sum(self._count) == self._N

            if (
                proposed_count[cluster_label_1] == 0
                or proposed_count[cluster_label_2] == 0
            ):
                print(
                    "merge cluster %d and %d during restricted gibbs sampling step..."
                    % (cluster_label_1, cluster_label_2)
                )

                if proposed_count[cluster_label_1] == 0:
                    collapsed_cluster = cluster_label_1
                elif proposed_count[cluster_label_2] == 0:
                    collapsed_cluster = cluster_label_2

                # since one cluster is empty now, switch it with the last one
                proposed_count[collapsed_cluster] = proposed_count[proposed_K - 1]
                proposed_label[numpy.nonzero(proposed_label == (proposed_K - 1))] = (
                    collapsed_cluster
                )
                proposed_sum[collapsed_cluster, :] = proposed_sum[proposed_K - 1, :]
                proposed_mu[collapsed_cluster, :] = proposed_mu[proposed_K - 1, :]
                proposed_sigma_inv[collapsed_cluster, :, :] = proposed_sigma_inv[
                    proposed_K - 1, :, :
                ]
                proposed_log_sigma_det[collapsed_cluster] = proposed_log_sigma_det[
                    proposed_K - 1
                ]

                # remove the very last empty cluster, to remain compact cluster
                proposed_count = numpy.delete(proposed_count, [proposed_K - 1], axis=0)
                proposed_sum = numpy.delete(proposed_sum, [proposed_K - 1], axis=0)
                proposed_mu = numpy.delete(proposed_mu, [proposed_K - 1], axis=0)
                proposed_sigma_inv = numpy.delete(
                    proposed_sigma_inv, [proposed_K - 1], axis=0
                )
                proposed_log_sigma_det = numpy.delete(
                    proposed_log_sigma_det, [proposed_K - 1], axis=0
                )
                proposed_K -= 1

                # print proposed_count
                # (proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det, proposed_sigma_inv) = model_parameter
                # print proposed_count
                model_parameter = (
                    proposed_label,
                    proposed_K,
                    proposed_count,
                    proposed_mu,
                    proposed_sum,
                    proposed_log_sigma_det,
                    proposed_sigma_inv,
                )

            log_proposal_probability = transition_log_probability
        elif self._merge_proposal == 2:
            # perform gibbs sampling for merge proposal
            cluster_log_probability = numpy.log(proposed_count)
            cluster_log_probability = (
                numpy.sum(cluster_log_probability) - cluster_log_probability
            )
            cluster_log_probability -= scipy.special.logsumexp(cluster_log_probability)
            cluster_probability = numpy.exp(cluster_log_probability)

            # choose a cluster that is inverse proportional to its size
            temp_cluster_probability = numpy.random.multinomial(1, cluster_probability)[
                numpy.newaxis, :
            ]
            cluster_label = numpy.nonzero(temp_cluster_probability == 1)[1][0]

            model_parameter = self.gibbs_sampling_merge(cluster_label, model_parameter)
            if model_parameter == None:
                return

            (
                proposed_label,
                proposed_K,
                proposed_count,
                proposed_mu,
                proposed_sum,
                proposed_log_sigma_det,
                proposed_sigma_inv,
            ) = model_parameter
            assert numpy.all(proposed_count > 0)

            self._K = proposed_K
            self._label = proposed_label

            self._count = proposed_count
            self._sum = proposed_sum

            self._mu = proposed_mu
            self._sigma_inv = proposed_sigma_inv
            self._log_sigma_det = proposed_log_sigma_det

            assert self._count.shape == (self._K,), (self._count.shape, self._K)
            assert self._sum.shape == (self._K, self._D)
            assert self._mu.shape == (self._K, self._D)
            assert self._sigma_inv.shape == (self._K, self._D, self._D)
            assert self._log_sigma_det.shape == (self._K,)

            return
        else:
            sys.stderr.write(
                "error: unrecognized merge proposal strategy %d...\n"
                % (self._merge_proposal)
            )

        assert proposed_K == len(proposed_log_sigma_det)
        assert numpy.max(proposed_label) < len(proposed_log_sigma_det)
        assert numpy.sum(proposed_count) == self._N
        assert proposed_mu.shape == (proposed_K, self._D)

        # model_parameter = (proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det, proposed_sigma_inv)
        new_log_posterior = self.log_posterior(model_parameter)

        acceptance_log_probability = (
            log_proposal_probability + new_log_posterior - old_log_posterior
        )
        acceptance_log_probability -= scipy.special.logsumexp(
            acceptance_log_probability
        )
        acceptance_probability = numpy.exp(acceptance_log_probability)

        (
            proposed_label,
            proposed_K,
            proposed_count,
            proposed_mu,
            proposed_sum,
            proposed_log_sigma_det,
            proposed_sigma_inv,
        ) = model_parameter
        assert numpy.all(proposed_count > 0)

        if numpy.random.random() < acceptance_probability:
            print(
                "merge operation granted from %s to %s with acceptance probability %s"
                % (self._count, proposed_count, acceptance_probability)
            )

            self._K = proposed_K
            self._label = proposed_label

            self._count = proposed_count
            self._sum = proposed_sum

            self._mu = proposed_mu
            self._sigma_inv = proposed_sigma_inv
            self._log_sigma_det = proposed_log_sigma_det

        assert self._count.shape == (self._K,), (self._count.shape, self._K)
        assert self._sum.shape == (self._K, self._D)
        assert self._mu.shape == (self._K, self._D)
        assert self._sigma_inv.shape == (self._K, self._D, self._D)
        assert self._log_sigma_det.shape == (self._K,)

    def random_merge(self, cluster_label_1, cluster_label_2, model_parameter):
        assert cluster_label_2 > cluster_label_1

        # sample the data points set
        (
            proposed_label,
            proposed_K,
            proposed_count,
            proposed_mu,
            proposed_sum,
            proposed_log_sigma_det,
            proposed_sigma_inv,
        ) = model_parameter

        # perform a merge operation
        proposed_label[numpy.nonzero(proposed_label == cluster_label_2)[0]] = (
            cluster_label_1
        )
        proposed_count[cluster_label_1] += proposed_count[cluster_label_2]
        proposed_sum[cluster_label_1, :] += proposed_sum[cluster_label_2, :]

        # since one cluster is empty now, switch it with the last one
        proposed_count[cluster_label_2] = proposed_count[proposed_K - 1]
        proposed_label[numpy.nonzero(proposed_label == (proposed_K - 1))] = (
            cluster_label_2
        )
        proposed_sum[cluster_label_2, :] = proposed_sum[proposed_K - 1, :]
        proposed_mu[cluster_label_2, :] = proposed_mu[proposed_K - 1, :]
        proposed_sigma_inv[cluster_label_2, :, :] = proposed_sigma_inv[
            proposed_K - 1, :, :
        ]
        proposed_log_sigma_det[cluster_label_2] = proposed_log_sigma_det[proposed_K - 1]

        # remove the very last empty cluster, to remain compact cluster
        proposed_count = numpy.delete(proposed_count, [proposed_K - 1], axis=0)
        proposed_sum = numpy.delete(proposed_sum, [proposed_K - 1], axis=0)
        proposed_mu = numpy.delete(proposed_mu, [proposed_K - 1], axis=0)
        proposed_sigma_inv = numpy.delete(proposed_sigma_inv, [proposed_K - 1], axis=0)
        proposed_log_sigma_det = numpy.delete(
            proposed_log_sigma_det, [proposed_K - 1], axis=0
        )
        proposed_K -= 1

        assert proposed_K == len(proposed_log_sigma_det)
        assert numpy.max(proposed_label) < len(proposed_log_sigma_det)
        assert numpy.sum(proposed_count) == self._N
        assert proposed_mu.shape == (proposed_K, self._D)

        model_parameter = (
            proposed_label,
            proposed_K,
            proposed_count,
            proposed_mu,
            proposed_sum,
            proposed_log_sigma_det,
            proposed_sigma_inv,
        )
        model_parameter = self.update_cluster_parameters(
            cluster_label_1, model_parameter
        )

        return model_parameter

    def gibbs_sampling_merge(self, cluster_label, model_parameter):
        new_label = self.propose_cluster_to_merge(cluster_label, model_parameter)

        (
            proposed_label,
            proposed_K,
            proposed_count,
            proposed_mu,
            proposed_sum,
            proposed_log_sigma_det,
            proposed_sigma_inv,
        ) = model_parameter

        if new_label != cluster_label:
            # always merge the later cluster to the earlier cluster
            # this is to avoid errors if new_label is the last cluster
            if new_label > cluster_label:
                temp_label = cluster_label
                cluster_label = new_label
                new_label = temp_label

            proposed_label[numpy.nonzero(proposed_label == cluster_label)[0]] = (
                new_label
            )
            proposed_count[new_label] += proposed_count[cluster_label]
            proposed_sum[new_label, :] += proposed_sum[cluster_label, :]

            # since one cluster is empty now, switch it with the last one
            proposed_count[cluster_label] = proposed_count[proposed_K - 1]
            proposed_label[numpy.nonzero(proposed_label == (proposed_K - 1))] = (
                cluster_label
            )
            proposed_sum[cluster_label, :] = proposed_sum[proposed_K - 1, :]
            proposed_mu[cluster_label, :] = proposed_mu[proposed_K - 1, :]
            proposed_sigma_inv[cluster_label, :, :] = proposed_sigma_inv[
                proposed_K - 1, :, :
            ]
            proposed_log_sigma_det[cluster_label] = proposed_log_sigma_det[
                proposed_K - 1
            ]

            # remove the very last empty cluster, to remain compact cluster
            proposed_count = numpy.delete(proposed_count, [proposed_K - 1], axis=0)
            proposed_sum = numpy.delete(proposed_sum, [proposed_K - 1], axis=0)
            proposed_mu = numpy.delete(proposed_mu, [proposed_K - 1], axis=0)
            proposed_sigma_inv = numpy.delete(
                proposed_sigma_inv, [proposed_K - 1], axis=0
            )
            proposed_log_sigma_det = numpy.delete(
                proposed_log_sigma_det, [proposed_K - 1], axis=0
            )
            proposed_K -= 1

            assert new_label < proposed_K
            assert proposed_K == len(proposed_log_sigma_det)
            assert numpy.max(proposed_label) < len(proposed_log_sigma_det)
            assert numpy.sum(proposed_count) == self._N
            assert proposed_mu.shape == (proposed_K, self._D)

            # if these points are merged to another cluster, adjust its covariance matrix
            model_parameter = (
                proposed_label,
                proposed_K,
                proposed_count,
                proposed_mu,
                proposed_sum,
                proposed_log_sigma_det,
                proposed_sigma_inv,
            )
            model_parameter = self.update_cluster_parameters(new_label, model_parameter)

            return model_parameter
        else:
            return None

    """
	"""

    def log_posterior(self, model_parameter=None, hyper_parameter=None):
        if model_parameter == None:
            label = self._label
            K = self._K
            count = self._count
            mu = self._mu
            sum = self._sum
            log_sigma_det = self._log_sigma_det
            sigma_inv = self._sigma_inv
        else:
            label, K, count, mu, sum, log_sigma_det, sigma_inv = model_parameter

        if hyper_parameter == None:
            alpha_alpha = self._alpha_alpha
        else:
            alpha_alpha = hyper_parameter

        # log likelihood probability, accumulated per cluster (vectorized) rather than per point
        log_likelihood = -0.5 * self._N * self._D * numpy.log(2.0 * numpy.pi)
        for k in range(K):
            point_indices = numpy.nonzero(label == k)[0]
            if len(point_indices) == 0:
                continue
            mean_offset = self._X[point_indices, :] - mu[k, :]
            quadratic_forms = numpy.sum(
                (mean_offset @ sigma_inv[k, :, :]) * mean_offset, axis=1
            )
            log_likelihood -= 0.5 * len(point_indices) * log_sigma_det[
                k
            ] + 0.5 * numpy.sum(quadratic_forms)

        # log prior probability
        log_prior = K * numpy.log(alpha_alpha)
        log_prior += numpy.sum(scipy.special.gammaln(count))
        log_prior -= scipy.special.gammaln(self._N + alpha_alpha)
        log_prior += scipy.special.gammaln(alpha_alpha)

        return log_likelihood + log_prior

    """
	"""

    def log_likelihood(self, model_parameter=None, hyper_parameter=None):
        if model_parameter == None:
            label = self._label
            K = self._K
            count = self._count
            mu = self._mu
            sum = self._sum
            log_sigma_det = self._log_sigma_det
            sigma_inv = self._sigma_inv
        else:
            label, K, count, mu, sum, log_sigma_det, sigma_inv = model_parameter

        if hyper_parameter == None:
            alpha_alpha = self._alpha_alpha
        else:
            alpha_alpha = hyper_parameter

        # log likelihood probability, accumulated per cluster (vectorized) rather than per point
        log_likelihood = -0.5 * self._N * self._D * numpy.log(2.0 * numpy.pi)
        for k in range(K):
            point_indices = numpy.nonzero(label == k)[0]
            if len(point_indices) == 0:
                continue
            mean_offset = self._X[point_indices, :] - mu[k, :]
            quadratic_forms = numpy.sum(
                (mean_offset @ sigma_inv[k, :, :]) * mean_offset, axis=1
            )
            log_likelihood -= 0.5 * len(point_indices) * log_sigma_det[
                k
            ] + 0.5 * numpy.sum(quadratic_forms)

        return log_likelihood

    """
	"""

    def export_snapshot(self, output_directory):
        label_path = os.path.join(
            output_directory, "label-%d" % (self._iteration_counter)
        )
        numpy.savetxt(label_path, self._label, fmt="%d")

        mu_path = os.path.join(output_directory, "mu-%d" % (self._iteration_counter))
        numpy.savetxt(mu_path, self._mu)

        sigma_path = os.path.join(
            output_directory, "sigma-%d" % (self._iteration_counter)
        )
        sigma_matrices = numpy.zeros(self._sigma_inv.shape)
        for k in range(self._K):
            sigma_matrices[k, :, :] = numpy.linalg.pinv(self._sigma_inv[k, :, :])
        numpy.savetxt(
            sigma_path, numpy.reshape(sigma_matrices, (self._K * self._D, self._D))
        )

        """
		sigma_inv_path = os.path.join(output_directory, "sigma_inv-%d" % (self._iteration_counter))
		sigma_inv_matrices = numpy.zeros((self._K*self._D, self._D))
		for k in range(self._K):
			sigma_inv_matrices[k*self._D:(k+1)*self._D, :] = self._sigma_inv[k, :, :]
		numpy.savetxt(sigma_inv_path, sigma_inv_matrices)
		"""

    def propose_cluster_to_merge(self, cluster_label, model_parameter=None):
        if model_parameter == None:
            proposed_label = self._label
            proposed_K = self._K
            proposed_count = self._count
            proposed_mu = self._mu
            proposed_sum = self._sum
            proposed_log_sigma_det = self._log_sigma_det
            proposed_sigma_inv = self._sigma_inv
        else:
            (
                proposed_label,
                proposed_K,
                proposed_count,
                proposed_mu,
                proposed_sum,
                proposed_log_sigma_det,
                proposed_sigma_inv,
            ) = model_parameter

        # if this cluster is empty, no need to resample the cluster assignment
        assert proposed_count[cluster_label] > 0

        # find the index of the data point in the current cluster
        data_point_indices = numpy.nonzero(proposed_label == cluster_label)[0]

        # compute the prior of being in any of the clusters
        cluster_prior = numpy.copy(proposed_count)
        cluster_prior[cluster_label] = self._alpha_alpha

        cluster_log_prior = scipy.special.gammaln(
            cluster_prior + proposed_count[cluster_label]
        )
        cluster_log_prior -= scipy.special.gammaln(cluster_prior)

        # adjust for current cluster label
        cluster_log_prior[cluster_label] = numpy.log(
            self._alpha_alpha
        ) + scipy.special.gammaln(proposed_count[cluster_label])

        # cluster_log_prior += scipy.special.gammaln(self._N - proposed_count[cluster_label] + self._alpha_alpha)
        # cluster_log_prior -= scipy.special.gammaln(self._N + self._alpha_alpha)

        # initialize the likelihood vector for all clusters
        cluster_log_likelihood = numpy.zeros(proposed_K)

        # compute the likelihood for the existing clusters
        for k in range(proposed_K):
            if self._count[k] == 0:
                cluster_log_likelihood[k] = negative_infinity
                continue

            if k == cluster_label:
                # compute the likelihood for new cluster
                mean_offset = self._X[data_point_indices, :] - self._mu_0
                assert mean_offset.shape == (proposed_count[cluster_label], self._D)

                cluster_log_likelihood[cluster_label] = (
                    -0.5 * proposed_count[cluster_label] * self._log_sigma_det_0
                )
                cluster_log_likelihood[cluster_label] += -0.5 * numpy.sum(
                    numpy.dot(
                        numpy.dot(mean_offset, self._sigma_inv_0), mean_offset.T
                    ).item()
                )
            else:
                mean_offset = self._X[data_point_indices, :] - proposed_mu[[k], :]
                assert mean_offset.shape == (proposed_count[cluster_label], self._D)

                cluster_log_likelihood[k] = (
                    -0.5 * proposed_count[cluster_label] * proposed_log_sigma_det[k]
                )
                cluster_log_likelihood[k] += -0.5 * numpy.sum(
                    numpy.dot(
                        numpy.dot(mean_offset, proposed_sigma_inv[k, :, :]),
                        mean_offset.T,
                    ).item()
                )

        # normalize the posterior distribution
        cluster_log_posterior = cluster_log_prior + cluster_log_likelihood
        cluster_log_posterior -= scipy.special.logsumexp(cluster_log_posterior)
        cluster_posterior = numpy.exp(cluster_log_posterior)

        cdf = numpy.cumsum(cluster_posterior)
        new_label = numpy.uint(numpy.nonzero(cdf >= numpy.random.random())[0][0])
        assert new_label >= 0 and new_label < proposed_K

        return new_label

    def resample_component(self):
        if self._K == 1:
            return

        """
		probability_to_merge = 1.0 / self._count
		probability_to_merge /= numpy.sum(probability_to_merge)
		temp_label_probability = numpy.random.multinomial(1, probability_to_merge)[numpy.newaxis, :]
		cluster_label = numpy.nonzero(temp_label_probability == 1)[1][0]
		"""

        cluster_label = numpy.random.randint(0, self._K)

        # resample the cluster assignment only if this cluster is not empty
        if self._count[cluster_label] > 0:
            new_label = self.propose_cluster_to_merge(cluster_label)

            # find the index of the data point in the current cluster
            data_point_indices = numpy.nonzero(self._label == cluster_label)[0]

            if new_label != cluster_label:
                print(
                    "merge cluster %d and %d after component resampling..."
                    % (new_label, cluster_label)
                )

                self._label[data_point_indices] = new_label
                self._count[new_label] += self._count[cluster_label]
                self._sum[new_label, :] += self._sum[cluster_label, :]

                self.update_cluster_parameters(new_label)

                # clear the current cluster
                self._count[cluster_label] = 0
                self._sum[cluster_label, :] = 0
                self._mu[cluster_label, :] = 0
                self._sigma_inv[cluster_label, :, :] = 0
                self._log_sigma_det[cluster_label] = 0

        empty_cluster = numpy.nonzero(self._count == 0)[0]
        non_empty_cluster = numpy.nonzero(self._count > 0)[0]
        for cluster_label in empty_cluster:
            assert numpy.all(self._label != cluster_label)

        # shift down all the cluster indices
        for cluster_label in range(len(non_empty_cluster)):
            self._label[
                numpy.nonzero(self._label == non_empty_cluster[cluster_label])[0]
            ] = cluster_label

        self._K -= len(empty_cluster)

        self._count = numpy.delete(self._count, empty_cluster, axis=0)
        assert self._count.shape == (self._K,)
        self._sum = numpy.delete(self._sum, empty_cluster, axis=0)
        assert self._sum.shape == (self._K, self._D)
        self._mu = numpy.delete(self._mu, empty_cluster, axis=0)
        assert self._mu.shape == (self._K, self._D)
        self._sigma_inv = numpy.delete(self._sigma_inv, empty_cluster, axis=0)
        assert self._sigma_inv.shape == (self._K, self._D, self._D)
        self._log_sigma_det = numpy.delete(self._log_sigma_det, empty_cluster, axis=0)
        assert self._log_sigma_det.shape == (self._K,)

        return

    def resample_components(self):
        if self._K == 1:
            return

        # sample cluster assignment for all the points in the current cluster
        for cluster_label in numpy.argsort(self._count):
            # if this cluster is empty, no need to resample the cluster assignment
            if self._count[cluster_label] <= 0:
                continue

            new_label = self.propose_cluster_to_merge(cluster_label)

            # find the index of the data point in the current cluster
            data_point_indices = numpy.nonzero(self._label == cluster_label)[0]

            if new_label != cluster_label:
                print(
                    "merge cluster %d and %d after component resampling..."
                    % (new_label, cluster_label)
                )

                self._label[data_point_indices] = new_label
                self._count[new_label] += self._count[cluster_label]
                self._sum[new_label, :] += self._sum[cluster_label, :]

                self.update_cluster_parameters(new_label)

                # clear the current cluster
                self._count[cluster_label] = 0
                self._sum[cluster_label, :] = 0
                self._mu[cluster_label, :] = 0
                self._sigma_inv[cluster_label, :, :] = 0
                self._log_sigma_det[cluster_label] = 0

        empty_cluster = numpy.nonzero(self._count == 0)[0]
        non_empty_cluster = numpy.nonzero(self._count > 0)[0]
        for cluster_label in empty_cluster:
            assert numpy.all(self._label != cluster_label)

        # shift down all the cluster indices
        for cluster_label in range(len(non_empty_cluster)):
            self._label[
                numpy.nonzero(self._label == non_empty_cluster[cluster_label])[0]
            ] = cluster_label

        self._K -= len(empty_cluster)

        self._count = numpy.delete(self._count, empty_cluster, axis=0)
        assert self._count.shape == (self._K,)
        self._sum = numpy.delete(self._sum, empty_cluster, axis=0)
        assert self._sum.shape == (self._K, self._D)
        self._mu = numpy.delete(self._mu, empty_cluster, axis=0)
        assert self._mu.shape == (self._K, self._D)
        self._sigma_inv = numpy.delete(self._sigma_inv, empty_cluster, axis=0)
        assert self._sigma_inv.shape == (self._K, self._D, self._D)
        self._log_sigma_det = numpy.delete(self._log_sigma_det, empty_cluster, axis=0)
        assert self._log_sigma_det.shape == (self._K,)

        return

    def model_assertion(self, model_parameter=None):
        if model_parameter == None:
            label = self._label
            K = self._K
            count = self._count
            mu = self._mu
            sum = self._sum
            log_sigma_det = self._log_sigma_det
            sigma_inv = self._sigma_inv
        else:
            label, K, count, mu, sum, log_sigma_det, sigma_inv = model_parameter

        test_count = numpy.zeros(K)
        for point_index in numpy.random.permutation(range(self._N)):
            test_count[label[point_index]] += 1
        assert numpy.all(test_count == count)

        return


def fit_dpgm_mc(
    data,
    alpha_alpha=1.0,
    training_iterations=100,
    split_merge_heuristics=-1,
    split_proposal=0,
    merge_proposal=0,
    initial_clusters=1,
    save_best=True,
    verbose=True,
):
    """
    Fit a Dirichlet Process Gaussian Mixture Model to the data.

    Parameters:
    -----------
    data : numpy.ndarray
        N x M array where N is the number of points and M is the number of features
    alpha_alpha : float, optional (default=1.0)
        Concentration parameter for the Dirichlet process. Higher values encourage more clusters.
    training_iterations : int, optional (default=100)
        Number of training iterations for the Gibbs sampler
    split_merge_heuristics : int, optional (default=-1)
        Split-merge heuristics:
        -1: no split-merge operation
        0: component resampling
        1: random choose candidate clusters by points
        2: random choose candidate clusters by point-cluster
        3: random choose candidate clusters by clusters
    split_proposal : int, optional (default=0)
        Propose split operation via:
        0: metropolis-hastings
        1: restricted gibbs sampler and metropolis-hastings
        2: sequential allocation and metropolis-hastings
    merge_proposal : int, optional (default=0)
        Propose merge operation via:
        0: metropolis-hastings
        1: restricted gibbs sampler and metropolis-hastings
        2: gibbs sampler and metropolis-hastings
    initial_clusters : int, optional (default=1)
        How the sampler is seeded:
        1: start with all points in a single cluster and let the Gibbs sampler grow
            clusters. Recommended for large N -- avoids the expensive first pass below.
        0: start with one cluster per point (K = N). Faithful to the original code but
            makes the first iteration O(N^2 * D^2), impractical for large datasets.
        k > 1: randomly assign points into k clusters (a random restart from K = k).
    save_best : bool, optional (default=True)
        If True, return the sweep with the highest log-posterior rather than the last sweep,
        and leave the model in that state (so prediction uses the best sweep too). Gibbs is a
        sampler, so the final sweep is not necessarily the best. The chosen sweep index is
        reported in results['best_iteration'].
    verbose : bool, optional (default=True)
        Whether to print progress information

    Returns:
    --------
    dict : Dictionary containing:
        - 'labels': numpy array of cluster assignments (N,)
        - 'n_clusters': number of clusters found
        - 'cluster_means': cluster centers (K x M)
        - 'cluster_counts': number of points in each cluster (K,)
        - 'log_likelihood': log-posterior of the returned sweep
        - 'best_iteration': index of the sweep that was returned (None if save_best=False)
        - 'model': the trained MonteCarlo model object
    """

    # Validate input
    if not isinstance(data, numpy.ndarray):
        data = numpy.array(data)

    if len(data.shape) != 2:
        raise ValueError("Data must be a 2D array (N x M)")

    N, M = data.shape

    if verbose:
        print("=" * 60)
        print("Fitting Dirichlet Process Gaussian Mixture Model")
        print("=" * 60)
        print("Data shape: {} points with {} features".format(N, M))
        print("Concentration parameter (alpha_alpha): {}".format(alpha_alpha))
        print("Training iterations: {}".format(training_iterations))
        print("Split-merge heuristics: {}".format(split_merge_heuristics))
        print("=" * 60)

    # Initialize the model
    dpgm = MonteCarlo(
        split_merge_heuristics=split_merge_heuristics,
        split_proposal=split_proposal,
        merge_proposal=merge_proposal,
    )

    dpgm._initialize(data, alpha_alpha=alpha_alpha, initial_clusters=initial_clusters)

    # Run training iterations
    log_likelihood = None
    best_log_likelihood = None
    best_iteration = None
    best_state = None
    for iteration in range(training_iterations):
        log_likelihood_val = dpgm.learning()
        # Convert to scalar if it's an array
        if isinstance(log_likelihood_val, numpy.ndarray):
            log_likelihood = (
                float(log_likelihood_val.item())
                if log_likelihood_val.size == 1
                else float(log_likelihood_val[0])
            )
        else:
            log_likelihood = float(log_likelihood_val)

        # snapshot the model whenever this sweep improves on the best log-posterior so far;
        # Gibbs is a sampler, so the last sweep is not necessarily the best one
        if save_best and (
            best_log_likelihood is None or log_likelihood > best_log_likelihood
        ):
            best_log_likelihood = log_likelihood
            best_iteration = iteration + 1
            best_state = (
                dpgm._label.copy(),
                dpgm._K,
                dpgm._count.copy(),
                dpgm._mu.copy(),
                dpgm._sum.copy(),
                dpgm._log_sigma_det.copy(),
                dpgm._sigma_inv.copy(),
            )

        if verbose and (iteration + 1) % max(1, training_iterations // 10) == 0:
            print(
                "Iteration {}/{}: {} clusters, log-likelihood = {:.4f}".format(
                    iteration + 1, training_iterations, dpgm._K, log_likelihood
                )
            )

    # restore the best sweep so the returned labels / model reflect it (used for prediction too)
    if save_best and best_state is not None:
        (
            dpgm._label,
            dpgm._K,
            dpgm._count,
            dpgm._mu,
            dpgm._sum,
            dpgm._log_sigma_det,
            dpgm._sigma_inv,
        ) = best_state
        log_likelihood = best_log_likelihood

    if verbose:
        print("=" * 60)
        print("Training completed!")
        if save_best and best_iteration is not None:
            print("Best sweep: {} (of {})".format(best_iteration, training_iterations))
        print("Final number of clusters: {}".format(dpgm._K))
        print("Final log-likelihood: {:.4f}".format(log_likelihood))
        print("=" * 60)

    # Extract results
    results = {
        "labels": dpgm._label.copy(),
        "n_clusters": dpgm._K,
        "cluster_means": dpgm._mu.copy(),
        "cluster_counts": dpgm._count.copy(),
        "log_likelihood": log_likelihood,
        "best_iteration": best_iteration,
        "model": dpgm,
    }

    return results


def predict_dpgm(model, data):
    """
    Predict cluster assignments for new data points using a trained DPGM model.

    Parameters:
    -----------
    model : MonteCarlo or dict
        Either a trained MonteCarlo model object or the results dictionary from fit_dpgm
    data : numpy.ndarray
        N x M array where N is the number of points and M is the number of features

    Returns:
    --------
    numpy.ndarray : Array of cluster assignments (N,)
    """

    # Extract model if results dict was passed
    if isinstance(model, dict):
        if "model" not in model:
            raise ValueError("Dictionary must contain 'model' key")
        model = model["model"]

    # Validate input
    if not isinstance(data, numpy.ndarray):
        data = numpy.array(data)

    if len(data.shape) != 2:
        raise ValueError("Data must be a 2D array (N x M)")

    # Use the inference method to predict labels
    # The inference method returns (labels, log_likelihood)
    predictions, log_likelihood = model.inference(data)

    return predictions


def _log_wishart_normalizer(logdet_W, nu, D):
    """log B(W, nu) from Bishop (B.79), given log|W| and degrees of freedom nu."""
    i = numpy.arange(1, D + 1)
    return (
        -(nu / 2.0) * logdet_W
        - (nu * D / 2.0) * numpy.log(2.0)
        - (D * (D - 1) / 4.0) * numpy.log(numpy.pi)
        - numpy.sum(scipy.special.gammaln((nu + 1 - i) / 2.0))
    )


class VariationalInference(object):
    """
    Mean-field variational inference (CAVI) for a Dirichlet Process Gaussian Mixture Model.

    Companion to MonteCarlo (collapsed Gibbs sampler). Where the sampler draws samples and
    mixes slowly, this performs deterministic coordinate-ascent variational inference and
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

    Reference: Blei & Jordan (2006), "Variational inference for Dirichlet process mixtures";
    Bishop (2006), PRML chapter 10.2.

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

    def __init__(
        self,
        truncation=50,
        alpha=1.0,
        max_iter=200,
        tol=1e-4,
        beta_0=1.0,
        nu_0=None,
        m_0=None,
        W_0=None,
        random_state=0,
        verbose=True,
    ):
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
        self._N, self._D = X.shape
        T, D = self.truncation, self._D

        # ---- prior hyperparameters (data-driven defaults) ----
        self._m0 = (
            numpy.mean(X, axis=0)
            if self._m_0 is None
            else numpy.asarray(self._m_0, float)
        )
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
        # squared distances via ||x-m||^2 = ||x||^2 - 2 x.m + ||m||^2, avoiding an (N, T, D) temporary
        sq_dist = (
            numpy.sum(X**2, axis=1)[:, None]
            - 2.0 * (X @ seed_means.T)
            + numpy.sum(seed_means**2, axis=1)[None, :]
        )
        log_resp = -0.5 * sq_dist
        log_resp -= scipy.special.logsumexp(log_resp, axis=1, keepdims=True)
        self._resp = numpy.exp(log_resp)

    # -------------------------------------------------------------- CAVI steps

    def _compute_statistics(self, resp):
        """Weighted count, mean, and scatter per component (Bishop 10.51-10.53)."""
        T, D = self.truncation, self._D
        Nk = resp.sum(axis=0) + 1e-10  # (T,)
        xbar = (resp.T @ self._X) / Nk[:, None]  # (T, D)
        S = numpy.zeros((T, D, D))
        for t in range(T):
            diff = self._X - xbar[t]  # (N, D)
            S[t] = (resp[:, t][:, None] * diff).T @ diff / Nk[t]
        return Nk, xbar, S

    def _update_niw(self, Nk, xbar, S):
        """Posterior NIW parameters per component (Bishop 10.60-10.63)."""
        self._beta = self._beta0 + Nk
        self._nu = self._nu0 + Nk
        self._m = (self._beta0 * self._m0[None, :] + Nk[:, None] * xbar) / self._beta[
            :, None
        ]
        self._W = numpy.zeros((self.truncation, self._D, self._D))
        for t in range(self.truncation):
            diff = (xbar[t] - self._m0)[:, None]
            W_inv = (
                self._W0_inv
                + Nk[t] * S[t]
                + (self._beta0 * Nk[t] / self._beta[t]) * (diff @ diff.T)
            )
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

        with numpy.errstate(over="ignore", divide="ignore", invalid="ignore"):
            logdet_W = numpy.linalg.slogdet(self._W)[1]  # (T,)
        E_logdet_Lambda = (
            scipy.special.digamma((self._nu[:, None] + 1 - i[None, :]) / 2.0).sum(
                axis=1
            )
            + D * numpy.log(2.0)
            + logdet_W
        )  # (T,)

        digamma_sum = scipy.special.digamma(self._gamma1 + self._gamma2)
        E_log_v = scipy.special.digamma(self._gamma1) - digamma_sum  # (T,)
        E_log_1mv = scipy.special.digamma(self._gamma2) - digamma_sum  # (T,)
        # E[ln pi_t] = E[ln v_t] + sum_{s<t} E[ln(1 - v_s)]
        cumulative = numpy.concatenate(([0.0], numpy.cumsum(E_log_1mv)[:-1]))
        E_log_pi = E_log_v + cumulative  # (T,)

        return E_logdet_Lambda, logdet_W, E_log_pi, E_log_v, E_log_1mv

    def _e_step(self, E_logdet_Lambda, E_log_pi):
        """Responsibilities from expected log-likelihood + expected log-weights (Bishop 10.46, 10.67)."""
        T, D = self.truncation, self._D
        # expected quadratic form E[(x-mu)^T Lambda (x-mu)] = D/beta_t + nu_t (x-m_t)^T W_t (x-m_t)
        log_rho = numpy.zeros((self._N, T))
        for t in range(T):
            diff = self._X - self._m[t]  # (N, D)
            maha = numpy.sum((diff @ self._W[t]) * diff, axis=1)
            E_quad = D / self._beta[t] + self._nu[t] * maha
            log_rho[:, t] = (
                E_log_pi[t]
                + 0.5 * E_logdet_Lambda[t]
                - 0.5 * D * numpy.log(2.0 * numpy.pi)
                - 0.5 * E_quad
            )
        log_resp = log_rho - scipy.special.logsumexp(log_rho, axis=1, keepdims=True)
        return log_resp, numpy.exp(log_resp)

    # ------------------------------------------------------------------- ELBO

    def _elbo(
        self,
        Nk,
        xbar,
        S,
        resp,
        log_resp,
        E_logdet_Lambda,
        logdet_W,
        E_log_pi,
        E_log_v,
        E_log_1mv,
    ):
        T, D = self.truncation, self._D
        ln2pi = numpy.log(2.0 * numpy.pi)

        # E[ln p(X | Z, mu, Lambda)]  (Bishop 10.71)
        term_x = 0.0
        for t in range(T):
            diff = (xbar[t] - self._m[t])[:, None]
            quad = (self._nu[t] * (diff.T @ self._W[t] @ diff)).item()
            trace_SW = self._nu[t] * numpy.trace(S[t] @ self._W[t])
            term_x += (
                0.5
                * Nk[t]
                * (E_logdet_Lambda[t] - D / self._beta[t] - trace_SW - quad - D * ln2pi)
            )

        # E[ln p(Z | pi)]  (Bishop 10.72)
        term_z = numpy.sum(Nk * E_log_pi)

        # E[ln p(v)] with Beta(1, alpha) stick prior
        term_v = numpy.sum(numpy.log(self.alpha) + (self.alpha - 1.0) * E_log_1mv)

        # E[ln p(mu, Lambda)]  (Bishop 10.74)
        term_ml = 0.0
        with numpy.errstate(over="ignore", divide="ignore", invalid="ignore"):
            logdet_W0 = numpy.linalg.slogdet(self._W0)[1]
        logB0 = _log_wishart_normalizer(logdet_W0, self._nu0, D)
        for t in range(T):
            diff = (self._m[t] - self._m0)[:, None]
            quad = (self._beta0 * self._nu[t] * (diff.T @ self._W[t] @ diff)).item()
            term_ml += 0.5 * (
                D * numpy.log(self._beta0 / (2.0 * numpy.pi))
                + E_logdet_Lambda[t]
                - D * self._beta0 / self._beta[t]
                - quad
            )
            term_ml += 0.5 * (self._nu0 - D - 1.0) * E_logdet_Lambda[t]
            term_ml -= 0.5 * self._nu[t] * numpy.trace(self._W0_inv @ self._W[t])
        term_ml += T * logB0

        # -E[ln q(Z)]  (entropy of responsibilities)
        term_qz = -numpy.sum(resp * log_resp)

        # -E[ln q(v)]  (entropy of the Beta factors)
        ln_beta_fn = (
            scipy.special.gammaln(self._gamma1)
            + scipy.special.gammaln(self._gamma2)
            - scipy.special.gammaln(self._gamma1 + self._gamma2)
        )
        term_qv = -numpy.sum(
            (self._gamma1 - 1.0) * E_log_v
            + (self._gamma2 - 1.0) * E_log_1mv
            - ln_beta_fn
        )

        # -E[ln q(mu, Lambda)]  (Bishop 10.77 with Wishart entropy B.82)
        term_qml = 0.0
        for t in range(T):
            logB_t = _log_wishart_normalizer(logdet_W[t], self._nu[t], D)
            H_lambda = (
                -logB_t
                - 0.5 * (self._nu[t] - D - 1.0) * E_logdet_Lambda[t]
                + 0.5 * self._nu[t] * D
            )
            e_ln_q = (
                0.5 * E_logdet_Lambda[t]
                + 0.5 * D * numpy.log(self._beta[t] / (2.0 * numpy.pi))
                - 0.5 * D
                - H_lambda
            )
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
            E_logdet_Lambda, logdet_W, E_log_pi, E_log_v, E_log_1mv = (
                self._expectations()
            )
            elbo = self._elbo(
                Nk,
                xbar,
                S,
                self._resp,
                numpy.log(self._resp + 1e-300),
                E_logdet_Lambda,
                logdet_W,
                E_log_pi,
                E_log_v,
                E_log_1mv,
            )
            self.elbo_trajectory_.append(elbo)

            _, self._resp = self._e_step(E_logdet_Lambda, E_log_pi)

            if self.verbose and (iteration + 1) % max(1, self.max_iter // 20) == 0:
                print(
                    "Iteration {}/{}: ELBO = {:.2f}, effective clusters = {}".format(
                        iteration + 1,
                        self.max_iter,
                        elbo,
                        self.n_effective_clusters(Nk),
                    )
                )

            if previous_elbo is not None:
                rel_change = abs(elbo - previous_elbo) / (abs(previous_elbo) + 1e-10)
                if rel_change < self.tol:
                    if self.verbose:
                        print(
                            "Converged at iteration {} (relative ELBO change {:.2e})".format(
                                iteration + 1, rel_change
                            )
                        )
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


def fit_dpgm_vi(
    data, alpha=1.0, truncation=50, max_iter=200, tol=1e-4, random_state=0, verbose=True
):
    """
    Fit a DP Gaussian mixture by variational inference. Returns a dict mirroring
    fit_dpgm so it drops into the same tooling.

    Cluster labels are remapped to a contiguous 0..K-1 over the effective (non-empty) components.
    """
    data = numpy.asarray(data, float)
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array (N x M)")

    model = VariationalInference(
        truncation=truncation,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        verbose=verbose,
    )
    model.fit(data)

    raw_labels = numpy.argmax(model._resp, axis=1)
    used = numpy.unique(raw_labels)
    remap = {old: new for new, old in enumerate(used)}
    labels = numpy.array([remap[l] for l in raw_labels], dtype=int)

    return {
        "labels": labels,
        "n_clusters": len(used),
        "cluster_means": model._m[used].copy(),
        "cluster_counts": numpy.bincount(labels, minlength=len(used)),
        "log_likelihood": model.converged_elbo_,  # ELBO of the converged fit
        "model": model,
    }


def predict_dpgm_vi(model, data):
    """Predict contiguous-relabeled cluster assignments for new data."""
    if isinstance(model, dict):
        model = model["model"]
    return model.predict(numpy.asarray(data, float))


if __name__ == "__main__":
    """
    Example usage of the DPGM API
    """
    import sys

    # Example 1: Generate synthetic data and fit DPGM
    print("\n" + "=" * 60)
    print("Example 1: Synthetic data clustering")
    print("=" * 60)

    # Generate synthetic data with 3 clusters
    numpy.random.seed(42)
    n_points_per_cluster = 100

    # Cluster 1: centered at (0, 0)
    cluster1 = numpy.random.randn(n_points_per_cluster, 2) * 0.5

    # Cluster 2: centered at (5, 5)
    cluster2 = numpy.random.randn(n_points_per_cluster, 2) * 0.5 + 5

    # Cluster 3: centered at (5, 0)
    cluster3 = numpy.random.randn(n_points_per_cluster, 2) * 0.5 + numpy.array([5, 0])

    # Combine all data
    synthetic_data = numpy.vstack([cluster1, cluster2, cluster3])

    # Shuffle the data
    indices = numpy.random.permutation(len(synthetic_data))
    synthetic_data = synthetic_data[indices]

    print("\nFitting DPGM on {} synthetic data points...".format(len(synthetic_data)))

    # Fit the model
    results = fit_dpgm_mc(
        synthetic_data, alpha_alpha=1.0, training_iterations=50, verbose=True
    )

    print("\nClustering results:")
    print("  Number of clusters found: {}".format(results["n_clusters"]))
    print("  Cluster counts: {}".format(results["cluster_counts"]))
    print("  Cluster means:\n{}".format(results["cluster_means"]))

    # Example 2: Load data from file if provided
    if len(sys.argv) > 1:
        print("\n" + "=" * 60)
        print("Example 2: Clustering data from file")
        print("=" * 60)

        data_file = sys.argv[1]
        print("\nLoading data from: {}".format(data_file))

        try:
            file_data = numpy.loadtxt(data_file)
            print("Data shape: {}".format(file_data.shape))

            # Fit the model
            file_results = fit_dpgm_mc(
                file_data, alpha_alpha=1.0, training_iterations=100, verbose=True
            )

            print("\nClustering results:")
            print("  Number of clusters found: {}".format(file_results["n_clusters"]))
            print("  Cluster counts: {}".format(file_results["cluster_counts"]))

            # Save results
            output_labels = data_file.replace(".dat", "_labels.dat")
            numpy.savetxt(output_labels, file_results["labels"], fmt="%d")
            print("\nCluster labels saved to: {}".format(output_labels))

        except Exception as e:
            print("Error loading or processing file: {}".format(e))
            sys.exit(1)

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
