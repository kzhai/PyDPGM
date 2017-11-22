"""
@author: Ke Zhai (zhaike@cs.umd.edu)

Implements collapsed Gibbs sampling for the Dirichlet process Gaussian mixture model (DPGM).
"""

import os
import random
import sys

import numpy
import scipy
import scipy.misc

negative_infinity = -1e500
# We will be taking log(0) = -Inf, so turn off this warning
numpy.seterr(divide='ignore')


# numpy.random.seed(100)

class MonteCarlo(object):
	"""
	@param truncation_level: the maximum number of clusters, used for speeding up the computation
	@param snapshot_interval: the interval for exporting a snapshot of the model
	"""

	def __init__(self,
	             split_merge_heuristics=-1,
	             split_proposal=0,
	             merge_proposal=0,
	             split_merge_iteration=1,
	             component_resampling_interval=100,
	             restrict_gibbs_sampling_iteration=10,
	             # gamma_shape_alpha=1,
	             # gamma_scale_beta=1,
	             hyper_parameter_interval=-1
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

	def _initialize(self,
	                data,
	                alpha_alpha=1.,
	                # alpha_kappa=1.,
	                # alpha_nu=1.,
	                alpha_mu=None,
	                alpha_sigma=None
	                ):
		self._X = data
		(self._N, self._D) = self._X.shape

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

		self._log_sigma_det_0 = numpy.log(numpy.linalg.det(self._sigma_0))
		self._sigma_inv_0 = numpy.linalg.pinv(self._sigma_0)

		# initialize the concentration parameter of the dirichlet distirbution
		self._alpha_alpha = alpha_alpha

		'''
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
		'''

		self.random_initialization()

		self._iteration_counter = 0

	def random_initialization(self, number_of_clusters=0):
		assert number_of_clusters <= self._N

		if number_of_clusters == 0:
			# initialize every point to one cluster per point
			self._label = numpy.arange(self._N)
		else:
			# initialize all points to one cluster
			self._label = numpy.zeros(self._N, dtype='int64')

		self._K = len(numpy.unique(self._label))
		self._count = numpy.bincount(self._label)
		assert numpy.all(
			self._count > 0), "initialization contains empty cluster, maybe try to reduce the number of clusters during initialization"
		assert numpy.sum(self._count) == self._N

		# initialize the sigma, inv(sigma) and log(det(sigma)) of all cluster up to truncation level
		self._sigma_inv = numpy.zeros((self._K, self._D, self._D))
		self._log_sigma_det = numpy.zeros(self._K)
		self._mu = numpy.zeros((self._K, self._D))

		# compute the sum and square sum of all cluster up to truncation level
		self._sum = numpy.zeros((self._K, self._D))
		for cluster_index in xrange(self._K):
			point_indices = numpy.nonzero(self._label == (cluster_index))[0]
			self._sum[cluster_index, :] = numpy.sum(self._X[point_indices, :], 0)

			# update the cluster parameters
			self.update_cluster_parameters(cluster_index)

	def learning(self):
		self._iteration_counter += 1

		self.sample_cgs()

		assert numpy.all(self._count > 0)

		if self._hyper_parameter_interval > 0 and self._iteration_counter % self._hyper_parameter_interval == 0:
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

		print "accumulated number of points for each cluster:", "[", " ".join("%d" % x for x in self._count), "]"
		# print "accumulated number of tokens:", numpy.sum(self._n_kv, axis=1)[:, numpy.newaxis].T

		return self.log_posterior()

	"""
	sample the data to train the parameters
	"""

	def sample_cgs(self):
		# sample the total data
		for point_index in numpy.random.permutation(xrange(self._N)):
			assert self._count.shape == (self._K,)
			assert self._sum.shape == (self._K, self._D)
			assert self._mu.shape == (self._K, self._D)
			assert self._sigma_inv.shape == (self._K, self._D, self._D)
			assert self._log_sigma_det.shape == (self._K,)

			# get the old label of current point
			old_label = self._label[point_index]
			assert old_label < self._K and old_label >= 0, "%d\t%d" % (old_label, self._K)

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
				self._log_sigma_det = numpy.delete(self._log_sigma_det, [self._K - 1], axis=0)

				self._K -= 1
				old_label = -1
			else:
				self.update_cluster_parameters(old_label)

			# compute the prior of being in any of the clusters
			cluster_prior = numpy.hstack((self._count[:self._K], self._alpha_alpha))
			cluster_prior = cluster_prior / (self._N - 1. + self._alpha_alpha)
			cluster_log_prior = numpy.log(cluster_prior)

			# initialize the likelihood vector for all clusters
			cluster_log_likelihood = numpy.zeros(self._K + 1)

			# compute the likelihood for new cluster
			mean_offset = self._X[[point_index], :] - self._mu_0
			assert mean_offset.shape == (1, self._D)

			cluster_log_likelihood[self._K] = -0.5 * self._log_sigma_det_0
			cluster_log_likelihood[self._K] += -0.5 * numpy.dot(numpy.dot(mean_offset, self._sigma_inv_0),
			                                                    mean_offset.T)

			# compute the likelihood for the existing clusters
			for k in xrange(self._K):
				mean_offset = self._X[[point_index], :] - self._mu[[k], :]
				assert mean_offset.shape == (1, self._D)

				cluster_log_likelihood[k] = -0.5 * self._log_sigma_det[k]
				cluster_log_likelihood[k] += -0.5 * numpy.dot(numpy.dot(mean_offset, self._sigma_inv[k, :, :]),
				                                              mean_offset.T)

			# normalize the posterior distribution
			cluster_log_posterior = cluster_log_prior + cluster_log_likelihood
			cluster_log_posterior -= scipy.misc.logsumexp(cluster_log_posterior)
			cluster_posterior = numpy.exp(cluster_log_posterior)

			# sample a new cluster label for current point 
			temp_label_probability = numpy.random.multinomial(1, cluster_posterior)[numpy.newaxis, :]
			new_label = numpy.nonzero(temp_label_probability == 1)[1][0]
			# cdf = numpy.cumsum(cluster_posterior)
			# new_label = numpy.uint(numpy.nonzero(cdf >= numpy.random.random())[0][0])
			assert (new_label >= 0 and new_label <= self._K), \
				(cluster_posterior, temp_label_probability, new_label, new_label >= 0, new_label <= self._K)

			# if this point starts up a new cluster
			if new_label == self._K:
				self._K += 1
				self._count = numpy.hstack((self._count, numpy.zeros(1)))

				self._sum = numpy.vstack((self._sum, numpy.zeros((1, self._D))))
				self._mu = numpy.vstack((self._mu, numpy.zeros((1, self._D))))
				self._sigma_inv = numpy.vstack((self._sigma_inv, numpy.zeros((1, self._D, self._D))))
				self._log_sigma_det = numpy.hstack((self._log_sigma_det, numpy.zeros(1)))

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
		(N, D) = X_prime.shape
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
		cluster_log_prior -= scipy.misc.logsumexp(cluster_log_prior)

		# sample the entire dataset
		for point_index in numpy.random.permutation(xrange(N)):
			# initialize the likelihood vector for all clusters
			cluster_log_likelihood = numpy.zeros(self._K)

			# compute the likelihood for the existing clusters
			for k in xrange(self._K):
				mean_offset = X_prime[[point_index], :] - self._mu[[k], :]
				assert mean_offset.shape == (1, self._D)

				cluster_log_likelihood[k] = -0.5 * self._log_sigma_det[k]
				cluster_log_likelihood[k] += -0.5 * numpy.dot(numpy.dot(mean_offset, self._sigma_inv[k, :, :]),
				                                              mean_offset.T)

			# normalize the posterior distribution
			cluster_log_posterior = cluster_log_prior + cluster_log_likelihood
			cluster_log_posterior -= scipy.misc.logsumexp(cluster_log_posterior)
			cluster_posterior = numpy.exp(cluster_log_posterior)

			# sample a new cluster label for current point
			temp_label_probability = numpy.random.multinomial(1, cluster_posterior)[numpy.newaxis, :]
			new_label = numpy.nonzero(temp_label_probability == 1)[1][0]
			assert (new_label >= 0 and new_label < self._K), \
				(cluster_posterior, temp_label_probability, new_label, new_label >= 0, new_label < self._K)

			label_prime[point_index] = new_label
			log_likelihood_prime += cluster_log_prior[new_label] + cluster_log_likelihood[new_label]

		assert numpy.all(label_prime >= 0)
		assert numpy.all(label_prime < self._K)

		return label_prime, log_likelihood_prime

	"""
	"""

	def optimize_hyperparameters(self, hyperparameter_samples=10, hyperparameter_step_size=1.0,
	                             hyperparameter_maximum_iteration=10):
		old_log_alpha_alpha = numpy.log(self._alpha_alpha)

		for ii in xrange(hyperparameter_samples):
			log_likelihood_old = self.log_posterior()
			log_likelihood_new = numpy.log(numpy.random.random()) + log_likelihood_old
			# print("OLD: %f\tNEW: %f at (%f, %f)" % (log_likelihood_old, log_likelihood_new, self._alpha, self._beta))

			l = old_log_alpha_alpha - numpy.random.random() * hyperparameter_step_size
			r = old_log_alpha_alpha + hyperparameter_step_size

			for jj in xrange(hyperparameter_maximum_iteration):
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

			print "update hyperparameter to %f" % (numpy.exp(new_log_alpha_alpha))

	def split_merge(self):
		for iteration in xrange(self._split_merge_iteration):
			label_probability = 1.0 * self._count / numpy.sum(self._count)

			if self._split_merge_heuristics == 1:
				temp_label_probability = numpy.random.multinomial(1, label_probability)[numpy.newaxis, :]
				random_label_1 = numpy.nonzero(temp_label_probability == 1)[1][0]
				temp_label_probability = numpy.random.multinomial(1, label_probability)[numpy.newaxis, :]
				random_label_2 = numpy.nonzero(temp_label_probability == 1)[1][0]
			elif self._split_merge_heuristics == 2:
				random_label_1 = numpy.random.randint(self._K)
				temp_label_probability = numpy.random.multinomial(1, label_probability)[numpy.newaxis, :]
				random_label_2 = numpy.nonzero(temp_label_probability == 1)[1][0]
			elif self._split_merge_heuristics == 3:
				random_label_1 = numpy.random.randint(self._K)
				random_label_2 = numpy.random.randint(self._K)
			else:
				sys.stderr.write("error: unrecognized split-merge heuristics %d...\n" % (self._split_merge_heuristics))
				return

			if random_label_1 == random_label_2:
				self.split_metropolis_hastings(random_label_1)
				assert numpy.all(self._count > 0)
			else:
				self.merge_metropolis_hastings(random_label_1, random_label_2)
				assert numpy.all(self._count > 0)

	'''
	def split_mh_merge_gs(self):
		if self._split_merge_heuristics==0:
			return
		
		for iteration in xrange(self._split_merge_iteration):
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
	'''

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
			(label, K, count, mu, sum, log_sigma_det, sigma_inv) = model_parameter

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
				points_in_cluster.shape, count[cluster_id])
			temp_sigma = numpy.cov(points_in_cluster.T)
			temp_sigma_inv = numpy.linalg.pinv(temp_sigma)

		# compute n*\Sigma^{-1}
		temp_a = count[cluster_id] * temp_sigma_inv
		# compute \Sigma_{0}^{-1} + n*\Sigma^{-1}
		temp_b = self._sigma_inv_0 + temp_a
		# compute (\Sigma_{0}^{-1} + n*\Sigma^{-1})^{-1}
		temp_c = numpy.linalg.pinv(temp_b)
		sigma_hat = self._sigma_0 + temp_c
		assert sigma_hat.shape == (self._D, self._D)

		sigma_inv[cluster_id, :, :] = numpy.linalg.pinv(sigma_hat)
		log_sigma_det[cluster_id] = numpy.log(numpy.linalg.det(sigma_hat))

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
			proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
			proposed_sigma_inv)

		if self._split_proposal == 0:
			# perform random split for split proposal
			model_parameter = self.random_split(cluster_label, model_parameter)

			if model_parameter == None:
				return

			(proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
			 proposed_sigma_inv) = model_parameter
			assert numpy.all(proposed_count > 0)

			log_proposal_probability = (proposed_count[cluster_label] + proposed_count[proposed_K - 1] - 2) * numpy.log(
				2)
		elif self._split_proposal == 1:
			# perform restricted gibbs sampling for split proposal
			model_parameter = self.random_split(cluster_label, model_parameter)
			# split a singleton cluster
			if model_parameter == None:
				return
			(proposed_label, proposed_K, old_proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
			 proposed_sigma_inv) = model_parameter

			# self.model_assertion(model_parameter)
			(model_parameter, transition_log_likelihood) = self.restrict_gibbs_sampling(cluster_label, proposed_K - 1,
			                                                                            model_parameter,
			                                                                            self._restrict_gibbs_sampling_iteration + 1)
			(proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
			 proposed_sigma_inv) = model_parameter
			# self.model_assertion(model_parameter)

			if proposed_count[cluster_label] == 0 or proposed_count[proposed_K - 1] == 0:
				return

			assert numpy.all(proposed_count > 0), (proposed_count, old_proposed_count, cluster_label, proposed_K - 1)

			log_proposal_probability = transition_log_likelihood
		elif self._split_proposal == 2:
			# perform sequential allocation gibbs sampling for split proposal
			model_parameter = self.sequential_allocation_split(cluster_label, model_parameter)

			if model_parameter == None:
				return

			(proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
			 proposed_sigma_inv) = model_parameter
			assert numpy.all(proposed_count > 0)

			log_proposal_probability = (proposed_count[cluster_label] + proposed_count[proposed_K - 1] - 2) * numpy.log(
				2)
		else:
			sys.stderr.write("error: unrecognized split proposal strategy %d...\n" % (self._split_proposal))

		# model_parameter = (proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det, proposed_sigma_inv)
		new_log_posterior = self.log_posterior(model_parameter)

		acceptance_log_probability = log_proposal_probability + new_log_posterior - old_log_posterior
		acceptance_log_probability -= scipy.misc.logsumexp(acceptance_log_probability)
		acceptance_probability = numpy.exp(acceptance_log_probability)

		(proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
		 proposed_sigma_inv) = model_parameter

		if numpy.random.random() < acceptance_probability:
			print "split operation granted from %s to %s with acceptance probability %s" % (
				self._count, proposed_count, acceptance_probability)

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
		(proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
		 proposed_sigma_inv) = model_parameter

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
		proposed_sigma_inv = numpy.vstack((proposed_sigma_inv, numpy.zeros((1, self._D, self._D))))
		proposed_log_sigma_det = numpy.hstack((proposed_log_sigma_det, numpy.zeros(1)))

		model_parameter = (
			proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
			proposed_sigma_inv)
		model_parameter = self.update_cluster_parameters(cluster_label, model_parameter)
		model_parameter = self.update_cluster_parameters(proposed_K - 1, model_parameter)

		return model_parameter

	def sequential_allocation_split(self, cluster_label, model_parameter):
		# sample the data points set
		(proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
		 proposed_sigma_inv) = model_parameter

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
		proposed_sigma_inv = numpy.vstack((proposed_sigma_inv, numpy.zeros((1, self._D, self._D))))
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
			proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
			proposed_sigma_inv)
		model_parameter = self.update_cluster_parameters(cluster_label, model_parameter)
		model_parameter = self.update_cluster_parameters(proposed_K - 1, model_parameter)
		(proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
		 proposed_sigma_inv) = model_parameter

		# perform a split operation
		for data_point_index in data_point_indices_list:
			# compute the probability of being in current cluster
			mean_offset = self._X[[data_point_index], :] - proposed_mu[[cluster_label], :]
			assert mean_offset.shape == (1, self._D)
			cluster_log_probability_1 = -0.5 * proposed_log_sigma_det[cluster_label]
			cluster_log_probability_1 += -0.5 * numpy.dot(
				numpy.dot(mean_offset, proposed_sigma_inv[cluster_label, :, :]), mean_offset.T)
			cluster_log_probability_1 += numpy.log(proposed_count[cluster_label])

			# compute the probability of being in cluster 2
			mean_offset = self._X[[data_point_index], :] - proposed_mu[[proposed_K - 1], :]
			assert mean_offset.shape == (1, self._D)
			cluster_log_probability_2 = -0.5 * proposed_log_sigma_det[proposed_K - 1]
			cluster_log_probability_2 += -0.5 * numpy.dot(
				numpy.dot(mean_offset, proposed_sigma_inv[proposed_K - 1, :, :]), mean_offset.T)
			cluster_log_probability_2 += numpy.log(proposed_count[proposed_K - 1])

			log_ratio_2_over_1 = cluster_log_probability_2 - cluster_log_probability_1
			log_ratio_2_over_1 -= scipy.misc.logsumexp(log_ratio_2_over_1)
			ratio_2_over_1 = numpy.exp(log_ratio_2_over_1)

			# sample a new cluster label for current point
			cluster_probability_1 = 1. / (1. + ratio_2_over_1)
			if numpy.random.random() <= cluster_probability_1:
				new_label = cluster_label
			else:
				new_label = proposed_K - 1

			proposed_label[data_point_index] = new_label
			proposed_count[new_label] += 1
			proposed_sum[new_label, :] += self._X[data_point_index, :]

			model_parameter = (
				proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
				proposed_sigma_inv)
			model_parameter = self.update_cluster_parameters(new_label, model_parameter)
			(proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
			 proposed_sigma_inv) = model_parameter

			assert proposed_count.shape == (proposed_K,)
			assert proposed_sum.shape == (proposed_K, self._D)
			assert proposed_mu.shape == (proposed_K, self._D)
			assert proposed_sigma_inv.shape == (proposed_K, self._D, self._D)
			assert proposed_log_sigma_det.shape == (proposed_K,)
			assert new_label == cluster_label or new_label == proposed_K - 1

		assert proposed_count[cluster_label] > 0 and proposed_count[proposed_K - 1] > 0

		model_parameter = (
			proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
			proposed_sigma_inv)
		return model_parameter

	def restrict_gibbs_sampling(self, cluster_index_1, cluster_index_2, model_parameter,
	                            restricted_gibbs_sampling_iteration=1):
		(proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
		 proposed_sigma_inv) = model_parameter
		data_point_indices = numpy.hstack(
			(numpy.nonzero(proposed_label == cluster_index_2)[0], numpy.nonzero(proposed_label == cluster_index_1)[0]))

		assert len(data_point_indices) == proposed_count[cluster_index_1] + proposed_count[cluster_index_2], (
			len(data_point_indices), proposed_count[cluster_index_1], proposed_count[cluster_index_2])

		# sample the data points set
		for restrict_gibbs_sampling_iteration_index in xrange(restricted_gibbs_sampling_iteration):
			transition_log_likelihood = 0
			for point_index in data_point_indices:
				# get the old label of current point
				old_label = proposed_label[point_index]
				assert old_label == cluster_index_1 or old_label == cluster_index_2, (
					old_label, cluster_index_1, cluster_index_2, point_index)

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
						proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
						proposed_sigma_inv)
					model_parameter = self.update_cluster_parameters(old_label, model_parameter)
					(proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
					 proposed_sigma_inv) = model_parameter

				# compute the probability of being in cluster 1
				mean_offset = self._X[[point_index], :] - proposed_mu[[cluster_index_1], :]
				assert mean_offset.shape == (1, self._D)
				cluster_log_probability_1 = -0.5 * proposed_log_sigma_det[cluster_index_1]
				cluster_log_probability_1 += -0.5 * numpy.dot(
					numpy.dot(mean_offset, proposed_sigma_inv[cluster_index_1, :, :]), mean_offset.T)
				if proposed_count[cluster_index_1] == 0:
					cluster_log_probability_1 += numpy.log(self._alpha_alpha)
				else:
					cluster_log_probability_1 += numpy.log(proposed_count[cluster_index_1])

				# compute the probability of being in cluster 2
				mean_offset = self._X[[point_index], :] - proposed_mu[[cluster_index_2], :]
				assert mean_offset.shape == (1, self._D)
				cluster_log_probability_2 = -0.5 * proposed_log_sigma_det[cluster_index_2]
				cluster_log_probability_2 += -0.5 * numpy.dot(
					numpy.dot(mean_offset, proposed_sigma_inv[cluster_index_2, :, :]), mean_offset.T)
				if proposed_count[cluster_index_2] == 0:
					cluster_log_probability_2 += numpy.log(self._alpha_alpha)
				else:
					cluster_log_probability_2 += numpy.log(proposed_count[cluster_index_2])

				# sample a new cluster label for current point
				ratio_2_over_1 = numpy.exp(cluster_log_probability_2 - cluster_log_probability_1)
				cluster_probability_1 = 1. / (1. + ratio_2_over_1)
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
						proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
						proposed_sigma_inv)
					model_parameter = self.update_cluster_parameters(new_label, model_parameter)
					(proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
					 proposed_sigma_inv) = model_parameter

				assert proposed_count.shape == (proposed_K,)
				assert proposed_sum.shape == (proposed_K, self._D)
				assert proposed_mu.shape == (proposed_K, self._D)
				assert proposed_sigma_inv.shape == (proposed_K, self._D, self._D)
				assert proposed_log_sigma_det.shape == (proposed_K,)
				assert new_label == cluster_index_1 or new_label == cluster_index_2
				assert numpy.all(proposed_count >= 0)

		model_parameter = (
			proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
			proposed_sigma_inv)
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
			proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
			proposed_sigma_inv)

		if self._merge_proposal == 0:
			# perform random merge for merge proposal
			model_parameter = self.random_merge(cluster_label_1, cluster_label_2, model_parameter)

			(proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
			 proposed_sigma_inv) = model_parameter
			assert numpy.all(proposed_count > 0)

			log_proposal_probability = -(proposed_count[cluster_label_1] - 2) * numpy.log(2)
		elif self._merge_proposal == 1:
			# perform restricted gibbs sampling for merge proposal
			(model_parameter, transition_log_probability) = self.restrict_gibbs_sampling(cluster_label_1,
			                                                                             cluster_label_2,
			                                                                             model_parameter,
			                                                                             self._restrict_gibbs_sampling_iteration + 1)

			(proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
			 proposed_sigma_inv) = model_parameter
			assert numpy.all(proposed_count >= 0)
			assert numpy.sum(proposed_count) == self._N
			assert numpy.sum(self._count) == self._N

			if proposed_count[cluster_label_1] == 0 or proposed_count[cluster_label_2] == 0:
				print "merge cluster %d and %d during restricted gibbs sampling step..." % (
					cluster_label_1, cluster_label_2)

				if proposed_count[cluster_label_1] == 0:
					collapsed_cluster = cluster_label_1
				elif proposed_count[cluster_label_2] == 0:
					collapsed_cluster = cluster_label_2

				# since one cluster is empty now, switch it with the last one
				proposed_count[collapsed_cluster] = proposed_count[proposed_K - 1]
				proposed_label[numpy.nonzero(proposed_label == (proposed_K - 1))] = collapsed_cluster
				proposed_sum[collapsed_cluster, :] = proposed_sum[proposed_K - 1, :]
				proposed_mu[collapsed_cluster, :] = proposed_mu[proposed_K - 1, :]
				proposed_sigma_inv[collapsed_cluster, :, :] = proposed_sigma_inv[proposed_K - 1, :, :]
				proposed_log_sigma_det[collapsed_cluster] = proposed_log_sigma_det[proposed_K - 1]

				# remove the very last empty cluster, to remain compact cluster
				proposed_count = numpy.delete(proposed_count, [proposed_K - 1], axis=0)
				proposed_sum = numpy.delete(proposed_sum, [proposed_K - 1], axis=0)
				proposed_mu = numpy.delete(proposed_mu, [proposed_K - 1], axis=0)
				proposed_sigma_inv = numpy.delete(proposed_sigma_inv, [proposed_K - 1], axis=0)
				proposed_log_sigma_det = numpy.delete(proposed_log_sigma_det, [proposed_K - 1], axis=0)
				proposed_K -= 1

				# print proposed_count
				# (proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det, proposed_sigma_inv) = model_parameter
				# print proposed_count
				model_parameter = (
					proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
					proposed_sigma_inv)

			log_proposal_probability = transition_log_probability
		elif self._merge_proposal == 2:
			# perform gibbs sampling for merge proposal
			cluster_log_probability = numpy.log(proposed_count)
			cluster_log_probability = numpy.sum(cluster_log_probability) - cluster_log_probability
			cluster_log_probability -= scipy.misc.logsumexp(cluster_log_probability)
			cluster_probability = numpy.exp(cluster_log_probability)

			# choose a cluster that is inverse proportional to its size
			temp_cluster_probability = numpy.random.multinomial(1, cluster_probability)[numpy.newaxis, :]
			cluster_label = numpy.nonzero(temp_cluster_probability == 1)[1][0]

			model_parameter = self.gibbs_sampling_merge(cluster_label, model_parameter)
			if model_parameter == None:
				return

			(proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
			 proposed_sigma_inv) = model_parameter
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
			sys.stderr.write("error: unrecognized merge proposal strategy %d...\n" % (self._merge_proposal))

		assert proposed_K == len(proposed_log_sigma_det)
		assert numpy.max(proposed_label) < len(proposed_log_sigma_det)
		assert numpy.sum(proposed_count) == self._N
		assert proposed_mu.shape == (proposed_K, self._D)

		# model_parameter = (proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det, proposed_sigma_inv)
		new_log_posterior = self.log_posterior(model_parameter)

		acceptance_log_probability = log_proposal_probability + new_log_posterior - old_log_posterior
		acceptance_log_probability -= scipy.misc.logsumexp(acceptance_log_probability)
		acceptance_probability = numpy.exp(acceptance_log_probability)

		(proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
		 proposed_sigma_inv) = model_parameter
		assert numpy.all(proposed_count > 0)

		if numpy.random.random() < acceptance_probability:
			print "merge operation granted from %s to %s with acceptance probability %s" % (
				self._count, proposed_count, acceptance_probability)

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
		(proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
		 proposed_sigma_inv) = model_parameter

		# perform a merge operation
		proposed_label[numpy.nonzero(proposed_label == cluster_label_2)[0]] = cluster_label_1
		proposed_count[cluster_label_1] += proposed_count[cluster_label_2]
		proposed_sum[cluster_label_1, :] += proposed_sum[cluster_label_2, :]

		# since one cluster is empty now, switch it with the last one
		proposed_count[cluster_label_2] = proposed_count[proposed_K - 1]
		proposed_label[numpy.nonzero(proposed_label == (proposed_K - 1))] = cluster_label_2
		proposed_sum[cluster_label_2, :] = proposed_sum[proposed_K - 1, :]
		proposed_mu[cluster_label_2, :] = proposed_mu[proposed_K - 1, :]
		proposed_sigma_inv[cluster_label_2, :, :] = proposed_sigma_inv[proposed_K - 1, :, :]
		proposed_log_sigma_det[cluster_label_2] = proposed_log_sigma_det[proposed_K - 1]

		# remove the very last empty cluster, to remain compact cluster
		proposed_count = numpy.delete(proposed_count, [proposed_K - 1], axis=0)
		proposed_sum = numpy.delete(proposed_sum, [proposed_K - 1], axis=0)
		proposed_mu = numpy.delete(proposed_mu, [proposed_K - 1], axis=0)
		proposed_sigma_inv = numpy.delete(proposed_sigma_inv, [proposed_K - 1], axis=0)
		proposed_log_sigma_det = numpy.delete(proposed_log_sigma_det, [proposed_K - 1], axis=0)
		proposed_K -= 1

		assert proposed_K == len(proposed_log_sigma_det)
		assert numpy.max(proposed_label) < len(proposed_log_sigma_det)
		assert numpy.sum(proposed_count) == self._N
		assert proposed_mu.shape == (proposed_K, self._D)

		model_parameter = (
			proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
			proposed_sigma_inv)
		model_parameter = self.update_cluster_parameters(cluster_label_1, model_parameter)

		return model_parameter

	def gibbs_sampling_merge(self, cluster_label, model_parameter):
		new_label = self.propose_cluster_to_merge(cluster_label, model_parameter)

		(proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
		 proposed_sigma_inv) = model_parameter

		if new_label != cluster_label:
			# always merge the later cluster to the earlier cluster
			# this is to avoid errors if new_label is the last cluster
			if new_label > cluster_label:
				temp_label = cluster_label
				cluster_label = new_label
				new_label = temp_label

			proposed_label[numpy.nonzero(proposed_label == cluster_label)[0]] = new_label
			proposed_count[new_label] += proposed_count[cluster_label]
			proposed_sum[new_label, :] += proposed_sum[cluster_label, :]

			# since one cluster is empty now, switch it with the last one
			proposed_count[cluster_label] = proposed_count[proposed_K - 1]
			proposed_label[numpy.nonzero(proposed_label == (proposed_K - 1))] = cluster_label
			proposed_sum[cluster_label, :] = proposed_sum[proposed_K - 1, :]
			proposed_mu[cluster_label, :] = proposed_mu[proposed_K - 1, :]
			proposed_sigma_inv[cluster_label, :, :] = proposed_sigma_inv[proposed_K - 1, :, :]
			proposed_log_sigma_det[cluster_label] = proposed_log_sigma_det[proposed_K - 1]

			# remove the very last empty cluster, to remain compact cluster
			proposed_count = numpy.delete(proposed_count, [proposed_K - 1], axis=0)
			proposed_sum = numpy.delete(proposed_sum, [proposed_K - 1], axis=0)
			proposed_mu = numpy.delete(proposed_mu, [proposed_K - 1], axis=0)
			proposed_sigma_inv = numpy.delete(proposed_sigma_inv, [proposed_K - 1], axis=0)
			proposed_log_sigma_det = numpy.delete(proposed_log_sigma_det, [proposed_K - 1], axis=0)
			proposed_K -= 1

			assert new_label < proposed_K
			assert proposed_K == len(proposed_log_sigma_det)
			assert numpy.max(proposed_label) < len(proposed_log_sigma_det)
			assert numpy.sum(proposed_count) == self._N
			assert proposed_mu.shape == (proposed_K, self._D)

			# if these points are merged to another cluster, adjust its covariance matrix
			model_parameter = (
				proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
				proposed_sigma_inv)
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
			(label, K, count, mu, sum, log_sigma_det, sigma_inv) = model_parameter

		if hyper_parameter == None:
			alpha_alpha = self._alpha_alpha
		else:
			alpha_alpha = hyper_parameter

		# log likelihood probability
		log_likelihood = 0.
		for n in xrange(self._N):
			log_likelihood -= 0.5 * self._D * numpy.log(2.0 * numpy.pi) + 0.5 * log_sigma_det[label[n]]
			mean_offset = self._X[n, :][numpy.newaxis, :] - mu[label[n], :]
			assert (mean_offset.shape == (1, self._D))
			log_likelihood -= 0.5 * numpy.dot(numpy.dot(mean_offset, sigma_inv[label[n], :, :]),
			                                  mean_offset.transpose())

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
			(label, K, count, mu, sum, log_sigma_det, sigma_inv) = model_parameter

		if hyper_parameter == None:
			alpha_alpha = self._alpha_alpha
		else:
			alpha_alpha = hyper_parameter

		# log likelihood probability
		log_likelihood = 0.
		for n in xrange(self._N):
			log_likelihood -= 0.5 * self._D * numpy.log(2.0 * numpy.pi) + 0.5 * log_sigma_det[label[n]]
			mean_offset = self._X[n, :][numpy.newaxis, :] - mu[label[n], :]
			assert (mean_offset.shape == (1, self._D))
			log_likelihood -= 0.5 * numpy.dot(numpy.dot(mean_offset, sigma_inv[label[n], :, :]),
			                                  mean_offset.transpose())

		return log_likelihood

	"""
	"""

	def export_snapshot(self, output_directory):
		label_path = os.path.join(output_directory, "label-%d" % (self._iteration_counter))
		numpy.savetxt(label_path, self._label, fmt="%d")

		mu_path = os.path.join(output_directory, "mu-%d" % (self._iteration_counter))
		numpy.savetxt(mu_path, self._mu)

		sigma_path = os.path.join(output_directory, "sigma-%d" % (self._iteration_counter))
		sigma_matrices = numpy.zeros(self._sigma_inv.shape)
		for k in range(self._K):
			sigma_matrices[k, :, :] = numpy.linalg.pinv(self._sigma_inv[k, :, :])
		numpy.savetxt(sigma_path, numpy.reshape(sigma_matrices, (self._K * self._D, self._D)))

		'''
		sigma_inv_path = os.path.join(output_directory, "sigma_inv-%d" % (self._iteration_counter))
		sigma_inv_matrices = numpy.zeros((self._K*self._D, self._D))
		for k in range(self._K):
			sigma_inv_matrices[k*self._D:(k+1)*self._D, :] = self._sigma_inv[k, :, :]
		numpy.savetxt(sigma_inv_path, sigma_inv_matrices)
		'''

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
			(proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det,
			 proposed_sigma_inv) = model_parameter

		# if this cluster is empty, no need to resample the cluster assignment 
		assert proposed_count[cluster_label] > 0

		# find the index of the data point in the current cluster
		data_point_indices = numpy.nonzero(proposed_label == cluster_label)[0]

		# compute the prior of being in any of the clusters
		cluster_prior = numpy.copy(proposed_count)
		cluster_prior[cluster_label] = self._alpha_alpha

		cluster_log_prior = scipy.special.gammaln(cluster_prior + proposed_count[cluster_label])
		cluster_log_prior -= scipy.special.gammaln(cluster_prior)

		# adjust for current cluster label
		cluster_log_prior[cluster_label] = numpy.log(self._alpha_alpha) + scipy.special.gammaln(
			proposed_count[cluster_label])

		# cluster_log_prior += scipy.special.gammaln(self._N - proposed_count[cluster_label] + self._alpha_alpha)
		# cluster_log_prior -= scipy.special.gammaln(self._N + self._alpha_alpha)

		# initialize the likelihood vector for all clusters
		cluster_log_likelihood = numpy.zeros(proposed_K)

		# compute the likelihood for the existing clusters
		for k in xrange(proposed_K):
			if self._count[k] == 0:
				cluster_log_likelihood[k] = negative_infinity
				continue

			if k == cluster_label:
				# compute the likelihood for new cluster
				mean_offset = self._X[data_point_indices, :] - self._mu_0
				assert mean_offset.shape == (proposed_count[cluster_label], self._D)

				cluster_log_likelihood[cluster_label] = -0.5 * proposed_count[cluster_label] * self._log_sigma_det_0
				cluster_log_likelihood[cluster_label] += -0.5 * numpy.sum(
					numpy.dot(numpy.dot(mean_offset, self._sigma_inv_0), mean_offset.T))
			else:
				mean_offset = self._X[data_point_indices, :] - proposed_mu[[k], :]
				assert mean_offset.shape == (proposed_count[cluster_label], self._D)

				cluster_log_likelihood[k] = -0.5 * proposed_count[cluster_label] * proposed_log_sigma_det[k]
				cluster_log_likelihood[k] += -0.5 * numpy.sum(
					numpy.dot(numpy.dot(mean_offset, proposed_sigma_inv[k, :, :]), mean_offset.T))

		# normalize the posterior distribution
		cluster_log_posterior = cluster_log_prior + cluster_log_likelihood
		cluster_log_posterior -= scipy.misc.logsumexp(cluster_log_posterior)
		cluster_posterior = numpy.exp(cluster_log_posterior)

		cdf = numpy.cumsum(cluster_posterior)
		new_label = numpy.uint(numpy.nonzero(cdf >= numpy.random.random())[0][0])
		assert new_label >= 0 and new_label < proposed_K

		return new_label

	def resample_component(self):
		if self._K == 1:
			return

		'''
		probability_to_merge = 1.0 / self._count
		probability_to_merge /= numpy.sum(probability_to_merge)
		temp_label_probability = numpy.random.multinomial(1, probability_to_merge)[numpy.newaxis, :]
		cluster_label = numpy.nonzero(temp_label_probability == 1)[1][0]
		'''

		cluster_label = numpy.random.randint(0, self._K)

		# resample the cluster assignment only if this cluster is not empty
		if self._count[cluster_label] > 0:
			new_label = self.propose_cluster_to_merge(cluster_label)

			# find the index of the data point in the current cluster
			data_point_indices = numpy.nonzero(self._label == cluster_label)[0]

			if new_label != cluster_label:
				print "merge cluster %d and %d after component resampling..." % (new_label, cluster_label)

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
		for cluster_label in xrange(len(non_empty_cluster)):
			self._label[numpy.nonzero(self._label == non_empty_cluster[cluster_label])[0]] = cluster_label

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
				print "merge cluster %d and %d after component resampling..." % (new_label, cluster_label)

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
		for cluster_label in xrange(len(non_empty_cluster)):
			self._label[numpy.nonzero(self._label == non_empty_cluster[cluster_label])[0]] = cluster_label

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
			(label, K, count, mu, sum, log_sigma_det, sigma_inv) = model_parameter

		test_count = numpy.zeros(K)
		for point_index in numpy.random.permutation(xrange(self._N)):
			test_count[label[point_index]] += 1
		assert numpy.all(test_count == count)

		return


if __name__ == '__main__':
	raise NotImplementedError()
