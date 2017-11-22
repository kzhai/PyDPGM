"""
@author: Ke Zhai (zhaike@cs.umd.edu)
"""
import datetime
import optparse
import os
import sys

import numpy

from plot_clusters import plot_data, plot_cluster

grid_scale = 10


def parse_args():
	parser = optparse.OptionParser()
	parser.set_defaults(  # parameter set 1
		output_directory=None,
		number_of_clusters=2,
		number_of_points=100,
		number_of_dimensions=2,
		alpha_alpha=0,
	)
	# parameter set 1
	parser.add_option("--output_directory", type="string", dest="output_directory",
	                  help="output directory [None]")
	parser.add_option("--number_of_clusters", type="int", dest="number_of_clusters",
	                  help="number of clusters [2]")
	parser.add_option("--number_of_points", type="int", dest="number_of_points",
	                  help="number of points [100]")
	parser.add_option("--number_of_dimensions", type="int", dest="number_of_dimensions",
	                  help="number of dimensions [2]")
	parser.add_option("--alpha_alpha", type="float", dest="alpha_alpha",
	                  help="hyperparameter for prior distribution [0 (default): uniform, -: powerlaw, +: dirichlet process]")

	(options, args) = parser.parse_args()
	return options


def main():
	options = parse_args()

	# parameter set 1
	assert (options.output_directory is not None)
	output_directory = options.output_directory
	number_of_clusters = options.number_of_clusters
	number_of_points = options.number_of_points
	number_of_dimensions = options.number_of_dimensions
	alpha_alpha = options.alpha_alpha

	now = datetime.datetime.now()
	suffix = now.strftime("%y%m%d-%H%M%S") + ""
	suffix += "-d%d" % (number_of_dimensions)
	suffix += "-p%d" % (number_of_points)
	suffix += "-aa%g" % (alpha_alpha)
	suffix += "-c%d" % (number_of_clusters)
	suffix += "/"
	output_directory = os.path.join(output_directory, suffix)
	os.mkdir(os.path.abspath(output_directory))

	data_vectors = numpy.zeros((0, number_of_dimensions))
	label_vector = numpy.zeros(0, dtype=numpy.int)

	mean_vectors = numpy.zeros((number_of_clusters, number_of_dimensions))
	cov_matrices = numpy.zeros((number_of_clusters, number_of_dimensions, number_of_dimensions))
	for x in xrange(number_of_clusters):
		mean_vectors[x, :] = (numpy.random.random(number_of_dimensions) - 0.5) * 2 * grid_scale
		# diagonal_elements = 1 + 0.1*numpy.random.random(number_of_dimensions)
		diagonal_elements = numpy.ones(number_of_dimensions) + 0.1 * numpy.random.random()
		# diagonal_elements = numpy.random.random(number_of_dimensions) * 2
		cov = numpy.diagflat(diagonal_elements)
		# cov = numpy.eye(number_of_dimensions)
		cov_matrices[x, :, :] = cov

	if alpha_alpha > 0:
		telescope_beta = numpy.random.beta(1, alpha_alpha, number_of_clusters - 1)
		one_minus_telescope_data = 1 - telescope_beta
		telescope_beta = numpy.hstack((telescope_beta, numpy.ones(1)))
		one_minus_telescope_data = numpy.hstack((numpy.ones(1), one_minus_telescope_data))
		cumprod_one_minus_telescope_data = numpy.cumprod(one_minus_telescope_data)
		cluster_probability = telescope_beta * cumprod_one_minus_telescope_data
	elif alpha_alpha < 0:
		# power-law distribution
		stick_weights = numpy.random.beta(1, 1, number_of_clusters - 1)
		left_over_sticks = numpy.cumprod(1 - stick_weights)
		left_over_sticks = numpy.hstack((numpy.ones(1), left_over_sticks))
		stick_weights = numpy.hstack((stick_weights, numpy.ones(1)))
		cluster_probability = stick_weights * left_over_sticks
	else:
		# uniform distribution
		cluster_probability = numpy.zeros(number_of_clusters) + 1.0 / number_of_clusters

	cluster_probability = numpy.sort(cluster_probability)[::-1]
	print cluster_probability

	for x in xrange(number_of_points):
		temp_label_probability = numpy.random.multinomial(1, cluster_probability)[numpy.newaxis, :]
		label = numpy.nonzero(temp_label_probability == 1)[1][0]

		label_vector = numpy.hstack((label_vector, label))
		data_vectors = numpy.vstack(
			(data_vectors, numpy.random.multivariate_normal(mean_vectors[label, :], cov_matrices[label, :, :])))

	data_file_path = os.path.join(output_directory, "train.dat")
	numpy.savetxt(data_file_path, data_vectors)
	label_file_path = os.path.join(output_directory, "label.dat")
	numpy.savetxt(label_file_path, label_vector)
	mu_file_path = os.path.join(output_directory, "mu.dat")
	numpy.savetxt(mu_file_path, mean_vectors)
	sigma_file_path = os.path.join(output_directory, "sigma.dat")
	sigma_matrices = numpy.reshape(cov_matrices, (number_of_clusters * number_of_dimensions, number_of_dimensions))
	numpy.savetxt(sigma_file_path, sigma_matrices)

	if number_of_dimensions != 2:
		sys.stderr.write("unable to visualize data points and clusters with dimension %d...\n" % number_of_dimensions)
		return

	output_data_figure_path = os.path.join(output_directory, "train.pdf")
	plot_data(data_vectors, output_data_figure_path)
	output_cluster_figure_path = os.path.join(output_directory, "cluster.pdf")
	plot_cluster(data_vectors, label_vector, mean_vectors, cov_matrices, output_cluster_figure_path)


if __name__ == '__main__':
	main()
