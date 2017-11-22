"""
@author: Ke Zhai (zhaike@cs.umd.edu)
"""
import datetime
import optparse
import os
import sys

import numpy

from plot_clusters import plot_data, plot_cluster


def parse_args():
	parser = optparse.OptionParser()
	parser.set_defaults(  # parameter set 1
		output_directory=None,
		# number_of_clusters=2,
		number_of_points=100,
		number_of_dimensions=2,
		alpha_alpha=1,
		grid_scale=100,

		# intra_cluster_distance=0,
	)
	# parameter set 1
	parser.add_option("--output_directory", type="string", dest="output_directory",
	                  help="output directory [None]")
	# parser.add_option("--number_of_clusters", type="int", dest="number_of_clusters",
	# help="number of clusters [2]")
	parser.add_option("--number_of_points", type="int", dest="number_of_points",
	                  help="number of points [100]")
	parser.add_option("--number_of_dimensions", type="int", dest="number_of_dimensions",
	                  help="number of dimensions [2]")
	parser.add_option("--alpha_alpha", type="float", dest="alpha_alpha",
	                  help="hyperparameter for prior distribution [1 (default)]")
	parser.add_option("--grid_scale", type="float", dest="grid_scale",
	                  help="grid scale [100 (default)]")

	# parser.add_option("--intra_cluster_distance", type="int", dest="intra_cluster_distance",
	# help="intra cluster radius [0 (default): no specified intra-cluster distance]")

	(options, args) = parser.parse_args()
	return options


def main():
	options = parse_args()

	# parameter set 1
	assert (options.output_directory != None)
	output_directory = options.output_directory
	# number_of_clusters = options.number_of_clusters
	number_of_points = options.number_of_points
	number_of_dimensions = options.number_of_dimensions
	alpha_alpha = options.alpha_alpha
	assert alpha_alpha > 0
	grid_scale = options.grid_scale
	assert grid_scale > 0
	# intra_cluster_distance = options.intra_cluster_distance

	now = datetime.datetime.now()
	suffix = now.strftime("%y%m%d-%H%M%S") + ""
	# suffix += "-c%d" % (number_of_clusters)
	suffix += "-d%d" % (number_of_dimensions)
	suffix += "-p%d" % (number_of_points)
	suffix += "-aa%g" % (alpha_alpha)
	suffix += "/"
	output_directory = os.path.join(output_directory, suffix)
	os.mkdir(os.path.abspath(output_directory))

	# mean_vectors = numpy.zeros((1, number_of_dimensions))
	# mean_vectors[0, :] = (numpy.random.random(number_of_dimensions) - 0.5) * 2 * grid_scale
	# cov_matrices = numpy.zeros((1, number_of_dimensions, number_of_dimensions))
	# diagonal_elements = numpy.ones(number_of_dimensions) + 0.1*numpy.random.random()
	# cov = numpy.diagflat(diagonal_elements)
	# cov_matrices[0, :, :] = cov
	# number_of_clusters = 1
	# cluster_counts = numpy.ones(1)
	# label_vector = numpy.zeros(1)
	# data_vectors = numpy.random.multivariate_normal(mean_vectors[0, :], cov_matrices[0, :, :])

	mean_vectors = numpy.zeros((0, number_of_dimensions))
	cov_matrices = numpy.zeros((0, number_of_dimensions, number_of_dimensions))
	number_of_clusters = 0
	cluster_counts = numpy.zeros(0)
	label_vector = numpy.zeros(0)
	data_vectors = numpy.zeros((0, number_of_dimensions))
	for x in xrange(number_of_points):
		if numpy.random.random() <= alpha_alpha / (alpha_alpha + numpy.sum(cluster_counts)):
			mean_vectors = numpy.vstack((mean_vectors, numpy.zeros((1, number_of_dimensions))))
			mean_vectors[-1, :] = (numpy.random.random(number_of_dimensions) - 0.5) * 2 * grid_scale
			'''
			if intra_cluster_distance>0:
				mean_vectors[-1, :] = [(number_of_clusters/intra_cluster_distance)*intra_cluster_distance, (number_of_clusters%intra_cluster_distance)*intra_cluster_distance]
			else:
				mean_vectors[-1, :] = (numpy.random.random(number_of_dimensions) - 0.5) * 2 * grid_scale
			'''

			cov_matrices = numpy.vstack((cov_matrices, numpy.zeros((1, number_of_dimensions, number_of_dimensions))))
			diagonal_elements = numpy.ones(number_of_dimensions) + 0.1 * numpy.random.random()
			cov = numpy.diagflat(diagonal_elements)
			cov_matrices[-1, :, :] = cov

			cluster_counts = numpy.hstack((cluster_counts, numpy.zeros(1)))

			number_of_clusters += 1
			new_label = number_of_clusters - 1
		else:
			cluster_probability = 1.0 * cluster_counts / numpy.sum(cluster_counts)
			temp_label_probability = numpy.random.multinomial(1, cluster_probability)[numpy.newaxis, :]
			new_label = numpy.nonzero(temp_label_probability == 1)[1][0]

		label_vector = numpy.hstack((label_vector, new_label))
		data_vectors = numpy.vstack((data_vectors, numpy.random.multivariate_normal(mean_vectors[new_label, :],
		                                                                            cov_matrices[new_label, :, :])))

		cluster_counts[new_label] += 1

	mean_vectors_mean = numpy.mean(mean_vectors, axis=0)[numpy.newaxis, :]
	mean_vectors -= mean_vectors_mean
	data_vectors -= mean_vectors_mean

	cluster_probability = 1.0 * cluster_counts / numpy.sum(cluster_counts)
	cluster_probability = numpy.sort(cluster_probability)[::-1]
	print cluster_probability

	assert numpy.sum(cluster_counts) == number_of_points, (cluster_counts, numpy.sum(cluster_counts), number_of_points)
	assert numpy.all(cluster_counts) > 0
	assert mean_vectors.shape == (number_of_clusters, number_of_dimensions)
	assert cov_matrices.shape == (number_of_clusters, number_of_dimensions, number_of_dimensions)

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
