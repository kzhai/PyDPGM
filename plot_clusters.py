"""
@author: Ke Zhai (zhaike@cs.umd.edu)
"""
import optparse
import os
import sys

import matplotlib
import matplotlib.pyplot
import numpy

delta = 0.1
margin = 1

color_options = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
marker_options = ['o', 'x', '+', 's', '*']
color_marker = []
for marker in marker_options:
	for color in color_options:
		color_marker.append(color + marker)


def parse_args():
	parser = optparse.OptionParser()
	parser.set_defaults(  # parameter set 1
		input_directory=None,
		# dataset_name=None,

		output_directory=None,
		snapshot_index=-1,
	)
	# parameter set 1
	parser.add_option("--input_directory", type="string", dest="input_directory",
	                  help="input directory [None]")
	# parser.add_option("--dataset_name", type="string", dest="dataset_name",
	# help="the corpus name [None]")

	parser.add_option("--output_directory", type="string", dest="output_directory",
	                  help="output directory [None]")
	parser.add_option("--snapshot_index", type="int", dest="snapshot_index",
	                  help="snapshot index [-1]")

	(options, args) = parser.parse_args()
	return options


def main():
	options = parse_args()

	# parameter set 1
	# assert(options.dataset_name!=None)
	assert (options.input_directory is not None)

	# dataset_name = options.dataset_name
	# input_directory = options.input_directory
	# input_directory = os.path.join(input_directory, dataset_name)

	input_directory = options.input_directory
	input_directory = input_directory.rstrip("/")
	dataset_name = os.path.basename(input_directory)

	input_file_path = os.path.join(input_directory, 'train.dat')
	train_data = numpy.loadtxt(input_file_path)
	print "successfully load all training data..."

	plot_data(train_data, os.path.join(input_directory, "train.pdf"))

	# plot the true clusters
	label_snapshot = os.path.join(input_directory, "label.dat")
	if os.path.exists(label_snapshot):
		labels = numpy.loadtxt(label_snapshot)
	else:
		labels = None

	mu_snapshot = os.path.join(input_directory, "mu.dat")
	if os.path.exists(mu_snapshot):
		mu_vectors = numpy.loadtxt(mu_snapshot)
		(K, D) = mu_vectors.shape
	else:
		mu_vectors = None

	sigma_snapshot = os.path.join(input_directory, "sigma.dat")
	if os.path.exists(sigma_snapshot):
		sigma_matrices = numpy.loadtxt(sigma_snapshot)
		sigma_matrices = numpy.reshape(sigma_matrices, (K, D, D))
	else:
		sigma_matrices = None

	if labels is not None:
		plot_cluster(train_data, labels, mu_vectors, sigma_matrices, os.path.join(input_directory, "cluster.pdf"))

	# plot the inferred clusters
	output_directory = options.output_directory
	if output_directory is None:
		return
	if not os.path.exists(output_directory):
		sys.stderr.write("error: directory %s does not exist..." % output_directory)
		return

	output_directory = output_directory.rstrip("/")
	# output_directory_parent = os.path.dirname(output_directory)
	# output_directory_parent = output_directory_parent.rstrip("/")
	assert os.path.basename(output_directory) == dataset_name, (output_directory, output_directory, dataset_name)
	snapshot_index = options.snapshot_index

	for model_name in os.listdir(output_directory):
		model_directory = os.path.join(output_directory, model_name)
		if os.path.isfile(model_directory):
			continue

		if snapshot_index == -1:
			for file_name in os.listdir(model_directory):
				if not file_name.startswith("label-"):
					continue

				snapshot_index = int(file_name.split("-")[-1])
				if snapshot_index == 0:
					continue

				plot_snapshot(train_data, model_directory, snapshot_index)
		else:
			plot_snapshot(train_data, model_directory, snapshot_index)


def plot_snapshot(train_data, output_directory, snapshot_index):
	label_snapshot = os.path.join(output_directory, "label-%d" % snapshot_index)
	if not os.path.exists(label_snapshot):
		sys.stderr.write("error: file %s does not exist..." % label_snapshot)
		return
	labels = numpy.loadtxt(label_snapshot)

	mu_snapshot = os.path.join(output_directory, "mu-%d" % snapshot_index)
	if os.path.exists(mu_snapshot):
		mu_vectors = numpy.loadtxt(mu_snapshot)
		(K, D) = mu_vectors.shape
	else:
		mu_vectors = None
		print "warning: file %s does not exist..." % mu_snapshot

	sigma_snapshot = os.path.join(output_directory, "sigma-%d" % (snapshot_index))
	if os.path.exists(sigma_snapshot):
		sigma_matrices = numpy.loadtxt(sigma_snapshot)
		sigma_matrices = numpy.reshape(sigma_matrices, (K, D, D))
	else:
		sigma_matrices = None
		print "warning: file %s does not exist..." % sigma_snapshot

	output_figure_path = os.path.join(output_directory, "cluster-%d.pdf" % (snapshot_index))
	plot_cluster(train_data, labels, mu_vectors, sigma_matrices, output_figure_path)


def plot_data(train_data, figure_path=None, figure_title=None):
	(N, D) = train_data.shape
	assert D == 2

	figure = matplotlib.pyplot.figure()
	ax = figure.add_subplot(111)

	ax.set_xlim(numpy.min(train_data[:, 0]) - 1, numpy.max(train_data[:, 0]) + 1)
	ax.set_ylim(numpy.min(train_data[:, 1]) - 1, numpy.max(train_data[:, 1]) + 1)

	matplotlib.rcParams['xtick.direction'] = 'out'
	matplotlib.rcParams['ytick.direction'] = 'out'
	matplotlib.rcParams['axes.unicode_minus'] = False

	matplotlib.pyplot.plot(train_data[:, 0], train_data[:, 1], color_marker[0])

	if figure_title is not None:
		matplotlib.pyplot.title(figure_title)

	if figure_path is None:
		matplotlib.pyplot.show()
	else:
		matplotlib.pyplot.savefig(figure_path)


def plot_cluster(train_data, labels=None, mu_vectors=None, sigma_matrices=None, figure_path=None, figure_title=None):
	(N, D) = train_data.shape
	assert D == 2

	if labels is None:
		labels = numpy.zeros(D)

	K = len(numpy.unique(labels))
	assert numpy.all(labels >= 0) and numpy.all(labels < K)

	if mu_vectors is not None and sigma_matrices is not None:
		assert mu_vectors.shape == (K, D)
		assert sigma_matrices.shape == (K, D, D)

	figure = matplotlib.pyplot.figure()

	matplotlib.rcParams['xtick.direction'] = 'out'
	matplotlib.rcParams['ytick.direction'] = 'out'
	matplotlib.rcParams['axes.unicode_minus'] = False

	x = numpy.arange(numpy.min(train_data[:, 0]) - 1, numpy.max(train_data[:, 0]) + 1, delta)
	y = numpy.arange(numpy.min(train_data[:, 1]) - 1, numpy.max(train_data[:, 1]) + 1, delta)
	X, Y = numpy.meshgrid(x, y)

	for k in xrange(K):
		points_indices = numpy.nonzero(labels == k)
		matplotlib.pyplot.plot(train_data[points_indices, 0], train_data[points_indices, 1], color_marker[k])

		if sigma_matrices is not None and mu_vectors is not None:
			sigma_x = sigma_matrices[k, 0, 0]
			sigma_y = sigma_matrices[k, 1, 1]
			sigma_xy = sigma_matrices[k, 0, 1]
			mu_x = mu_vectors[k, 0]
			mu_y = mu_vectors[k, 1]

			Z = matplotlib.mlab.bivariate_normal(X, Y, sigma_x, sigma_y, mu_x, mu_y, sigma_xy)
			contour = matplotlib.pyplot.contour(X, Y, Z)
			matplotlib.pyplot.clabel(contour, inline=1, fontsize=10)

	if figure_title is not None:
		matplotlib.pyplot.title(figure_title)

	if figure_path is None:
		matplotlib.pyplot.show()
	else:
		matplotlib.pyplot.savefig(figure_path)


if __name__ == '__main__':
	main()
