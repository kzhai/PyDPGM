import cPickle
import datetime
import optparse
import os
import time

import numpy


def parse_args():
	parser = optparse.OptionParser()
	parser.set_defaults(  # parameter set 1
		input_directory=None,
		output_directory=None,
		# dataset_name=None,

		# parameter set 2
		alpha_alpha=1,
		# alpha_kappa=1,
		# alpha_nu=1,
		# mu_0=None,
		# lambda_0=None,
		training_iterations=1000,
		snapshot_interval=100,

		# parameter set 3
		split_proposal=0,
		merge_proposal=0,
		split_merge_heuristics=-1,
	)
	# parameter set 1
	parser.add_option("--input_directory", type="string", dest="input_directory",
	                  help="input directory [None]")
	parser.add_option("--output_directory", type="string", dest="output_directory",
	                  help="output directory [None]")
	# parser.add_option("--dataset_name", type="string", dest="dataset_name",
	# help="the corpus name [None]")

	# parameter set 2
	parser.add_option("--alpha_alpha", type="float", dest="alpha_alpha",
	                  help="hyper-parameter for Dirichlet process of cluster [1]")
	# parser.add_option("--alpha_kappa", type="float", dest="alpha_kappa",
	# help="hyper-parameter for top level Dirichlet process of distribution over topics [1]")
	# parser.add_option("--alpha_nu", type="float", dest="alpha_nu",
	# help="hyper-parameter for bottom level Dirichlet process of distribution over topics [1]")
	parser.add_option("--training_iterations", type="int", dest="training_iterations",
	                  help="number of training iterations [1000]")
	parser.add_option("--snapshot_interval", type="int", dest="snapshot_interval",
	                  help="snapshot interval [100]")

	# parameter set 3                  
	parser.add_option("--merge_proposal", type="int", dest="merge_proposal",
	                  help="propose merge operation via [ " +
	                       "0 (default): metropolis-hastings, " +
	                       "1: restricted gibbs sampler and metropolis-hastings, " +
	                       "2: gibbs sampler and metropolis-hastings " +
	                       "]")
	parser.add_option("--split_proposal", type="int", dest="split_proposal",
	                  help="propose split operation via [ " +
	                       "0 (default): metropolis-hastings, " +
	                       "1: restricted gibbs sampler and metropolis-hastings, " +
	                       "2: sequential allocation and metropolis-hastings " +
	                       "]")
	parser.add_option("--split_merge_heuristics", type="int", dest="split_merge_heuristics",
	                  help="split-merge heuristics [ " +
	                       "-1 (default): no split-merge operation, " +
	                       "0: component resampling, " +
	                       "1: random choose candidate clusters by points, " +
	                       "2: random choose candidate clusters by point-cluster, " +
	                       "3: random choose candidate clusters by clusters " +
	                       "]")
	# parser.add_option("--sampling_approach", type="int", dest="sampling_approach",
	# help="sampling approach and heuristic [ " + 
	# "0 (default): collapsed gibbs sampling, " + 
	# "1: blocked gibbs sampling]")

	(options, args) = parser.parse_args()
	return options


def main():
	options = parse_args()

	# parameter set 1
	# assert(options.dataset_name!=None)
	assert (options.input_directory != None)
	assert (options.output_directory != None)

	# dataset_name = options.dataset_name
	input_directory = options.input_directory
	input_directory = input_directory.rstrip("/")
	dataset_name = os.path.basename(input_directory)
	# input_directory = os.path.join(input_directory, dataset_name)

	output_directory = options.output_directory
	if not os.path.exists(output_directory):
		os.mkdir(output_directory)
	output_directory = os.path.join(output_directory, dataset_name)
	if not os.path.exists(output_directory):
		os.mkdir(output_directory)

	# Dataset
	input_file_path = os.path.join(input_directory, 'train.dat')
	train_data = numpy.loadtxt(input_file_path)
	print "successfully load all training data..."

	# parameter set 2
	assert options.alpha_alpha > 0
	alpha_alpha = options.alpha_alpha
	# assert options.alpha_kappa>0
	# alpha_kappa = options.alpha_kappa
	# assert options.alpha_nu>0
	# alpha_nu=options.alpha_nu
	if options.training_iterations > 0:
		training_iterations = options.training_iterations
	if options.snapshot_interval > 0:
		snapshot_interval = options.snapshot_interval
	# sampling_approach = options.sampling_approach

	# parameter set 3
	split_merge_heuristics = options.split_merge_heuristics
	split_proposal = options.split_proposal
	merge_proposal = options.merge_proposal

	# create output directory
	now = datetime.datetime.now()
	suffix = now.strftime("%y%m%d-%H%M%S") + ""
	suffix += "-%s" % ("dpgm")
	suffix += "-I%d" % (training_iterations)
	suffix += "-S%d" % (snapshot_interval)
	suffix += "-aa%g" % (alpha_alpha)
	# suffix += "-ak%g" % (alpha_kappa)
	# suffix += "-an%g" % (alpha_nu)
	# suffix += "-SA%d" % (sampling_approach)
	if split_merge_heuristics >= 0:
		suffix += "-smh%d" % (split_merge_heuristics)
	if split_merge_heuristics >= 1:
		suffix += "-sp%d" % (split_proposal)
		suffix += "-mp%d" % (merge_proposal)
	suffix += "/"

	output_directory = os.path.join(output_directory, suffix)
	os.mkdir(os.path.abspath(output_directory))

	# store all the options to a file
	options_output_file = open(output_directory + "option.txt", 'w')
	# parameter set 1
	options_output_file.write("input_directory=" + input_directory + "\n")
	options_output_file.write("dataset_name=" + dataset_name + "\n")
	# parameter set 2
	options_output_file.write("alpha_alpha=" + str(alpha_alpha) + "\n")
	# options_output_file.write("alpha_nu=" + str(alpha_nu) + "\n")
	# options_output_file.write("alpha_kappa=" + str(alpha_kappa) + "\n")
	options_output_file.write("training_iteration=%d\n" % training_iterations)
	options_output_file.write("snapshot_interval=%d\n" % snapshot_interval)
	# options_output_file.write("sampling_approach=%d\n" % sampling_approach)
	# parameter set 3
	if split_merge_heuristics >= 0:
		options_output_file.write("split_merge_heuristics=%d\n" % split_merge_heuristics)
	if split_merge_heuristics >= 1:
		options_output_file.write("split_proposal=%d\n" % split_proposal)
		options_output_file.write("merge_proposal=%d\n" % merge_proposal)
	options_output_file.close()

	print "========== ========== ========== ========== =========="
	# parameter set 1
	print "output_directory=" + output_directory
	print "input_directory=" + input_directory
	print "dataset_name=" + dataset_name
	# parameter set 2
	print "alpha_alpha=" + str(alpha_alpha)
	# print "alpha_nu=" + str(alpha_nu)
	# print "alpha_kappa=" + str(alpha_kappa)
	print "training_iteration=%d" % (training_iterations)
	print "snapshot_interval=%d" % (snapshot_interval)
	# print "sampling_approach=%d" % (sampling_approach)
	# parameter set 3
	if split_merge_heuristics >= 0:
		print "split_merge_heuristics=%d" % (split_merge_heuristics)
	if split_merge_heuristics >= 1:
		print "split_proposal=%d" % split_proposal
		print "merge_proposal=%d" % merge_proposal
	print "========== ========== ========== ========== =========="

	import monte_carlo
	dpgm = monte_carlo.MonteCarlo(split_merge_heuristics, split_proposal, merge_proposal)
	dpgm._initialize(train_data, alpha_alpha)

	dpgm.export_snapshot(output_directory)

	for iteration in xrange(training_iterations):
		clock = time.time()
		log_likelihood = dpgm.learning()
		clock = time.time() - clock
		print 'training iteration %d finished in %f seconds: number-of-clusters = %d, log-likelihood = %f' % (
			dpgm._iteration_counter, clock, dpgm._K, log_likelihood)

		if ((dpgm._iteration_counter) % snapshot_interval == 0):
			dpgm.export_snapshot(output_directory)
			model_snapshot_path = os.path.join(output_directory, 'model-' + str(dpgm._iteration_counter))
			cPickle.dump(dpgm, open(model_snapshot_path, 'wb'))


if __name__ == '__main__':
	main()
