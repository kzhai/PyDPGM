import cPickle
import datetime
import optparse
import os
import re
import shutil
import sys
import time

# model_settings_pattern = re.compile('\d+-\d+-dpgm-I(?P<iteration>\d+)-S(?P<snapshot>\d+)-aa(?P<alpha>[\d\.]+)(-smh(?P<smh>[\d]+))?(-sp(?P<sp>[\d]+)-mp(?P<mp>[\d]+))?')
model_settings_pattern = re.compile(
	'\d+-\d+-dpgm-I(?P<iteration>\d+)-S(?P<snapshot>\d+)-aa(?P<alpha>[\d\.]+)(-(?P<mode>.+))?')


def parse_args():
	parser = optparse.OptionParser()
	parser.set_defaults(  # parameter set 1
		# input_file=None,
		model_directory=None,
		snapshot_index=-1,

		# parameter set 2
		output_directory=None,
		training_iterations=-1,
		snapshot_interval=-1,

		# parameter set 3
		split_proposal=0,
		merge_proposal=0,
		split_merge_heuristics=-2,
	)
	# parameter set 1
	# parser.add_option("--input_file", type="string", dest="input_file",
	# help="input directory [None]")
	# parser.add_option("--input_directory", type="string", dest="input_directory",
	# help="input directory [None]")
	parser.add_option("--model_directory", type="string", dest="model_directory",
	                  help="model directory [None]")
	parser.add_option("--snapshot_index", type="int", dest="snapshot_index",
	                  help="snapshot index [-1]")
	# parser.add_option("--training_iterations", type="int", dest="training_iterations",
	# help="number of training iterations [1000]")
	# parser.add_option("--dataset_name", type="string", dest="dataset_name",
	# help="the corpus name [None]")

	# parameter set 2
	parser.add_option("--output_directory", type="string", dest="output_directory",
	                  help="output directory [None]")
	# parser.add_option("--alpha_alpha", type="float", dest="alpha_alpha",
	# help="hyper-parameter for Dirichlet process of cluster [1]")
	# parser.add_option("--alpha_kappa", type="float", dest="alpha_kappa",
	# help="hyper-parameter for top level Dirichlet process of distribution over topics [1]")
	# parser.add_option("--alpha_nu", type="float", dest="alpha_nu",
	# help="hyper-parameter for bottom level Dirichlet process of distribution over topics [1]")
	parser.add_option("--training_iterations", type="int", dest="training_iterations",
	                  help="number of training iterations [1000]")
	parser.add_option("--snapshot_interval", type="int", dest="snapshot_interval",
	                  help="snapshot interval [-1 (default): remain unchanged]")

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
	                       "-2 (default): remain unchanged, " +
	                       "-1: no split-merge operation, " +
	                       "0: component resampling, " +
	                       "1: random choose candidate clusters by points, " +
	                       "2: random choose candidate clusters by point-cluster, " +
	                       "3: random choose candidate clusters by clusters " +
	                       "]")

	(options, args) = parser.parse_args()
	return options


def main():
	options = parse_args()

	# parameter set 1
	# assert(options.input_file != None)
	# input_file = options.input_file
	# data_prime = numpy.loadtxt(input_file)

	assert (options.model_directory != None)
	model_directory = options.model_directory

	if not os.path.exists(model_directory):
		sys.stderr.write("model directory %s not exists...\n" % (model_directory))
		return
	model_directory = model_directory.rstrip("/")
	model_settings = os.path.basename(model_directory)

	assert options.snapshot_index > 0
	snapshot_index = options.snapshot_index

	# load the existing model
	model_snapshot_file_path = os.path.join(model_directory, "model-%d" % snapshot_index)
	if not os.path.exists(model_snapshot_file_path):
		sys.stderr.write("error: model snapshot file unfound %s...\n" % (model_snapshot_file_path))
		return

	dpgm = cPickle.load(open(model_snapshot_file_path, "rb"))
	print 'successfully load model snpashot %s...' % (os.path.join(model_directory, "model-%d" % snapshot_index))

	# set the resume options  
	matches = re.match(model_settings_pattern, model_settings)
	# training_iterations = int(matches.group('iteration'))
	training_iterations = options.training_iterations
	if options.snapshot_interval == -1:
		snapshot_interval = int(matches.group('snapshot'))
	else:
		snapshot_interval = options.snapshot_interval
	alpha_alpha = float(matches.group('alpha'))
	inference_mode = matches.group('mode')

	split_merge_heuristics = options.split_merge_heuristics
	split_proposal = options.split_proposal
	merge_proposal = options.merge_proposal

	now = datetime.datetime.now()
	suffix = now.strftime("%y%m%d-%H%M%S") + ""
	suffix += "-%s" % ("dpgm")
	suffix += "-I%d" % (training_iterations)
	suffix += "-S%d" % (snapshot_interval)
	suffix += "-aa%g" % (alpha_alpha)
	# suffix += "-ak%g" % (alpha_kappa)
	# suffix += "-an%g" % (alpha_nu)
	# suffix += "-SA%d" % (sampling_approach)
	if split_merge_heuristics == -2:
		suffix += inference_mode
	elif split_merge_heuristics >= -1 and split_merge_heuristics <= 3:
		dpgm.split_merge_heuristics = split_merge_heuristics
		dpgm.split_proposal = split_proposal
		dpgm.merge_proposal = merge_proposal
		if split_merge_heuristics >= 0:
			suffix += "-smh%d" % (split_merge_heuristics)
			if split_merge_heuristics >= 1:
				suffix += "-sp%d" % (split_proposal)
				suffix += "-mp%d" % (merge_proposal)
	else:
		sys.stderr.write("error: unrecognized split-merge heuristics %d...\n" % (split_merge_heuristics))
		return

	output_directory = options.output_directory
	output_directory = output_directory.rstrip("/")
	output_directory = os.path.join(output_directory, suffix)
	assert (not os.path.exists(os.path.abspath(output_directory)))
	os.mkdir(os.path.abspath(output_directory))

	shutil.copy(model_snapshot_file_path, os.path.join(output_directory, "model-" + str(snapshot_index)))
	shutil.copy(model_snapshot_file_path, os.path.join(output_directory, "label-" + str(snapshot_index)))
	shutil.copy(model_snapshot_file_path, os.path.join(output_directory, "mu-" + str(snapshot_index)))
	shutil.copy(model_snapshot_file_path, os.path.join(output_directory, "sigma-" + str(snapshot_index)))

	for iteration in xrange(snapshot_index, training_iterations):
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
