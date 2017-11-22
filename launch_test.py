import cPickle
import optparse
import os
import re
import sys
import time

import numpy

model_settings_pattern = re.compile(
	'\d+-\d+-dpgm-I(?P<iteration>\d+)-S(?P<snapshot>\d+)-aa(?P<alpha>[\d\.]+)(-smh(?P<smh>[\d]+))?(-sp(?P<sp>[\d]+)-mp(?P<mp>[\d]+))?')


def parse_args():
	parser = optparse.OptionParser()
	parser.set_defaults(  # parameter set 1
		input_file=None,
		model_directory=None,
		snapshot_index=-1,
		# training_iterations=1000
	)
	# parameter set 1
	parser.add_option("--input_file", type="string", dest="input_file",
	                  help="input directory [None]")
	parser.add_option("--model_directory", type="string", dest="model_directory",
	                  help="model directory [None]")
	parser.add_option("--snapshot_index", type="int", dest="snapshot_index",
	                  help="snapshot index [-1]")
	# parser.add_option("--training_iterations", type="int", dest="training_iterations",
	# help="number of training iterations [1000]")

	(options, args) = parser.parse_args()
	return options


def main():
	options = parse_args()

	# parameter set 1
	assert (options.input_file != None)
	input_file = options.input_file
	data_prime = numpy.loadtxt(input_file)

	assert (options.model_directory != None)
	model_directory = options.model_directory

	if not os.path.exists(model_directory):
		sys.stderr.write("model directory %s not exists...\n" % (model_directory))
		return
	model_directory = model_directory.rstrip("/")
	model_settings = os.path.basename(model_directory)

	if options.snapshot_index > 0:
		snapshot_indices = [options.snapshot_index]
	else:
		matches = re.match(model_settings_pattern, model_settings)
		training_iterations = int(matches.group('iteration'))
		snapshot_interval = int(matches.group('snapshot'))
		snapshot_indices = xrange(0, training_iterations + 1, snapshot_interval)

	for snapshot_index in snapshot_indices:
		model_snapshot_file_path = os.path.join(model_directory, "model-%d" % snapshot_index)
		if not os.path.exists(model_snapshot_file_path):
			continue

		dpgmm_inferencer = cPickle.load(open(model_snapshot_file_path, "rb"))
		print 'successfully load model snpashot %s...' % (os.path.join(model_directory, "model-%d" % snapshot_index))

		clock = time.time()
		label_prime, log_likelihood_prime = dpgmm_inferencer.inference(data_prime)
		clock = time.time() - clock
		print 'testing snapshot %d finished in %f seconds: number-of-topics = %d, log-likelihood = %f' % (
			snapshot_index, clock, dpgmm_inferencer._K, log_likelihood_prime)

	# if (dpgm._iteration_counter % snapshot_interval == 0):
	# dpgm.export_beta(os.path.join(output_directory, 'exp_beta-' + str(dpgm._iteration_counter)), 50)
	# model_snapshot_path = os.path.join(output_directory, 'model-' + str(dpgm._iteration_counter))
	# cPickle.dump(dpgm, open(model_snapshot_path, 'wb'))


if __name__ == '__main__':
	main()
