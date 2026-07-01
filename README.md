PyDPGM
==========

PyDPGM is a Dirichlet Process Gaussian Mixture package, please download the latest version from our [GitHub repository](https://github.com/kzhai/PyDPGM).

Please send any bugs of problems to Ke Zhai (kzhai@umd.edu).

Install and Build
----------

This package depends on many external python libraries, such as numpy, scipy and nltk.

Launch and Execute
----------

Assume the PyDPGM package is downloaded under directory ```$PROJECT_SPACE/src/```, i.e.,

	$PROJECT_SPACE/src/PyDPGM

To prepare the example dataset,

	tar zxvf point-clusters.tar.gz

To launch PyDPGM, first redirect to the directory of PyDPGM source code,

	cd $PROJECT_SPACE/src/PyDPGM

and run the following command on example dataset,

	python -m launch_train --input_directory=./point-clusters --output_directory=./ --training_iterations=100

The generic argument to run PyDPGM is

	python -m launch_train --input_directory=$INPUT_DIRECTORY/$DATASET_NAME --output_directory=$OUTPUT_DIRECTORY --training_iterations=$NUMBER_OF_ITERATIONS

You should be able to find the output at directory ```$OUTPUT_DIRECTORY/$DATASET_NAME```.

Under any circumstances, you may also get help information and usage hints by running the following command

	python -m launch_train --help

Additional Scripts
----------

To generate synthetic data

	python generate_data_crp.py --output_directory=./ --number_of_clusters=5 --number_of_points=1000
	#python generate_data_sbp.py --output_directory=./ --number_of_clusters=5 --number_of_points=1000

To plot result

	python plot_clusters.py --input_directory=./point-clusters/ --output_directory=./point-clusters/
