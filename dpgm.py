"""
Consolidated Dirichlet Process Gaussian Mixture Model (DPGM) script
Author: Consolidated from PyDPGM repository

This script provides a simple high-level API for clustering data using DPGM.
"""

import numpy
import monte_carlo


def fit_dpgm(data, 
             alpha_alpha=1.0,
             training_iterations=100,
             split_merge_heuristics=-1,
             split_proposal=0,
             merge_proposal=0,
             verbose=True):
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
    verbose : bool, optional (default=True)
        Whether to print progress information
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'labels': numpy array of cluster assignments (N,)
        - 'n_clusters': number of clusters found
        - 'cluster_means': cluster centers (K x M)
        - 'cluster_counts': number of points in each cluster (K,)
        - 'log_likelihood': final log-likelihood value
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
    dpgm = monte_carlo.MonteCarlo(
        split_merge_heuristics=split_merge_heuristics,
        split_proposal=split_proposal,
        merge_proposal=merge_proposal
    )
    
    dpgm._initialize(data, alpha_alpha=alpha_alpha)
    
    # Run training iterations
    log_likelihood = None
    for iteration in range(training_iterations):
        log_likelihood = dpgm.learning()
        
        if verbose and (iteration + 1) % max(1, training_iterations // 10) == 0:
            print("Iteration {}/{}: {} clusters, log-likelihood = {:.4f}".format(
                iteration + 1, training_iterations, dpgm._K, log_likelihood
            ))
    
    if verbose:
        print("=" * 60)
        print("Training completed!")
        print("Final number of clusters: {}".format(dpgm._K))
        print("Final log-likelihood: {:.4f}".format(log_likelihood))
        print("=" * 60)
    
    # Extract results
    results = {
        'labels': dpgm._label.copy(),
        'n_clusters': dpgm._K,
        'cluster_means': dpgm._mu.copy(),
        'cluster_counts': dpgm._count.copy(),
        'log_likelihood': log_likelihood,
        'model': dpgm
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
        if 'model' not in model:
            raise ValueError("Dictionary must contain 'model' key")
        model = model['model']
    
    # Validate input
    if not isinstance(data, numpy.ndarray):
        data = numpy.array(data)
    
    if len(data.shape) != 2:
        raise ValueError("Data must be a 2D array (N x M)")
    
    # Use the inference method to predict labels
    predictions = model.inference(data)
    
    return predictions


if __name__ == '__main__':
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
    results = fit_dpgm(
        synthetic_data,
        alpha_alpha=1.0,
        training_iterations=50,
        verbose=True
    )
    
    print("\nClustering results:")
    print("  Number of clusters found: {}".format(results['n_clusters']))
    print("  Cluster counts: {}".format(results['cluster_counts']))
    print("  Cluster means:\n{}".format(results['cluster_means']))
    
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
            file_results = fit_dpgm(
                file_data,
                alpha_alpha=1.0,
                training_iterations=100,
                verbose=True
            )
            
            print("\nClustering results:")
            print("  Number of clusters found: {}".format(file_results['n_clusters']))
            print("  Cluster counts: {}".format(file_results['cluster_counts']))
            
            # Save results
            output_labels = data_file.replace('.dat', '_labels.dat')
            numpy.savetxt(output_labels, file_results['labels'], fmt='%d')
            print("\nCluster labels saved to: {}".format(output_labels))
            
        except Exception as e:
            print("Error loading or processing file: {}".format(e))
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
