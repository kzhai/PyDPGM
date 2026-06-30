"""
Simple example demonstrating the consolidated DPGM API
"""

import numpy as np
from dpgm import fit_dpgm, predict_dpgm

# Example 1: Simple clustering
print("=" * 60)
print("Example 1: Basic Clustering")
print("=" * 60)

# Generate synthetic data with 3 clusters
np.random.seed(42)
cluster1 = np.random.randn(50, 2) * 0.5 + np.array([0, 0])
cluster2 = np.random.randn(50, 2) * 0.5 + np.array([5, 5])
cluster3 = np.random.randn(50, 2) * 0.5 + np.array([5, 0])
data = np.vstack([cluster1, cluster2, cluster3])

# Shuffle
indices = np.random.permutation(len(data))
data = data[indices]

# Fit the model
results = fit_dpgm(
    data,
    alpha_alpha=1.0,
    training_iterations=50,
    verbose=False  # Set to True to see progress
)

print("\nResults:")
print(f"  Number of clusters found: {results['n_clusters']}")
print(f"  Cluster counts: {results['cluster_counts']}")
print(f"  Final log-likelihood: {results['log_likelihood']:.4f}")

# Example 2: Prediction on new data
print("\n" + "=" * 60)
print("Example 2: Predicting on New Data")
print("=" * 60)

# Generate new test points
test_data = np.array([
    [0.1, 0.1],   # Should be near cluster 1
    [5.1, 5.1],   # Should be near cluster 2
    [5.1, 0.1],   # Should be near cluster 3
])

# Predict cluster assignments
predictions = predict_dpgm(results, test_data)

print("\nTest points and predictions:")
for i, (point, pred) in enumerate(zip(test_data, predictions)):
    print(f"  Point {i+1}: {point} -> Cluster {pred}")

# Example 3: Higher dimensional data
print("\n" + "=" * 60)
print("Example 3: Higher Dimensional Data")
print("=" * 60)

# Generate 5-dimensional data
np.random.seed(123)
high_dim_data = np.random.randn(200, 5)

# Add some structure
high_dim_data[:100] += np.array([2, 0, 0, 0, 0])
high_dim_data[100:] += np.array([0, 0, 0, 0, 2])

results_5d = fit_dpgm(
    high_dim_data,
    alpha_alpha=1.0,
    training_iterations=30,
    verbose=False
)

print(f"\n5D Data Results:")
print(f"  Data shape: {high_dim_data.shape}")
print(f"  Number of clusters found: {results_5d['n_clusters']}")
print(f"  Cluster counts: {results_5d['cluster_counts']}")

print("\n" + "=" * 60)
print("All examples completed successfully!")
print("=" * 60)
