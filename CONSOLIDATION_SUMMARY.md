# PyDPGM Consolidation Summary

## Overview
This repository has been consolidated into a high-level script `dpgm.py` that provides a simple API for clustering using Dirichlet Process Gaussian Mixture Models (DPGM).

## Key Changes

### 1. New Consolidated API (`dpgm.py`)
Created a new high-level script that provides two main functions:

- **`fit_dpgm(data, ...)`**: Fits a DPGM model to data
  - Takes numpy array (N x M) as input
  - Returns dictionary with clustering results including:
    - `labels`: cluster assignments
    - `n_clusters`: number of clusters found
    - `cluster_means`: cluster centers
    - `cluster_counts`: points per cluster
    - `log_likelihood`: final log-likelihood
    - `model`: trained model object

- **`predict_dpgm(model, data)`**: Predicts cluster assignments for new data
  - Takes trained model and new data
  - Returns cluster assignments

### 2. Python 2 to Python 3 Compatibility (`monte_carlo.py`)
Fixed numerous compatibility issues:
- Converted `print` statements to `print()` function calls
- Replaced `xrange` with `range`
- Fixed numpy scalar extraction (`.item()` method)
- Updated scipy API (`scipy.misc.logsumexp` → `scipy.special.logsumexp`)

### 3. Documentation (`README.md`)
Updated README with:
- Quick start guide using the new API
- Code examples
- API reference
- Command-line usage

### 4. Example Script (`example_usage.py`)
Created comprehensive example demonstrating:
- Basic 2D clustering
- Prediction on new data
- Higher dimensional (5D) data clustering

## Usage Examples

### Basic Usage (Python)
```python
import numpy as np
from dpgm import fit_dpgm

# Your data: N x M array
data = np.random.randn(300, 2)

# Fit the model
results = fit_dpgm(data, alpha_alpha=1.0, training_iterations=100)

# Get results
print("Clusters found:", results['n_clusters'])
print("Cluster labels:", results['labels'])
```

### Command Line
```bash
# Run with synthetic example data
python dpgm.py

# Cluster data from a file
python dpgm.py data.dat
```

## Testing
All functionality has been tested with:
1. Synthetic 2D data (3 clusters) ✓
2. Real data from point-clusters.tar.gz ✓
3. Higher dimensional data (5D) ✓
4. Prediction on new data points ✓

## Files Modified/Created
- **Created**: `dpgm.py` - Main consolidated API
- **Created**: `example_usage.py` - Comprehensive examples
- **Modified**: `monte_carlo.py` - Python 3 compatibility
- **Modified**: `README.md` - Updated documentation

## Backward Compatibility
The original command-line interface via `launch_train.py` remains available and functional.

## Dependencies
- numpy
- scipy

Install with: `pip install numpy scipy`

## Notes
- The DPGM model automatically determines the number of clusters
- The `alpha_alpha` parameter controls the preference for more/fewer clusters
- Higher `alpha_alpha` values encourage more clusters
- The model uses collapsed Gibbs sampling for inference
