# Standalone dpgm.py Usage Guide

## Overview

The file `dpgm.py` is now a **completely standalone script** with no dependencies on other files in this repository. You can copy it to any other project and use it independently.

## Dependencies

The only external dependencies are standard scientific Python libraries:
- `numpy` (for numerical operations)
- `scipy` (for special functions)
- Standard library: `os`, `random`, `sys`

Install dependencies:
```bash
pip install numpy scipy
```

## Features

The standalone `dpgm.py` includes:
1. **MonteCarlo class** - The core DPGM implementation (inlined from monte_carlo.py)
2. **fit_dpgm()** - High-level API for fitting DPGM models
3. **predict_dpgm()** - Prediction API for new data points
4. **Example usage** - Built-in examples when run as `__main__`

## Quick Start

### As a Python Module

```python
import numpy as np
import dpgm

# Create some data
data = np.random.randn(100, 2)

# Fit DPGM model
results = dpgm.fit_dpgm(
    data,
    alpha_alpha=1.0,           # Concentration parameter
    training_iterations=100,    # Number of iterations
    verbose=True
)

# Access results
print("Number of clusters:", results['n_clusters'])
print("Cluster labels:", results['labels'])
print("Cluster means:", results['cluster_means'])
print("Cluster counts:", results['cluster_counts'])

# Predict on new data
new_data = np.random.randn(10, 2)
predictions = dpgm.predict_dpgm(results, new_data)
```

### Running Examples

Run the built-in examples:
```bash
# Example 1: Synthetic data
python dpgm.py

# Example 2: Load data from file
python dpgm.py your_data_file.dat
```

## API Reference

### fit_dpgm(data, ...)

Fit a Dirichlet Process Gaussian Mixture Model.

**Parameters:**
- `data` (numpy.ndarray): N x M array where N is number of points, M is number of features
- `alpha_alpha` (float, default=1.0): Concentration parameter (higher = more clusters)
- `training_iterations` (int, default=100): Number of Gibbs sampling iterations
- `split_merge_heuristics` (int, default=-1): Split-merge strategy
  - -1: no split-merge
  - 0: component resampling
  - 1, 2, 3: various candidate selection strategies
- `split_proposal` (int, default=0): Split proposal method (0, 1, or 2)
- `merge_proposal` (int, default=0): Merge proposal method (0, 1, or 2)
- `verbose` (bool, default=True): Print progress information

**Returns:**
Dictionary containing:
- `labels`: Cluster assignments (N,)
- `n_clusters`: Number of clusters found
- `cluster_means`: Cluster centers (K x M)
- `cluster_counts`: Points per cluster (K,)
- `log_likelihood`: Final log-likelihood
- `model`: Trained MonteCarlo model

### predict_dpgm(model, data)

Predict cluster assignments for new data.

**Parameters:**
- `model`: Either a MonteCarlo model or results dict from fit_dpgm()
- `data` (numpy.ndarray): N x M array of new data points

**Returns:**
- numpy.ndarray: Cluster assignments for new points

## File Structure

```
dpgm.py (1604 lines, ~56KB)
├── Imports (numpy, scipy, os, random, sys)
├── MonteCarlo class (core DPGM implementation)
│   ├── Initialization
│   ├── Gibbs sampling
│   ├── Split-merge operations
│   ├── Inference
│   └── Helper methods
├── fit_dpgm() - High-level fitting API
├── predict_dpgm() - Prediction API
└── __main__ - Example usage code
```

## Copying to Another Repository

Simply copy the `dpgm.py` file:

```bash
# Copy to another repo
cp dpgm.py /path/to/your/other/repo/

# Or download directly from GitHub
wget https://raw.githubusercontent.com/kzhai/PyDPGM/main/dpgm.py
```

No other files from this repository are needed!

## Technical Details

- **Algorithm**: Collapsed Gibbs sampling for Dirichlet Process Gaussian Mixture Models
- **Original Author**: Ke Zhai (zhaike@cs.umd.edu)
- **Lines of Code**: 1604 lines
- **Local Dependencies**: None (all code is self-contained)

## License

See the main repository README for license information.
