# Cluster Utils - K-means Clustering Module

This module provides reusable functions for K-means clustering analysis, extracted from the `cluster_example.py` script to enable modular, package-like usage.

## Installation

Ensure you have the required dependencies installed:

```bash
pip install pandas numpy scikit-learn plotly
```

## Quick Start

```python
import cluster_utils as cu

# Load data
data_dem, data_rep = cu.load_voting_data('house_votes_Dem.csv', 'house_votes_Rep.csv')

# Perform clustering
kmeans_obj, cluster_data = cu.perform_kmeans(data_dem, n_clusters=2, columns=["aye", "nay", "other"])

# Find optimal k using elbow method
wcss, k_values = cu.calculate_elbow_method(cluster_data, max_clusters=10)

# Find optimal k using silhouette scores
scores, k_vals, best_k = cu.calculate_silhouette_scores(cluster_data)
```

See `example_usage.py` for a complete working example.

## Available Functions

### Data Loading
- **`load_voting_data(dem_filepath, rep_filepath=None, encoding='latin')`** - Load voting data from CSV files

### Clustering
- **`perform_kmeans(data, n_clusters, random_state, columns)`** - Perform K-means clustering
- **`calculate_elbow_method(data, max_clusters, random_state)`** - Calculate WCSS for different k values
- **`calculate_silhouette_scores(data, min_clusters, max_clusters, random_state)`** - Calculate silhouette scores

### Analysis
- **`calculate_variance_explained(data, kmeans_obj)`** - Calculate variance explained by clustering

### Visualization
- **`visualize_clusters_3d(data, labels, x_col, y_col, z_col, title, show)`** - 3D scatter plot of clusters
- **`visualize_clusters_with_party(data, cluster_labels, party_col, ...)`** - 3D plot with cluster and party labels
- **`visualize_elbow_curve(wcss, k_values, title, show)`** - Plot elbow curve
- **`visualize_silhouette_scores(silhouette_scores, k_values, title, show)`** - Plot silhouette scores

### Machine Learning
- **`train_decision_tree(data, target_col, feature_cols, test_size, tune_size, random_state)`** - Train decision tree
- **`compare_models(data, target_col, cluster_col, drop_cols, ...)`** - Compare models with/without cluster features

## Function Documentation

Each function includes detailed docstrings with:
- Parameter descriptions
- Return value descriptions
- Usage examples

Access documentation in Python:
```python
help(cu.perform_kmeans)
```

## Example Workflow

1. **Load data**: Use `load_voting_data()` to read CSV files
2. **Initial clustering**: Use `perform_kmeans()` with a starting k value
3. **Find optimal k**: Use `calculate_elbow_method()` or `calculate_silhouette_scores()`
4. **Re-cluster**: Use `perform_kmeans()` with optimal k
5. **Evaluate**: Use `calculate_variance_explained()` to assess clustering quality
6. **Visualize**: Use visualization functions to explore results
7. **Apply**: Use `compare_models()` to see if clusters improve predictions

## Comparison with Original Script

The original `cluster_example.py` contains all code in one file with inline execution. This new `cluster_utils.py` module:

- ✅ Separates functionality into reusable functions
- ✅ Can be imported like a Python package
- ✅ Includes comprehensive docstrings
- ✅ Allows users to call specific functions as needed
- ✅ Supports customization through function parameters
- ✅ Maintains all original functionality

The original file remains unchanged for reference and educational purposes.
