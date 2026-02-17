#!/usr/bin/env python3
"""
Example usage of cluster_utils module

This script demonstrates how to use cluster_utils.py as a Python package
for K-means clustering analysis on voting data.
"""

import cluster_utils as cu

# 1. Load voting data
print("Loading voting data...")
house_votes_Dem, house_votes_Rep = cu.load_voting_data(
    'house_votes_Dem.csv',
    'house_votes_Rep.csv'
)
print(f"Loaded {len(house_votes_Dem)} Democrat voting records")

# 2. Perform initial K-means clustering
print("\nPerforming K-means clustering with k=2...")
kmeans_obj, cluster_data = cu.perform_kmeans(
    house_votes_Dem,
    n_clusters=2,
    columns=["aye", "nay", "other"]
)
print(f"Cluster centers:\n{kmeans_obj.cluster_centers_}")
print(f"Inertia (WCSS): {kmeans_obj.inertia_:.2f}")

# 3. Find optimal k using elbow method
print("\nCalculating elbow method for k=1 to k=10...")
wcss, k_values = cu.calculate_elbow_method(cluster_data, max_clusters=10)
print(f"WCSS values: {[f'{w:.0f}' for w in wcss]}")

# 4. Find optimal k using silhouette scores
print("\nCalculating silhouette scores...")
silhouette_scores, k_vals, best_k = cu.calculate_silhouette_scores(
    cluster_data, min_clusters=2, max_clusters=10
)
print(f"Best k by silhouette score: {best_k}")
print(f"Best silhouette score: {max(silhouette_scores):.4f}")

# 5. Perform clustering with optimal k
print(f"\nPerforming K-means clustering with k={best_k}...")
kmeans_final, _ = cu.perform_kmeans(
    house_votes_Dem,
    n_clusters=best_k,
    columns=["aye", "nay", "other"]
)

# 6. Calculate variance explained
var_exp, total_ss, between_ss, within_ss = cu.calculate_variance_explained(
    cluster_data, kmeans_final
)
print(f"Variance explained: {var_exp:.4f} ({var_exp*100:.2f}%)")

# 7. Visualize clusters (optional - set show=False to skip browser display)
print("\nCreating visualizations (set show=False to skip)...")
# Uncomment to show visualizations:
# cu.visualize_clusters_3d(
#     house_votes_Dem, kmeans_final.labels_,
#     "aye", "nay", "other",
#     title="Voting Clusters"
# )
# cu.visualize_elbow_curve(wcss, k_values)
# cu.visualize_silhouette_scores(silhouette_scores, k_vals)

# 8. Add cluster labels and compare models
print("\nTraining decision trees with and without cluster features...")
house_votes_Dem['clusters'] = kmeans_final.labels_
results = cu.compare_models(
    house_votes_Dem,
    target_col='party.labels',
    cluster_col='clusters',
    drop_cols=['Last.Name']
)

print("\nConfusion Matrix WITH clusters:")
print(results['confusion_matrix_with'])
print("\nConfusion Matrix WITHOUT clusters:")
print(results['confusion_matrix_without'])

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)
