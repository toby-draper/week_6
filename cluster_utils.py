"""
Cluster Analysis Utilities Module

This module provides standalone functions for K-means clustering analysis,
extracted from cluster_example.py to enable reusable, modular code.

Functions:
    - load_voting_data: Load and prepare voting data
    - perform_kmeans: Run K-means clustering with specified parameters
    - calculate_elbow_method: Calculate WCSS for different k values
    - calculate_silhouette_scores: Calculate silhouette scores for different k values
    - calculate_variance_explained: Calculate total and between-cluster variance
    - visualize_clusters_3d: Create 3D scatter plot of clusters
    - visualize_clusters_with_party: Create 3D plot with cluster and party labels
    - visualize_elbow_curve: Plot elbow curve for optimal k selection
    - visualize_silhouette_scores: Plot silhouette scores
    - train_decision_tree: Train decision tree with optional cluster features
    - compare_models: Compare decision tree models with/without cluster features
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, silhouette_score


def load_voting_data(dem_filepath, rep_filepath=None, encoding='latin'):
    """
    Load voting data from CSV files.
    
    Parameters:
    -----------
    dem_filepath : str
        Path to the Democrat-introduced bills voting data CSV
    rep_filepath : str, optional
        Path to the Republican-introduced bills voting data CSV
    encoding : str, default='latin'
        Encoding to use for reading the CSV file
    
    Returns:
    --------
    house_votes_Dem : pd.DataFrame
        DataFrame containing Democrat bills voting data
    house_votes_Rep : pd.DataFrame or None
        DataFrame containing Republican bills voting data, or None if not provided
    """
    house_votes_Dem = pd.read_csv(dem_filepath, encoding=encoding)
    house_votes_Rep = pd.read_csv(rep_filepath) if rep_filepath else None
    
    return house_votes_Dem, house_votes_Rep


def perform_kmeans(data, n_clusters=2, random_state=1, columns=None):
    """
    Perform K-means clustering on the specified data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data for clustering
    n_clusters : int, default=2
        Number of clusters to form
    random_state : int, default=1
        Random state for reproducibility
    columns : list, optional
        List of column names to use for clustering. If None, uses all numeric columns
    
    Returns:
    --------
    kmeans_obj : KMeans
        Fitted KMeans object
    cluster_data : pd.DataFrame
        Data used for clustering
    """
    if columns is None:
        # Use all numeric columns
        cluster_data = data.select_dtypes(include=[np.number])
    else:
        cluster_data = data[columns]
    
    kmeans_obj = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans_obj.fit(cluster_data)
    
    return kmeans_obj, cluster_data


def calculate_elbow_method(data, max_clusters=10, random_state=1):
    """
    Calculate within-cluster sum of squares (WCSS) for different k values.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to cluster
    max_clusters : int, default=10
        Maximum number of clusters to test
    random_state : int, default=1
        Random state for reproducibility
    
    Returns:
    --------
    wcss : list
        List of WCSS values for k from 1 to max_clusters
    k_values : list
        List of k values tested
    """
    wcss = []
    k_values = list(range(1, max_clusters + 1))
    
    for k in k_values:
        kmeans_obj = KMeans(n_clusters=k, random_state=random_state)
        kmeans_obj.fit(data)
        wcss.append(kmeans_obj.inertia_)
    
    return wcss, k_values


def calculate_silhouette_scores(data, min_clusters=2, max_clusters=10, random_state=1):
    """
    Calculate silhouette scores for different k values.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to cluster
    min_clusters : int, default=2
        Minimum number of clusters to test (must be at least 2)
    max_clusters : int, default=10
        Maximum number of clusters to test
    random_state : int, default=1
        Random state for reproducibility
    
    Returns:
    --------
    silhouette_scores : list
        List of silhouette scores
    k_values : list
        List of k values tested
    best_k : int
        Optimal number of clusters based on highest silhouette score
    """
    silhouette_scores = []
    k_values = list(range(min_clusters, max_clusters + 1))
    
    for k in k_values:
        kmeans_obj = KMeans(n_clusters=k, algorithm="lloyd", random_state=random_state)
        kmeans_obj.fit(data)
        score = silhouette_score(data, kmeans_obj.labels_)
        silhouette_scores.append(score)
    
    best_k = k_values[silhouette_scores.index(max(silhouette_scores))]
    
    return silhouette_scores, k_values, best_k


def calculate_variance_explained(data, kmeans_obj):
    """
    Calculate variance explained by clustering.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data that was clustered
    kmeans_obj : KMeans
        Fitted KMeans object
    
    Returns:
    --------
    variance_explained : float
        Proportion of variance explained (between 0 and 1)
    total_ss : float
        Total sum of squares
    between_ss : float
        Between-cluster sum of squares
    within_ss : float
        Within-cluster sum of squares (inertia)
    """
    # Calculate total sum of squares
    total_sum_squares = np.sum((data - np.mean(data))**2)
    total_ss = np.sum(total_sum_squares)
    
    # Within-cluster sum of squares
    within_ss = kmeans_obj.inertia_
    
    # Between-cluster sum of squares
    between_ss = total_ss - within_ss
    
    # Variance explained
    variance_explained = between_ss / total_ss
    
    return variance_explained, total_ss, between_ss, within_ss


def visualize_clusters_3d(data, labels, x_col, y_col, z_col, title=None, show=True):
    """
    Create interactive 3D scatter plot showing clusters.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to visualize
    labels : array-like
        Cluster labels for each data point
    x_col : str
        Column name for x-axis
    y_col : str
        Column name for y-axis
    z_col : str
        Column name for z-axis
    title : str, optional
        Plot title
    show : bool, default=True
        Whether to display the plot
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure object
    """
    if title is None:
        title = f"{x_col} vs. {y_col} vs. {z_col}"
    
    fig = px.scatter_3d(
        data, x=x_col, y=y_col, z=z_col,
        color=labels,
        title=title
    )
    
    if show:
        fig.show(renderer="browser")
    
    return fig


def visualize_clusters_with_party(data, cluster_labels, party_col, x_col, y_col, z_col, 
                                   cluster_centers=None, title=None, show=True):
    """
    Create 3D plot showing both cluster assignments and party labels.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to visualize
    cluster_labels : array-like
        Cluster labels for each data point
    party_col : str
        Column name for party labels
    x_col : str
        Column name for x-axis
    y_col : str
        Column name for y-axis
    z_col : str
        Column name for z-axis
    cluster_centers : np.ndarray, optional
        Cluster centers to display
    title : str, optional
        Plot title
    show : bool, default=True
        Whether to display the plot
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure object
    """
    if title is None:
        title = f"{x_col} vs. {y_col} vs. {z_col}"
    
    fig = px.scatter_3d(
        data, x=x_col, y=y_col, z=z_col,
        color=party_col, symbol=cluster_labels,
        title=title
    )
    
    # Add cluster centers if provided
    if cluster_centers is not None:
        fig.add_trace(go.Scatter3d(
            x=cluster_centers[:, 0],
            y=cluster_centers[:, 1],
            z=cluster_centers[:, 2],
            mode="markers",
            marker=dict(size=20, color="black"),
            name="Cluster Centers"
        ))
    
    if show:
        fig.show(renderer="browser")
    
    return fig


def visualize_elbow_curve(wcss, k_values, title="Elbow Method", show=True):
    """
    Plot the elbow curve for optimal k selection.
    
    Parameters:
    -----------
    wcss : list
        List of within-cluster sum of squares values
    k_values : list
        List of k values
    title : str, default="Elbow Method"
        Plot title
    show : bool, default=True
        Whether to display the plot
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure object
    """
    elbow_data = pd.DataFrame({"k": k_values, "wcss": wcss})
    fig = px.line(elbow_data, x="k", y="wcss", title=title)
    
    if show:
        fig.show()
    
    return fig


def visualize_silhouette_scores(silhouette_scores, k_values, 
                                title="Silhouette Score by Number of Clusters", 
                                show=True):
    """
    Plot silhouette scores across different k values.
    
    Parameters:
    -----------
    silhouette_scores : list
        List of silhouette scores
    k_values : list
        List of k values
    title : str, default="Silhouette Score by Number of Clusters"
        Plot title
    show : bool, default=True
        Whether to display the plot
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure object
    """
    fig = go.Figure(data=go.Scatter(
        x=k_values,
        y=silhouette_scores,
        mode='lines+markers'
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Silhouette Score"
    )
    
    if show:
        fig.show()
    
    return fig


def train_decision_tree(data, target_col, feature_cols=None, test_size=0.3, 
                       tune_size=0.5, random_state=1):
    """
    Train a decision tree classifier.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    target_col : str
        Name of the target column
    feature_cols : list, optional
        List of feature column names. If None, uses all columns except target
    test_size : float, default=0.3
        Proportion of data to reserve for tune+test
    tune_size : float, default=0.5
        Proportion of tune+test data to use for tuning
    random_state : int, default=1
        Random state for reproducibility
    
    Returns:
    --------
    model : DecisionTreeClassifier
        Trained decision tree model
    train : pd.DataFrame
        Training data
    tune : pd.DataFrame
        Tuning data
    test : pd.DataFrame
        Test data
    """
    # Split data
    train, tune_and_test = train_test_split(data, test_size=test_size, random_state=random_state)
    tune, test = train_test_split(tune_and_test, test_size=tune_size, random_state=random_state)
    
    # Prepare features and target
    if feature_cols is None:
        features = train.drop(columns=[target_col])
    else:
        features = train[feature_cols]
    
    target = train[target_col]
    
    # Train model
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(features, target)
    
    return model, train, tune, test


def compare_models(data, target_col, cluster_col, drop_cols=None, 
                  test_size=0.3, tune_size=0.5, random_state=1):
    """
    Compare decision tree models with and without cluster features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    target_col : str
        Name of the target column
    cluster_col : str
        Name of the cluster column
    drop_cols : list, optional
        Additional columns to drop (e.g., 'Last.Name')
    test_size : float, default=0.3
        Proportion of data to reserve for tune+test
    tune_size : float, default=0.5
        Proportion of tune+test data to use for tuning
    random_state : int, default=1
        Random state for reproducibility
    
    Returns:
    --------
    results : dict
        Dictionary containing models, predictions, and confusion matrices
    """
    # Prepare data
    if drop_cols is None:
        drop_cols = []
    
    tree_data = data.drop(columns=drop_cols)
    
    # Convert categorical variables to category dtype if not already numeric
    for col in [target_col, cluster_col]:
        if col in tree_data.columns and tree_data[col].dtype == 'object':
            tree_data[col] = tree_data[col].astype('category')
    
    # Train model WITH clusters
    model_with, train_with, tune_with, test_with = train_decision_tree(
        tree_data, target_col, test_size=test_size, 
        tune_size=tune_size, random_state=random_state
    )
    
    # Predict and evaluate
    features_tune_with = tune_with.drop(columns=[target_col])
    predictions_with = model_with.predict(features_tune_with)
    cm_with = confusion_matrix(tune_with[target_col], predictions_with)
    
    # Train model WITHOUT clusters
    tree_data_nc = tree_data.drop(columns=[cluster_col])
    model_without, train_without, tune_without, test_without = train_decision_tree(
        tree_data_nc, target_col, test_size=test_size, 
        tune_size=tune_size, random_state=random_state
    )
    
    # Predict and evaluate
    features_tune_without = tune_without.drop(columns=[target_col])
    predictions_without = model_without.predict(features_tune_without)
    cm_without = confusion_matrix(tune_without[target_col], predictions_without)
    
    results = {
        'model_with_clusters': model_with,
        'model_without_clusters': model_without,
        'confusion_matrix_with': cm_with,
        'confusion_matrix_without': cm_without,
        'predictions_with': predictions_with,
        'predictions_without': predictions_without,
        'tune_with': tune_with,
        'tune_without': tune_without
    }
    
    return results
