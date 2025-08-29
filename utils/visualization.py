import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import torch
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import warnings 
import os
import yaml

def plot_tool_performance(metrics: Dict, save_path: str = None):
    """Plot tool performance metrics."""
    tool_names = ['oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither']
    metric_names = ['tpr', 'fpr', 'accuracy', 'precision', 'recall']
    
    # Extract MAE values
    mae_values = np.zeros((len(tool_names), len(metric_names)))
    
    for i, tool in enumerate(tool_names):
        for j, metric in enumerate(metric_names):
            key = f'{tool}_{metric}_mae'
            if key in metrics:
                mae_values[i, j] = metrics[key]
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(mae_values, annot=True, fmt='.3f', 
                    xticklabels=metric_names, yticklabels=tool_names,
                    cmap='viridis_r')
    plt.title('Tool Performance Prediction - Mean Absolute Error', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_attention_weights(attention_weights: Dict, save_dir: str = None):
    """Plot attention weights for interpretation."""
    if 'cross_attention' not in attention_weights:
        return
    
    weights = attention_weights['cross_attention']
    
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()
    
    # Handle multi-head attention if necessary
    if len(weights.shape) == 4:  # [batch, head, seq_len, seq_len]
        num_heads = weights.shape[1]
        
        # Create directory if needed
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
        
        # Plot each attention head
        for head in range(num_heads):
            head_weights = weights[0, head]  # First batch item
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(head_weights, cmap='viridis')
            plt.title(f'Attention Weights - Head {head}')
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(save_dir / f'attention_head_{head}.png')
                plt.close()
            else:
                plt.show()
    else:
        # Plot single attention matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(weights[0], cmap='viridis')  # First batch item
        plt.title('Attention Weights')
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / 'attention.png'
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

def plot_prediction_distribution(predictions: torch.Tensor, targets: torch.Tensor, save_path: str = None):
    """Plot distribution of predictions vs targets."""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Reshape to (samples, tools, metrics)
    pred_shape = predictions.shape
    target_shape = targets.shape
    
    if len(pred_shape) == 2 and pred_shape[1] % 5 == 0:
        num_tools = pred_shape[1] // 5
        predictions = predictions.reshape(-1, num_tools, 5)
        targets = targets.reshape(-1, num_tools, 5)
    
    # Flatten for overall distribution
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    plt.figure(figsize=(12, 10))
    
    # Overall distribution
    plt.subplot(2, 1, 1)
    plt.hist(target_flat, bins=20, alpha=0.5, label='Targets')
    plt.hist(pred_flat, bins=20, alpha=0.5, label='Predictions')
    plt.title('Distribution of Targets vs Predictions')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Scatter plot
    plt.subplot(2, 1, 2)
    plt.scatter(target_flat, pred_flat, alpha=0.3)
    plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line for perfect predictions
    plt.title('Predictions vs Targets')
    plt.xlabel('Target')
    plt.ylabel('Prediction')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_per_tool_metrics(metrics: Dict, save_dir: str = None):
    """Plot metrics for each tool separately."""
    tool_names = ['oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither']
    metric_names = ['tpr', 'fpr', 'accuracy', 'precision', 'recall']
    
    # Create directory if needed
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract MAE values for each tool
    for tool in tool_names:
        mae_values = []
        for metric in metric_names:
            key = f'{tool}_{metric}_mae'
            if key in metrics:
                mae_values.append(metrics[key])
            else:
                mae_values.append(0)
        
        plt.figure(figsize=(10, 6))
        plt.bar(metric_names, mae_values)
        plt.title(f'{tool.title()} - Metric Prediction Errors (MAE)')
        plt.ylabel('Mean Absolute Error')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(save_dir / f'{tool}_metrics.png')
            plt.close()
        else:
            plt.show()
    
    # Overall comparison across tools
    overall_mae = []
    for tool in tool_names:
        key = f'{tool}_mae'
        if key in metrics:
            overall_mae.append(metrics[key])
        else:
            overall_mae.append(0)
    
    plt.figure(figsize=(10, 6))
    plt.bar(tool_names, overall_mae)
    plt.title('Overall Performance by Tool (MAE)')
    plt.ylabel('Mean Absolute Error')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(save_dir / 'overall_tool_comparison.png')
        plt.close()
    else:
        plt.show()
        
        
def plot_comprehensive_metrics(metrics_dict, save_path=None, title="Tool Performance Metrics (MAE)"):
    """
    Plot a heatmap summarizing MAE (or any other scalar error metric) 
    per tool and performance type (TPR, FPR, Accuracy, etc.).

    Parameters:
    - metrics_dict (dict): Expected structure:
        {
            'ToolName1': {'TPR': 0.1, 'FPR': 0.2, 'Accuracy': 0.05, 'Precision': 0.08, 'Recall': 0.09},
            'ToolName2': {...},
            ...
        }
    - save_path (str or None): If specified, saves the plot to this path
    - title (str): Title of the plot
    """
    if not metrics_dict:
        print("[WARN] plot_comprehensive_metrics(): Empty metrics_dict provided.")
        return

    # Convert dict of dicts to matrix
    tools = sorted(metrics_dict.keys())
    metrics = sorted(next(iter(metrics_dict.values())).keys())  # Assume all tools have same metrics
    data = [[metrics_dict[tool][metric] for metric in metrics] for tool in tools]

    plt.figure(figsize=(12, 6))
    sns.set(font_scale=1.1)
    ax = sns.heatmap(data, annot=True, fmt=".3f", xticklabels=metrics, yticklabels=tools, cmap="YlGnBu")

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Performance Metric", fontsize=12)
    ax.set_ylabel("Tool Name", fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Saved metric heatmap to: {save_path}")
    else:
        plt.show()
        
        
# Robust embedding visualization functions
def robust_cluster_embeddings(embeddings_np: np.ndarray, min_clusters: int = 2, max_clusters: int = 10) -> Tuple[np.ndarray, int, List[float]]:
    """
    Perform robust clustering on embeddings with error handling.
    
    Args:
        embeddings_np: Numpy array of embeddings
        min_clusters: Minimum number of clusters to try
        max_clusters: Maximum number of clusters to try
        
    Returns:
        Tuple of (cluster_labels, optimal_clusters, silhouette_scores)
    """
    from sklearn.cluster import KMeans
    
    # Ensure we don't try more clusters than samples
    n_samples = embeddings_np.shape[0]
    max_clusters = min(max_clusters, n_samples - 1) if n_samples > 2 else 2
    min_clusters = min(min_clusters, max_clusters)
    
    if n_samples < 2:
        # Too few samples for clustering
        return np.zeros(n_samples, dtype=int), 1, []
    
    # Try to find optimal number of clusters
    silhouette_scores = []
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress warnings
        
        if min_clusters <= max_clusters:
            for n_clusters in range(min_clusters, max_clusters + 1):
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(embeddings_np)
                    
                    # Check if we have enough distinct clusters
                    if len(np.unique(cluster_labels)) > 1:
                        try:
                            from sklearn.metrics import silhouette_score
                            score = silhouette_score(embeddings_np, cluster_labels)
                            silhouette_scores.append(score)
                        except Exception as e:
                            print(f"Error calculating silhouette score: {e}")
                            silhouette_scores.append(-1)
                    else:
                        silhouette_scores.append(-1)
                except Exception as e:
                    print(f"Error during clustering with {n_clusters} clusters: {e}")
                    silhouette_scores.append(-1)
        
    # Determine optimal number of clusters
    if silhouette_scores and max(silhouette_scores) > -1:
        optimal_clusters = np.argmax(silhouette_scores) + min_clusters
    else:
        # Default to 2 clusters if no valid scores
        optimal_clusters = min(2, n_samples)
    
    # Final clustering with optimal clusters
    try:
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_np)
        
        # If clustering fails to produce distinct clusters, use a fallback
        if len(np.unique(cluster_labels)) < min(2, n_samples):
            # Simple alternative: just split the data in half
            cluster_labels = np.array([i % 2 for i in range(n_samples)])
    except Exception as e:
        print(f"Error during final clustering: {e}")
        # Fallback to dummy clustering
        cluster_labels = np.array([i % min(2, n_samples) for i in range(n_samples)])
    
    return cluster_labels, optimal_clusters, silhouette_scores

def plot_embedding_visualization(embeddings: np.ndarray, cluster_labels: np.ndarray, 
                               save_dir: Optional[str] = None, title_prefix: str = ''):
    """
    Create and save PCA and t-SNE visualizations of embeddings.
    """
    # Create directory if specified
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if we have enough data
    if len(embeddings) < 2:
        print("Not enough embeddings for visualization")
        return
    
    # PCA for dimensionality reduction
    try:
        pca = PCA(n_components=2)
        pca_embeddings = pca.fit_transform(embeddings)
        
        # TSNE for visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        tsne_embeddings = tsne.fit_transform(embeddings)
        
        # Plot visualizations
        plt.figure(figsize=(15, 10))
        
        # PCA visualization
        plt.subplot(2, 1, 1)
        for cluster in np.unique(cluster_labels):
            plt.scatter(
                pca_embeddings[cluster_labels == cluster, 0],
                pca_embeddings[cluster_labels == cluster, 1],
                label=f'Cluster {cluster}',
                alpha=0.7
            )
        
        plt.title(f'{title_prefix}PCA Visualization of Contract Embeddings')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        if len(np.unique(cluster_labels)) <= 10:  # Only show legend if not too many clusters
            plt.legend()
        
        # t-SNE visualization
        plt.subplot(2, 1, 2)
        for cluster in np.unique(cluster_labels):
            plt.scatter(
                tsne_embeddings[cluster_labels == cluster, 0],
                tsne_embeddings[cluster_labels == cluster, 1],
                label=f'Cluster {cluster}',
                alpha=0.7
            )
        
        plt.title(f'{title_prefix}t-SNE Visualization of Contract Embeddings')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        if len(np.unique(cluster_labels)) <= 10:  # Only show legend if not too many clusters
            plt.legend()
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(save_dir / f'{title_prefix.replace(" ", "_")}embedding_visualization.png')
            plt.close()
        else:
            plt.show()
            
    except Exception as e:
        print(f"Error creating embedding visualization: {e}")

def analyze_and_visualize_embeddings(embeddings_np: np.ndarray, contracts: List[Dict] = None, 
                                   save_dir: Optional[str] = None) -> Dict:
    """
    Analyze embeddings and create visualizations with robust error handling.
    This function can be used as a replacement for the analyze_embeddings method in evaluate.py.
    """
    # Create save directory if specified
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
    
    # Perform robust clustering
    cluster_labels, optimal_clusters, silhouette_scores = robust_cluster_embeddings(
        embeddings_np, min_clusters=2, max_clusters=min(10, len(embeddings_np)-1)
    )
    
    # Create visualizations
    plot_embedding_visualization(
        embeddings_np, cluster_labels, 
        save_dir=save_dir, 
        title_prefix='Contract '
    )
    
    # Try dimensionality reduction for statistics
    try:
        pca = PCA(n_components=2)
        pca_embeddings = pca.fit_transform(embeddings_np)
        explained_variance = pca.explained_variance_ratio_.tolist()
    except:
        pca_embeddings = np.zeros((len(embeddings_np), 2))
        explained_variance = [0, 0]
    
    # Gather statistics
    analysis = {
        'cluster_labels': cluster_labels.tolist(),
        'optimal_clusters': int(optimal_clusters),
        'silhouette_scores': silhouette_scores,
        'pca_embeddings': pca_embeddings.tolist(),
        'embedding_statistics': {
            'mean': np.mean(embeddings_np, axis=0).tolist(),
            'std': np.std(embeddings_np, axis=0).tolist(),
            'pca_explained_variance': explained_variance
        }
    }
    
    # Add category distribution if contracts provided
    if contracts:
        try:
            category_distribution = {}
            for i, label in enumerate(cluster_labels):
                if i < len(contracts):
                    category = contracts[i].get('category_name', 'unknown')
                    if label not in category_distribution:
                        category_distribution[label] = {}
                    if category not in category_distribution[label]:
                        category_distribution[label][category] = 0
                    category_distribution[label][category] += 1
            
            analysis['category_distribution'] = category_distribution
        except Exception as e:
            print(f"Error analyzing category distribution: {e}")
    
    return analysis 

def plot_tool_metrics(yaml_file: str):
    with open(yaml_file, 'r') as f:
        metrics = yaml.safe_load(f)

    tools = list(metrics.keys())
    tprs = [metrics[t]['tpr'] for t in tools]
    fprs = [metrics[t]['fpr'] for t in tools]
    precision = [metrics[t]['precision'] for t in tools]

    x = range(len(tools))
    width = 0.2

    plt.figure(figsize=(10, 6))
    plt.bar([i - width for i in x], tprs, width=width, label='TPR')
    plt.bar(x, fprs, width=width, label='FPR')
    plt.bar([i + width for i in x], precision, width=width, label='Precision')

    plt.xticks(x, tools, rotation=45)
    plt.ylabel("Score")
    plt.title("Tool Performance Metrics")
    plt.legend()
    plt.tight_layout()
    plt.show() 
    
    
def plot_tool_confusion_matrix(tool_confusion: dict, save_path: str = None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    df = pd.DataFrame(tool_confusion).T

    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt="d", cmap="coolwarm", linewidths=0.5, linecolor='gray')
    plt.title("Confusion Matrix Components per Tool")
    plt.ylabel("Tool")
    plt.xlabel("Confusion Component")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()