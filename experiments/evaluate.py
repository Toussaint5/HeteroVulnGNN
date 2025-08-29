# -*- coding: utf-8 -*-

import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
import json
import logging

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*invalid value encountered in divide.*")
warnings.filterwarnings("ignore", message=".*KMeans is known to have a memory leak.*")
warnings.filterwarnings("ignore", message=".*distinct clusters.*")

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.loader import SolidiFIDataLoader
from data.processor import SolidityCodeProcessor
from data.graph_builder import HeterogeneousGraphBuilder
from models.heterognn import HeteroToolGNN
from utils.metrics import ToolMetricsCalculator
from utils.visualization import (
    plot_tool_performance, plot_attention_weights, plot_prediction_distribution,
    plot_per_tool_metrics, analyze_and_visualize_embeddings, plot_comprehensive_metrics
)
from sklearn.metrics import r2_score


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToolPerformanceEvaluator:
    """Comprehensive model evaluation for tool performance prediction."""
    
    def __init__(self, config_path: str, checkpoint_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.data_loader = SolidiFIDataLoader(self.config)
        self.processor = SolidityCodeProcessor(self.config)
        self.graph_builder = HeterogeneousGraphBuilder(self.config)
        self.metrics_calculator = ToolMetricsCalculator()
        
        # Load model
        self.model = HeteroToolGNN(self.config).to(self.device)
        # Handle torch.load based on PyTorch version
        if hasattr(torch, '__version__') and torch.__version__ >= '2.0.0':
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Create results directory
        self.results_dir = Path('evaluation_results')
        self.results_dir.mkdir(exist_ok=True)

    def evaluate_model(self, test_contracts: List[Dict]) -> Dict:
        """Comprehensive model evaluation for tool performance prediction."""
        results = {}
        
        # Prepare test data
        tool_results = self.data_loader.load_tool_results()
        performance_metrics = self.data_loader.load_performance_metrics()
        
        all_predictions = []
        all_targets = []
        all_embeddings = []
        contract_ids = []
        successful_contracts = []
        
        print(f"Evaluating {len(test_contracts)} test contracts...")
        
        # Process each contract with proper error handling
        with torch.no_grad():
            for i, contract in enumerate(test_contracts):
                try:
                    contract_id = contract['id']
                    
                    if i % 50 == 0:
                        print(f"Processing contract {i+1}/{len(test_contracts)}")
                    
                    # Extract features and build graph
                    ast_features = self.processor.extract_ast_features(contract.get('ast', {}))
                    hetero_data = self.graph_builder.build_heterogeneous_graph(
                        contract, ast_features, self.processor
                    )
                    
                    # Add tool performance labels
                    tool_labels = self.processor.create_tool_performance_labels(
                        contract_id, performance_metrics
                    )
                    
                    hetero_data = self.graph_builder.add_tool_performance_labels(hetero_data, tool_labels)
                    
                    # Move to device
                    hetero_data = hetero_data.to(self.device)
                    
                    # Get predictions and embeddings
                    predictions, tool_preds, embeddings = self.model(hetero_data, return_embeddings=True)
                    
                    # Handle predictions properly
                    if isinstance(predictions, tuple):
                        pred_tensor = predictions[0].cpu()
                    else:
                        pred_tensor = predictions.cpu()
                    
                    target_tensor = hetero_data['contract'].y_tool.cpu()
                    
                    # Ensure compatible shapes - both should be [batch_size, num_tools * 5]
                    if pred_tensor.dim() == 1:
                        pred_tensor = pred_tensor.unsqueeze(0)
                    if target_tensor.dim() == 1:
                        target_tensor = target_tensor.unsqueeze(0)
                    
                    # Verify shapes match
                    if pred_tensor.shape[1] != target_tensor.shape[1]:
                        # Adjust to the smaller dimension
                        min_features = min(pred_tensor.shape[1], target_tensor.shape[1])
                        pred_tensor = pred_tensor[:, :min_features]
                        target_tensor = target_tensor[:, :min_features]
                    
                    # Collect results
                    all_predictions.append(pred_tensor)
                    all_targets.append(target_tensor)
                    all_embeddings.append(embeddings['contract_embedding'].cpu())
                    contract_ids.append(contract_id)
                    successful_contracts.append(contract)
                    
                except Exception as e:
                    print(f"Error processing contract {i} ({contract.get('id', 'unknown')}): {e}")
                    continue
        
        print(f"Successfully processed {len(all_predictions)} contracts")
        
        if not all_predictions:
            print("No contracts were successfully processed!")
            return {'error': 'No contracts processed'}
        
        # Concatenate results
        try:
            final_predictions = torch.cat(all_predictions, dim=0)
            final_targets = torch.cat(all_targets, dim=0)
            final_embeddings = torch.cat(all_embeddings, dim=0)
            
            print(f"Final concatenated shapes - Predictions: {final_predictions.shape}, Targets: {final_targets.shape}")
            
        except Exception as e:
            print(f"Error concatenating results: {e}")
            return {'error': f'Concatenation failed: {e}'}
        
        # Calculate comprehensive metrics including F1 score
        try:
            results['metrics'] = self.metrics_calculator.calculate_tool_metrics(
                final_predictions, final_targets, detailed=True
            )
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            results['metrics'] = {'error': str(e)}
        
        # Analyze tool performance predictions
        try:
            results['tool_analysis'] = self.analyze_tool_performance(
                final_predictions, final_targets, contract_ids
            )
        except Exception as e:
            print(f"Error in tool analysis: {e}")
            results['tool_analysis'] = {'error': str(e)}
        
        # Analyze embeddings
        try:
            results['embedding_analysis'] = analyze_and_visualize_embeddings(
                final_embeddings.numpy(), 
                successful_contracts,
                save_dir=self.results_dir / 'embeddings'
            )
        except Exception as e:
            print(f"Error in embedding analysis: {e}")
            results['embedding_analysis'] = {'error': str(e)}
        
        # Create contract-specific analysis
        try:
            results['contract_analysis'] = self.create_contract_specific_analysis(
                final_predictions, final_targets, contract_ids
            )
        except Exception as e:
            print(f"Error in contract analysis: {e}")
            results['contract_analysis'] = {'error': str(e)}
        
        # Calculate correlation analysis
        try:
            results['correlation_analysis'] = self.metrics_calculator.analyze_tool_correlations(
                final_predictions, final_targets
            )
        except Exception as e:
            print(f"Error in correlation analysis: {e}")
            results['correlation_analysis'] = {'error': str(e)} 
            

        # Add FP/FN analysis
        try:
            results['fp_fn_analysis'] = self.analyze_false_positives_negatives(test_contracts)
        except Exception as e:
            print(f"Error in FP/FN analysis: {e}")
            results['fp_fn_analysis'] = {'error': str(e)}
        
        return results
    
    def analyze_tool_performance(self, predictions: torch.Tensor, targets: torch.Tensor, 
                               contract_ids: List[str]) -> Dict:
        """Analyze tool-specific performance with F1 scores."""
        tool_names = ['oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither']
        metric_names = ['tpr', 'fpr', 'accuracy', 'precision', 'recall']
        
        preds = predictions.numpy()
        targs = targets.numpy()
        
        # Reshape to [batch_size, num_tools, num_metrics]
        batch_size = preds.shape[0]
        num_tools = 6
        num_metrics = 5
        
        if preds.shape[1] == num_tools * num_metrics:
            preds = preds.reshape(batch_size, num_tools, num_metrics)
            targs = targs.reshape(batch_size, num_tools, num_metrics)
        
        analysis = {}
        
        # Per-tool analysis
        for i, tool in enumerate(tool_names):
            if i >= preds.shape[1]:
                continue
                
            tool_preds = preds[:, i, :]
            tool_targs = targs[:, i, :]
            
            # Calculate F1 scores
            precision_pred = tool_preds[:, 3]
            recall_pred = tool_preds[:, 4]
            f1_pred = 2 * precision_pred * recall_pred / (precision_pred + recall_pred + 1e-8)
            
            precision_targ = tool_targs[:, 3]
            recall_targ = tool_targs[:, 4]
            f1_targ = 2 * precision_targ * recall_targ / (precision_targ + recall_targ + 1e-8)
            
            analysis[tool] = {
                'mae': np.mean(np.abs(tool_preds - tool_targs), axis=0).tolist(),
                'mse': np.mean((tool_preds - tool_targs) ** 2, axis=0).tolist(),
                'bias': np.mean(tool_preds - tool_targs, axis=0).tolist(),
                'std': np.std(tool_preds - tool_targs, axis=0).tolist(),
                'f1_mae': float(np.mean(np.abs(f1_pred - f1_targ))),
                'f1_mse': float(np.mean((f1_pred - f1_targ) ** 2)),
                'predictions': {
                    'mean': np.mean(tool_preds, axis=0).tolist(),
                    'std': np.std(tool_preds, axis=0).tolist()
                },
                'targets': {
                    'mean': np.mean(tool_targs, axis=0).tolist(),
                    'std': np.std(tool_targs, axis=0).tolist()
                }
            }
            
            # Add per-metric analysis
            for j, metric in enumerate(metric_names):
                analysis[f'{tool}_{metric}'] = {
                    'mae': float(np.mean(np.abs(tool_preds[:, j] - tool_targs[:, j]))),
                    'mse': float(np.mean((tool_preds[:, j] - tool_targs[:, j]) ** 2)),
                    'r2': float(np.corrcoef(tool_preds[:, j], tool_targs[:, j])[0, 1] ** 2) 
                          if np.std(tool_preds[:, j]) > 0 and np.std(tool_targs[:, j]) > 0 else 0
                }
        
        return analysis
    
    def create_contract_specific_analysis(self, predictions: torch.Tensor, targets: torch.Tensor,
                                        contract_ids: List[str]) -> Dict:
        """Create detailed analysis for each contract."""
        preds = predictions.numpy()
        targs = targets.numpy()
        
        # Calculate per-contract metrics
        contract_metrics = {}
        
        for i, contract_id in enumerate(contract_ids):
            if i >= preds.shape[0]:
                continue
                
            contract_pred = preds[i]
            contract_targ = targs[i]
            
            # Calculate MAE for this contract
            mae = np.mean(np.abs(contract_pred - contract_targ))
            mse = np.mean((contract_pred - contract_targ) ** 2)
            
            # Find worst predicted metric for this contract
            errors = np.abs(contract_pred - contract_targ)
            if len(errors) >= 30:  # 6 tools * 5 metrics
                errors_reshaped = errors.reshape(6, 5)
                worst_tool_idx = np.unravel_index(np.argmax(errors_reshaped), errors_reshaped.shape)[0]
                worst_metric_idx = np.unravel_index(np.argmax(errors_reshaped), errors_reshaped.shape)[1]
                
                tool_names = ['oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither']
                metric_names = ['tpr', 'fpr', 'accuracy', 'precision', 'recall']
                
                worst_tool = tool_names[worst_tool_idx] if worst_tool_idx < len(tool_names) else f'tool_{worst_tool_idx}'
                worst_metric = metric_names[worst_metric_idx] if worst_metric_idx < len(metric_names) else f'metric_{worst_metric_idx}'
                
                contract_metrics[contract_id] = {
                    'mae': float(mae),
                    'mse': float(mse),
                    'worst_prediction': {
                        'tool': worst_tool,
                        'metric': worst_metric,
                        'error': float(np.max(errors))
                    }
                }
        
        # Summary statistics
        all_maes = [m['mae'] for m in contract_metrics.values()]
        
        return {
            'per_contract': contract_metrics,
            'summary': {
                'best_contract': min(contract_metrics.items(), key=lambda x: x[1]['mae'])[0] if contract_metrics else None,
                'worst_contract': max(contract_metrics.items(), key=lambda x: x[1]['mae'])[0] if contract_metrics else None,
                'average_mae': float(np.mean(all_maes)) if all_maes else 0,
                'std_mae': float(np.std(all_maes)) if all_maes else 0
            }
        }
    
    def generate_visualizations(self, results: Dict):
        """Generate comprehensive visualizations for the evaluation results."""
        try:
            # Only generate visualizations if we have valid results
            if 'error' in results:
                print(f"Skipping visualizations due to error: {results['error']}")
                return
            
            # Create visualization subdirectories
            vis_dirs = {
                'overall': self.results_dir / 'overall',
                'tools': self.results_dir / 'tools',
                'metrics': self.results_dir / 'metrics',
                'distributions': self.results_dir / 'distributions'
            }
            
            for dir_path in vis_dirs.values():
                dir_path.mkdir(exist_ok=True)
            
            # 1. Overall tool performance comparison including F1
            if 'metrics' in results and 'error' not in results['metrics']:
                plot_comprehensive_metrics(
                    results['metrics'],
                    save_dir=vis_dirs['overall']
                )
            
            # 2. Tool-specific performance
            if 'tool_analysis' in results and 'error' not in results['tool_analysis']:
                plot_per_tool_metrics(
                    results['metrics'],
                    save_dir=vis_dirs['tools']
                )
            
            # 3. Prediction distribution analysis
            if 'metrics' in results and 'prediction_stats' in results['metrics']:
                self._plot_prediction_statistics(
                    results['metrics'],
                    save_dir=vis_dirs['distributions']
                )
            
            # 4. Correlation heatmaps
            if 'correlation_analysis' in results and 'error' not in results['correlation_analysis']:
                self._plot_correlation_heatmaps(
                    results['correlation_analysis'],
                    save_dir=vis_dirs['metrics']
                )
            
            # 5. Per-metric performance across all tools
            if 'metrics' in results:
                self._plot_metric_comparison(
                    results['metrics'],
                    save_dir=vis_dirs['metrics']
                )
            
            print(f"Visualizations saved to {self.results_dir}")
            
        except Exception as e:
            print(f"Error generating visualizations: {e}")
    
    def _plot_prediction_statistics(self, metrics: Dict, save_dir: Path):
        """Plot statistics of predictions and targets."""
        if 'prediction_stats' not in metrics or 'target_stats' not in metrics:
            return
        
        pred_stats = metrics['prediction_stats']
        target_stats = metrics['target_stats']
        
        # Global statistics comparison
        plt.figure(figsize=(12, 8))
        
        # Mean values comparison
        plt.subplot(2, 2, 1)
        tool_names = ['oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither']
        metric_names = ['tpr', 'fpr', 'accuracy', 'precision', 'recall']
        
        pred_means = []
        target_means = []
        labels = []
        
        for tool in tool_names:
            if f'{tool}_mean' in pred_stats and f'{tool}_mean' in target_stats:
                for i, metric in enumerate(metric_names):
                    if i < len(pred_stats[f'{tool}_mean']):
                        pred_means.append(pred_stats[f'{tool}_mean'][i])
                        target_means.append(target_stats[f'{tool}_mean'][i])
                        labels.append(f'{tool[:3]}-{metric[:3]}')
        
        x = np.arange(len(labels))
        width = 0.35
        
        plt.bar(x - width/2, pred_means, width, label='Predictions', alpha=0.8)
        plt.bar(x + width/2, target_means, width, label='Targets', alpha=0.8)
        plt.xlabel('Tool-Metric')
        plt.ylabel('Mean Value')
        plt.title('Mean Predictions vs Targets')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Distribution of prediction errors
        plt.subplot(2, 2, 2)
        if 'detailed_errors' in metrics:
            all_errors = []
            for tool_errors in metrics['detailed_errors'].values():
                for metric_errors in tool_errors.values():
                    all_errors.extend(metric_errors)
            
            plt.hist(all_errors, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Absolute Error')
            plt.ylabel('Frequency')
            plt.title('Distribution of Prediction Errors')
            plt.axvline(np.mean(all_errors), color='red', linestyle='--', label=f'Mean: {np.mean(all_errors):.3f}')
            plt.legend()
        
        # Per-metric statistics
        plt.subplot(2, 2, 3)
        metric_means = []
        metric_stds = []
        
        for metric in metric_names:
            if f'{metric}_mean' in pred_stats:
                metric_means.append(pred_stats[f'{metric}_mean'])
                metric_stds.append(pred_stats[f'{metric}_std'])
        
        plt.bar(metric_names, metric_means, yerr=metric_stds, capsize=5)
        plt.xlabel('Metric')
        plt.ylabel('Predicted Value')
        plt.title('Average Predicted Values by Metric (with std)')
        plt.xticks(rotation=45)
        
        # Box plot of errors by tool
        plt.subplot(2, 2, 4)
        if 'detailed_errors' in metrics:
            tool_errors_list = []
            tool_labels = []
            
            for tool in tool_names:
                if tool in metrics['detailed_errors']:
                    tool_errors = []
                    for metric_errors in metrics['detailed_errors'][tool].values():
                        tool_errors.extend(metric_errors)
                    if tool_errors:
                        tool_errors_list.append(tool_errors)
                        tool_labels.append(tool.title())
            
            if tool_errors_list:
                plt.boxplot(tool_errors_list, labels=tool_labels)
                plt.xlabel('Tool')
                plt.ylabel('Absolute Error')
                plt.title('Error Distribution by Tool')
                plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'prediction_statistics.png')
        plt.close()
    
    def _plot_correlation_heatmaps(self, correlation_analysis: Dict, save_dir: Path):
        """Plot correlation heatmaps."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Tool correlation heatmap
        if 'tool_error_correlation' in correlation_analysis:
            tool_corr = np.array(correlation_analysis['tool_error_correlation'])
            tool_names = ['Oyente', 'Securify', 'Mythril', 'SmartCheck', 'Manticore', 'Slither']
            
            sns.heatmap(tool_corr, annot=True, fmt='.2f', cmap='coolwarm',
                       xticklabels=tool_names, yticklabels=tool_names,
                       center=0, ax=ax1)
            ax1.set_title('Tool Error Correlations')
        
        # Metric correlation heatmap
        if 'metric_error_correlation' in correlation_analysis:
            metric_corr = np.array(correlation_analysis['metric_error_correlation'])
            metric_names = ['TPR', 'FPR', 'Accuracy', 'Precision', 'Recall', 'F1']
            
            # Ensure we have the right shape
            if metric_corr.shape[0] >= len(metric_names):
                metric_corr = metric_corr[:len(metric_names), :len(metric_names)]
            
            sns.heatmap(metric_corr, annot=True, fmt='.2f', cmap='coolwarm',
                       xticklabels=metric_names[:metric_corr.shape[0]], 
                       yticklabels=metric_names[:metric_corr.shape[1]],
                       center=0, ax=ax2)
            ax2.set_title('Metric Error Correlations')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'correlation_heatmaps.png')
        plt.close()
    
    def _plot_metric_comparison(self, metrics: Dict, save_dir: Path):
        """Plot comprehensive metric comparison across all tools."""
        metric_names = ['tpr', 'fpr', 'accuracy', 'precision', 'recall', 'f1_score']
        tool_names = ['oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither']
        
        # Create figure with subplots for each metric
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metric_names):
            ax = axes[i]
            
            # Collect MAE values for this metric across all tools
            mae_values = []
            mse_values = []
            tool_labels = []
            
            for tool in tool_names:
                key_mae = f'{tool}_{metric}_mae'
                key_mse = f'{tool}_{metric}_mse'
                
                if key_mae in metrics:
                    mae_values.append(metrics[key_mae])
                    mse_values.append(metrics[key_mse])
                    tool_labels.append(tool.title())
            
            if mae_values:
                x = np.arange(len(tool_labels))
                width = 0.35
                
                ax.bar(x - width/2, mae_values, width, label='MAE', alpha=0.8)
                ax.bar(x + width/2, mse_values, width, label='MSE', alpha=0.8)
                
                ax.set_xlabel('Tool')
                ax.set_ylabel('Error')
                ax.set_title(f'{metric.upper()} Performance')
                ax.set_xticks(x)
                ax.set_xticklabels(tool_labels, rotation=45, ha='right')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'metric_comparison.png')
        plt.close()
    
    def generate_report(self, results: Dict):
        """Generate comprehensive evaluation report with F1 scores."""
        report_path = self.results_dir / 'tool_performance_evaluation_report.md'
        
        try:
            with open(report_path, 'w') as f:
                f.write("# Tool Performance Prediction Evaluation Report\n\n")
                
                # Check for errors
                if 'error' in results:
                    f.write(f"## Error\n\nEvaluation encountered an error: {results['error']}\n\n")
                    return
                
                # Summary metrics
                if 'metrics' in results and 'error' not in results['metrics']:
                    f.write("## Summary Metrics\n\n")
                    metrics = results['metrics']
                    f.write(f"**Overall Performance:**\n")
                    f.write(f"- Overall MSE: {metrics.get('overall_mse', 0.0):.4f}\n")
                    f.write(f"- Overall MAE: {metrics.get('overall_mae', 0.0):.4f}\n\n")
                    
                    # Per-metric performance including F1
                    f.write("## Per-Metric Performance\n\n")
                    metric_names = ['tpr', 'fpr', 'accuracy', 'precision', 'recall', 'f1_score']
                    
                    f.write("| Metric | Average MAE | Average MSE | Average R^2 |\n")
                    f.write("|--------|-------------|-------------|------------|\n")
                    
                    for metric in metric_names:
                        mae = metrics.get(f'{metric}_avg_mae', 0.0)
                        mse = metrics.get(f'{metric}_avg_mse', 0.0)
                        r2 = metrics.get(f'{metric}_avg_r2', 0.0)
                        f.write(f"| {metric.upper()} | {mae:.4f} | {mse:.4f} | {r2:.4f} |\n")
                    
                    f.write("\n")
                    
                    # Per-tool performance
                    f.write("## Per-Tool Performance\n\n")
                    tool_names = ['oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither']
                    
                    f.write("| Tool | Overall MAE | Overall MSE | R^2 | F1 MAE | F1 MSE |\n")
                    f.write("|------|-------------|-------------|-------|---------|--------|\n")
                    
                    for tool in tool_names:
                        mae = metrics.get(f'{tool}_mae', 0.0)
                        mse = metrics.get(f'{tool}_mse', 0.0)
                        r2 = metrics.get(f'{tool}_r2', 0.0)
                        
                        # Get F1 specific metrics from tool analysis
                        f1_mae = 0.0
                        f1_mse = 0.0
                        if 'tool_analysis' in results and tool in results['tool_analysis']:
                            f1_mae = results['tool_analysis'][tool].get('f1_mae', 0.0)
                            f1_mse = results['tool_analysis'][tool].get('f1_mse', 0.0)
                        
                        f.write(f"| {tool.title()} | {mae:.4f} | {mse:.4f} | {r2:.4f} | {f1_mae:.4f} | {f1_mse:.4f} |\n")
                    
                    f.write("\n")
                    
                    # Tool-Metric Detailed Analysis
                    f.write("## Tool-Metric Detailed Analysis\n\n")
                    
                    for tool in tool_names:
                        f.write(f"### {tool.title()}\n\n")
                        f.write("| Metric | MAE | MSE | R^2 | Binary Accuracy |\n")
                        f.write("|--------|-----|-----|----|-----------------|\n")
                        
                        for metric in metric_names:
                            key_mae = f'{tool}_{metric}_mae'
                            key_mse = f'{tool}_{metric}_mse'
                            key_r2 = f'{tool}_{metric}_r2'
                            
                            mae = metrics.get(key_mae, 0.0)
                            mse = metrics.get(key_mse, 0.0)
                            r2 = metrics.get(key_r2, 0.0)
                            
                            # Get binary accuracy if available
                            binary_acc = 'N/A'
                            if 'binary_metrics' in metrics and tool in metrics['binary_metrics']:
                                tool_binary = metrics['binary_metrics'][tool]
                                if f'{metric}_binary_accuracy' in tool_binary:
                                    binary_acc = f"{tool_binary[f'{metric}_binary_accuracy']:.4f}"
                            
                            f.write(f"| {metric.upper()} | {mae:.4f} | {mse:.4f} | {r2:.4f} | {binary_acc} |\n")
                        
                        f.write("\n")
                
                # Embedding Analysis
                if ('embedding_analysis' in results and 
                    'error' not in results['embedding_analysis']):
                    f.write("\n## Embedding Analysis\n\n")
                    embed_analysis = results['embedding_analysis']
                    f.write(f"- Optimal number of clusters: {embed_analysis.get('optimal_clusters', 'N/A')}\n")
                    
                    pca_var = embed_analysis.get('embedding_statistics', {}).get('pca_explained_variance', [])
                    if len(pca_var) >= 2:
                        f.write(f"- PCA explained variance: PC1={pca_var[0]:.4f}, PC2={pca_var[1]:.4f}\n\n")
                    
                    cluster_labels = embed_analysis.get('cluster_labels', [])
                    if cluster_labels:
                        f.write("**Cluster distribution:**\n")
                        cluster_counts = np.bincount(np.array(cluster_labels))
                        for i, count in enumerate(cluster_counts):
                            f.write(f"- Cluster {i}: {count} contracts\n")
                    
                    # Category distribution in clusters
                    if 'category_distribution' in embed_analysis:
                        f.write("\n**Category distribution by cluster:**\n")
                        for cluster_id, categories in embed_analysis['category_distribution'].items():
                            f.write(f"\nCluster {cluster_id}:\n")
                            for category, count in categories.items():
                                f.write(f"  - {category}: {count}\n")
                
                # Contract-specific analysis
                if ('contract_analysis' in results and 
                    'error' not in results['contract_analysis']):
                    f.write("\n## Contract-Specific Analysis\n\n")
                    contract_analysis = results['contract_analysis']
                    
                    if 'summary' in contract_analysis:
                        summary = contract_analysis['summary']
                        f.write(f"- Best predicted contract: {summary.get('best_contract', 'N/A')}\n")
                        f.write(f"- Worst predicted contract: {summary.get('worst_contract', 'N/A')}\n")
                        f.write(f"- Average contract MAE: {summary.get('average_mae', 0):.4f}\n")
                        f.write(f"- Standard deviation of MAE: {summary.get('std_mae', 0):.4f}\n")
                
                # Correlation Analysis
                if ('correlation_analysis' in results and 
                    'error' not in results['correlation_analysis']):
                    f.write("\n## Correlation Analysis\n\n")
                    corr_analysis = results['correlation_analysis']
                    
                    f.write("**Tool Error Correlations:**\n")
                    f.write("High correlations indicate tools tend to make similar prediction errors.\n\n")
                    
                    if 'tool_error_correlation' in corr_analysis:
                        tool_corr = np.array(corr_analysis['tool_error_correlation'])
                        tool_names_cap = ['Oyente', 'Securify', 'Mythril', 'SmartCheck', 'Manticore', 'Slither']
                        
                        # Find highest correlations
                        high_corrs = []
                        for i in range(len(tool_names_cap)):
                            for j in range(i+1, len(tool_names_cap)):
                                if i < tool_corr.shape[0] and j < tool_corr.shape[1]:
                                    corr = tool_corr[i, j]
                                    if abs(corr) > 0.5:
                                        high_corrs.append((tool_names_cap[i], tool_names_cap[j], corr))
                        
                        if high_corrs:
                            f.write("Notable correlations (|r| > 0.5):\n")
                            for tool1, tool2, corr in sorted(high_corrs, key=lambda x: abs(x[2]), reverse=True):
                                f.write(f"- {tool1} ? {tool2}: {corr:.3f}\n")
                
                # Conclusion
                f.write("\n## Conclusion\n\n")
                if 'metrics' in results and 'error' not in results['metrics']:
                    metrics = results['metrics']
                    f.write("The model's predictions for tool performance metrics show varying levels of accuracy across different tools and metrics. ")
                    
                    # Find best and worst predicted tool
                    tool_names = ['oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither']
                    tool_maes = [(tool, metrics.get(f'{tool}_mae', float('inf'))) for tool in tool_names]
                    valid_tool_maes = [(tool, mae) for tool, mae in tool_maes if mae != float('inf')]
                    
                    if valid_tool_maes:
                        best_tool = min(valid_tool_maes, key=lambda x: x[1])
                        worst_tool = max(valid_tool_maes, key=lambda x: x[1])
                        
                        f.write(f"The model performs best when predicting {best_tool[0].title()} (MAE: {best_tool[1]:.4f}) ")
                        f.write(f"and worst when predicting {worst_tool[0].title()} (MAE: {worst_tool[1]:.4f}).\n\n")
                    
                    # Find best and worst predicted metric
                    metric_names = ['tpr', 'fpr', 'accuracy', 'precision', 'recall', 'f1_score']
                    metric_maes = [(metric, metrics.get(f'{metric}_avg_mae', float('inf'))) for metric in metric_names]
                    valid_metric_maes = [(metric, mae) for metric, mae in metric_maes if mae != float('inf')]
                    
                    if valid_metric_maes:
                        best_metric = min(valid_metric_maes, key=lambda x: x[1])
                        worst_metric = max(valid_metric_maes, key=lambda x: x[1])
                        
                        f.write(f"Among the metrics, {best_metric[0].upper()} is predicted most accurately (MAE: {best_metric[1]:.4f}) ")
                        f.write(f"while {worst_metric[0].upper()} shows the highest prediction error (MAE: {worst_metric[1]:.4f}).\n")
                    
                    # Add insights about F1 score performance
                    f1_mae = metrics.get('f1_score_avg_mae', None)
                    if f1_mae is not None:
                        f.write(f"\nThe F1 score predictions have an average MAE of {f1_mae:.4f}, ")
                        f.write("indicating the model's ability to balance precision and recall predictions.\n")
            
            print(f"Evaluation report saved to {report_path}")
            
            # Generate detailed CSV output
            self._generate_csv_outputs(results)
                
        except Exception as e:
            print(f"Error generating report: {e}")
    
    def _generate_csv_outputs(self, results: Dict):
        """Generate CSV files with detailed metrics."""
        try:
            # Tool-metric analysis CSV
            if 'metrics' in results and 'error' not in results['metrics']:
                metrics = results['metrics']
                tool_names = ['oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither']
                metric_names = ['tpr', 'fpr', 'accuracy', 'precision', 'recall', 'f1_score']
                
                df_data = []
                for tool in tool_names:
                    for metric in metric_names:
                        key_mae = f'{tool}_{metric}_mae'
                        key_mse = f'{tool}_{metric}_mse'
                        key_r2 = f'{tool}_{metric}_r2'
                        
                        row = {
                            'Tool': tool,
                            'Metric': metric,
                            'MAE': metrics.get(key_mae, 0),
                            'MSE': metrics.get(key_mse, 0),
                            'R2': metrics.get(key_r2, 0)
                        }
                        
                        # Add binary accuracy if available
                        if 'binary_metrics' in metrics and tool in metrics['binary_metrics']:
                            tool_binary = metrics['binary_metrics'][tool]
                            if f'{metric}_binary_accuracy' in tool_binary:
                                row['Binary_Accuracy'] = tool_binary[f'{metric}_binary_accuracy']
                        
                        df_data.append(row)
                
                if df_data:
                    df = pd.DataFrame(df_data)
                    df.to_csv(self.results_dir / 'tool_metric_detailed_analysis.csv', index=False)
                    print(f"Detailed analysis saved to {self.results_dir / 'tool_metric_detailed_analysis.csv'}")
            
            # Contract-specific analysis CSV
            if ('contract_analysis' in results and 
                'per_contract' in results['contract_analysis']):
                contract_data = []
                for contract_id, metrics in results['contract_analysis']['per_contract'].items():
                    row = {
                        'Contract_ID': contract_id,
                        'MAE': metrics['mae'],
                        'MSE': metrics['mse'],
                        'Worst_Tool': metrics['worst_prediction']['tool'],
                        'Worst_Metric': metrics['worst_prediction']['metric'],
                        'Worst_Error': metrics['worst_prediction']['error']
                    }
                    contract_data.append(row)
                
                if contract_data:
                    df = pd.DataFrame(contract_data)
                    df.to_csv(self.results_dir / 'contract_specific_analysis.csv', index=False)
                    print(f"Contract analysis saved to {self.results_dir / 'contract_specific_analysis.csv'}")
                    
        except Exception as e:
            print(f"Error generating CSV outputs: {e}")
    
    def save_detailed_results(self, results: Dict):
        """Save detailed results in JSON format."""
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict):
                # Convert both keys and values
                return {str(key): convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy(item) for item in obj)
            else:
                return obj
        
        with open(self.results_dir / 'detailed_results.json', 'w') as f:
            json.dump(convert_numpy(results), f, indent=2) 
            
            
    def analyze_false_positives_negatives(self, test_contracts: List[Dict]) -> Dict:
        """Detailed analysis of false positives and false negatives."""
        logger.info("Analyzing False Positives and False Negatives...")
        
        # Get detailed confusion matrix
        detailed_confusion = self.data_loader.get_detailed_confusion_matrix_per_vulnerability()
        
        # Calculate metrics
        vulnerability_metrics = self.metrics_calculator.calculate_per_vulnerability_metrics(detailed_confusion)
        
        # Create analysis report
        report = self.metrics_calculator.create_vulnerability_performance_report(vulnerability_metrics)
        
        # Save detailed results
        results_dir = self.results_dir / 'fp_fn_analysis'
        results_dir.mkdir(exist_ok=True)
        
        # Save detailed confusion matrices
        with open(results_dir / 'detailed_confusion_matrices.json', 'w') as f:
            json.dump(detailed_confusion, f, indent=2)
        
        # Save vulnerability metrics
        with open(results_dir / 'vulnerability_metrics.json', 'w') as f:
            json.dump(vulnerability_metrics, f, indent=2)
        
        # Save report
        with open(results_dir / 'fp_fn_analysis_report.md', 'w') as f:
            f.write(report)
        
        # Create visualizations
        self._create_fp_fn_visualizations(vulnerability_metrics, results_dir)
        
        logger.info(f"FP/FN analysis saved to {results_dir}")
        
        return {
            'detailed_confusion': detailed_confusion,
            'vulnerability_metrics': vulnerability_metrics,
            'report': report
        }  
        
        

    def generate_fp_fn_tables(self, test_contracts: List[Dict]):
        """Generate FP/FN tables matching reference format"""
        
        from analysis.fp_fn_analyzer import FalsePositiveFalseNegativeAnalyzer
        
        analyzer = FalsePositiveFalseNegativeAnalyzer(self.data_loader)
        
        # Generate tables
        fn_table = analyzer.generate_false_negative_table()
        fp_table = analyzer.generate_false_positive_table()
        
        # Format output
        formatted_report = analyzer.format_tables_for_output(fn_table, fp_table)
        
        # Save results
        results_dir = self.results_dir / 'fp_fn_analysis'
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / 'fp_fn_tables.md', 'w') as f:
            f.write(formatted_report)
        
        # Save raw data
        with open(results_dir / 'fn_table.json', 'w') as f:
            json.dump(fn_table, f, indent=2)
        
        with open(results_dir / 'fp_table.json', 'w') as f:
            json.dump(fp_table, f, indent=2)
        
        logger.info(f"FP/FN tables saved to {results_dir}")
        
        return fn_table, fp_table
        
        

    def _create_fp_fn_visualizations(self, vulnerability_metrics: Dict, save_dir: Path):
        """Create visualizations for FP/FN analysis."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        
        # 1. FPR vs FNR scatter plot
        plt.figure(figsize=(12, 8))
        
        for tool in vulnerability_metrics:
            fprs = []
            fnrs = []
            labels = []
            
            for vuln_type, metrics in vulnerability_metrics[tool].items():
                fprs.append(metrics['false_positive_rate'])
                fnrs.append(metrics['false_negative_rate'])
                labels.append(f"{tool}-{vuln_type}")
            
            plt.scatter(fprs, fnrs, label=tool.title(), alpha=0.7, s=100)
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('False Negative Rate')
        plt.title('Tool Performance: FPR vs FNR by Vulnerability Type')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_dir / 'fpr_vs_fnr_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Heatmap of FP counts
        tools = list(vulnerability_metrics.keys())
        vuln_types = list(next(iter(vulnerability_metrics.values())).keys())
        
        fp_matrix = []
        fn_matrix = []
        
        for tool in tools:
            fp_row = [vulnerability_metrics[tool][vuln]['FP'] for vuln in vuln_types]
            fn_row = [vulnerability_metrics[tool][vuln]['FN'] for vuln in vuln_types]
            fp_matrix.append(fp_row)
            fn_matrix.append(fn_row)
        
        # False Positives Heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(fp_matrix, annot=True, fmt='d', 
                    xticklabels=[v.replace('_', ' ').title() for v in vuln_types],
                    yticklabels=[t.title() for t in tools],
                    cmap='Reds', cbar_kws={'label': 'False Positives'})
        plt.title('False Positives by Tool and Vulnerability Type')
        plt.tight_layout()
        plt.savefig(save_dir / 'false_positives_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # False Negatives Heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(fn_matrix, annot=True, fmt='d',
                    xticklabels=[v.replace('_', ' ').title() for v in vuln_types],
                    yticklabels=[t.title() for t in tools],
                    cmap='Blues', cbar_kws={'label': 'False Negatives'})
        plt.title('False Negatives by Tool and Vulnerability Type')
        plt.tight_layout()
        plt.savefig(save_dir / 'false_negatives_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
            

def main():
    parser = argparse.ArgumentParser(description='Evaluate HeteroToolGNN')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = ToolPerformanceEvaluator(args.config, args.checkpoint)
        
        # Load test data
        contracts = evaluator.data_loader.load_contracts()
        _, _, test_contracts = evaluator.data_loader.get_dataset_split(contracts)
        
        print(f"Evaluating on {len(test_contracts)} test contracts...")
        
        # Evaluate model
        results = evaluator.evaluate_model(test_contracts)
        
        # Generate visualizations
        evaluator.generate_visualizations(results)
        
        # Generate report
        evaluator.generate_report(results)
        
        # Save detailed results
        evaluator.save_detailed_results(results)
        
        print("Evaluation completed!")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detailed FP/FN Analysis')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    
    evaluator = ToolPerformanceEvaluator(args.config, args.checkpoint)
    
    contracts = evaluator.data_loader.load_contracts()
    _, _, test_contracts = evaluator.data_loader.get_dataset_split(contracts)
    
    # Run detailed FP/FN analysis
    results = evaluator.analyze_false_positives_negatives(test_contracts)
    
    print("=== FALSE POSITIVE/NEGATIVE ANALYSIS COMPLETE ===")
    print(f"Results saved to: evaluation_results/fp_fn_analysis/")
