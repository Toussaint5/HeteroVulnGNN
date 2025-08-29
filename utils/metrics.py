# -*- coding: utf-8 -*-

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import logging

# Define the logger
logger = logging.getLogger(__name__)

class ToolMetricsCalculator:
    """Calculate metrics for tool performance prediction."""
    
    def __init__(self, num_tools: int = 6):
        self.num_tools = num_tools
        self.tool_names = [
            'oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither'
        ]
        self.metric_names = ['tpr', 'fpr', 'accuracy', 'precision', 'recall']
    
    
    def calculate_tool_metrics(self, predictions: torch.Tensor, 
                        targets: torch.Tensor,
                        detailed: bool = False) -> Dict:
        """Calculate comprehensive metrics with robust shape handling."""
        
        # Handle tuple predictions
        if isinstance(predictions, tuple):
            if len(predictions) == 2:
                all_preds, _ = predictions
                preds = all_preds.detach().cpu().numpy()
            else:
                preds = predictions[0].detach().cpu().numpy()
        else:
            preds = predictions.detach().cpu().numpy()
        
        targs = targets.detach().cpu().numpy()
        
        logger.debug(f"[Metrics] Initial shapes - preds: {preds.shape}, targets: {targs.shape}")
        
        # ROBUST SHAPE ALIGNMENT
        # Step 1: Ensure both are 2D
        if preds.ndim == 1:
            preds = preds.reshape(1, -1)
        if targs.ndim == 1:
            targs = targs.reshape(1, -1)
        
        # Step 2: Align batch dimension
        min_batch = min(preds.shape[0], targs.shape[0])
        preds = preds[:min_batch]
        targs = targs[:min_batch]
        
        # Step 3: Align feature dimension
        min_features = min(preds.shape[1], targs.shape[1])
        preds = preds[:, :min_features]
        targs = targs[:, :min_features]
        
        logger.debug(f"[Metrics] Aligned shapes - preds: {preds.shape}, targets: {targs.shape}")
        
        # Step 4: Determine effective dimensions
        expected_features = self.num_tools * 5  
        actual_features = min_features
        
        if actual_features % 5 != 0:
            # Can't reshape properly, use what we have
            logger.warning(f"Features ({actual_features}) not divisible by 5, using partial analysis")
            effective_tools = actual_features // 5
            usable_features = effective_tools * 5
            if usable_features > 0:
                preds = preds[:, :usable_features]
                targs = targs[:, :usable_features]
                actual_features = usable_features
            else:
                # Too few features, return basic metrics only
                return {
                    'overall_mse': float(np.mean((preds - targs) ** 2)),
                    'overall_mae': float(np.mean(np.abs(preds - targs))),
                    'error': 'Insufficient features for tool-wise analysis'
                }
        
        effective_tools = actual_features // 5
        
        # Step 5: Reshape to (batch_size, num_tools, 5)
        try:
            batch_size = preds.shape[0]
            preds_reshaped = preds.reshape(batch_size, effective_tools, 5)
            targs_reshaped = targs.reshape(batch_size, effective_tools, 5)
        except ValueError as e:
            logger.error(f"Reshape failed: {e}")
            return {
                'overall_mse': float(np.mean((preds - targs) ** 2)),
                'overall_mae': float(np.mean(np.abs(preds - targs))),
                'error': f'Reshape failed: {e}'
            } 
            
        #Define RMSE calculations
        def compute_rmse(preds, targs):
            return float(np.sqrt(np.mean((preds - targs) ** 2)))

        # #Define MAPE calculations
        # def compute_mape(preds, targs, epsilon=1e-8):
        #     mask = np.abs(targs) > epsilon
        #     if np.sum(mask) == 0:
        #         return 0.0
        #     return float(np.mean(np.abs((preds[mask] - targs[mask]) / targs[mask])) * 100) 
        
        #Define MAPE calculations
        def compute_mape(preds, targs, epsilon=1e-8):
            # Avoid division by zero: mask out elements where target is too small
            valid_mask = np.abs(targs) > epsilon

            if np.sum(valid_mask) == 0:
                return 0.0  # No valid targets to compute MAPE

            absolute_percentage_errors = np.abs((targs[valid_mask] - preds[valid_mask]) / targs[valid_mask])
            mape = np.mean(absolute_percentage_errors) * 100.0

            return float(mape)    
        
        metrics = {}
        
        # Overall metrics
        with np.errstate(invalid='ignore', divide='ignore'):
            metrics['overall_mse'] = float(np.mean((preds - targs) ** 2))
            metrics['overall_mae'] = float(np.mean(np.abs(preds - targs))) 
            metrics['overall_rmse'] = compute_rmse(preds, targs)
            metrics['overall_mape'] = compute_mape(preds, targs)

        
        # Per-tool metrics
        tool_names_subset = self.tool_names[:effective_tools]
        
        for i, tool_name in enumerate(tool_names_subset):
            tool_preds = preds_reshaped[:, i, :]  # Shape: (batch_size, 5)
            tool_targs = targs_reshaped[:, i, :]
            
            with np.errstate(invalid='ignore', divide='ignore'):
                # Tool-level aggregated metrics
                tool_mse = np.mean((tool_preds - tool_targs) ** 2)
                tool_mae = np.mean(np.abs(tool_preds - tool_targs)) 
                tool_rmse = compute_rmse(tool_preds, tool_targs)
                tool_mape = compute_mape(tool_preds, tool_targs)
                
                # R-squared calculation
                ss_res = np.sum((tool_targs - tool_preds) ** 2)
                ss_tot = np.sum((tool_targs - np.mean(tool_targs)) ** 2)
                tool_r2 = 1 - (ss_res / (ss_tot + 1e-8)) if ss_tot > 1e-8 else 0.0
                
                metrics[f'{tool_name}_mse'] = float(tool_mse) if np.isfinite(tool_mse) else 0.0
                metrics[f'{tool_name}_mae'] = float(tool_mae) if np.isfinite(tool_mae) else 0.0
                metrics[f'{tool_name}_r2'] = float(tool_r2) if np.isfinite(tool_r2) else 0.0 
                metrics[f'{tool_name}_rmse'] = tool_rmse
                metrics[f'{tool_name}_mape'] = tool_mape
            
            # Per-metric analysis for each tool
            for j, metric_name in enumerate(self.metric_names):
                pred_metric = tool_preds[:, j]
                targ_metric = tool_targs[:, j]
                
                with np.errstate(invalid='ignore', divide='ignore'):
                    metric_mse = np.mean((pred_metric - targ_metric) ** 2)
                    metric_mae = np.mean(np.abs(pred_metric - targ_metric))
                    
                    # R2 score with safety checks
                    if np.var(targ_metric) > 1e-8 and len(targ_metric) > 1:
                        try:
                            from sklearn.metrics import r2_score
                            metric_r2 = r2_score(targ_metric, pred_metric)
                        except Exception:
                            metric_r2 = 0.0
                    else:
                        metric_r2 = 0.0
                    
                    # Store with safety checks
                    metrics[f'{tool_name}_{metric_name}_mse'] = float(metric_mse) if np.isfinite(metric_mse) else 0.0
                    metrics[f'{tool_name}_{metric_name}_mae'] = float(metric_mae) if np.isfinite(metric_mae) else 0.0
                    metrics[f'{tool_name}_{metric_name}_r2'] = float(metric_r2) if np.isfinite(metric_r2) else 0.0 
                    metrics[f'{tool_name}_rmse'] = tool_rmse
                    metrics[f'{tool_name}_mape'] = tool_mape
        
        # Per-metric averages across all tools
        for j, metric_name in enumerate(self.metric_names):
            all_preds_metric = preds_reshaped[:, :, j].ravel()
            all_targs_metric = targs_reshaped[:, :, j].ravel()
            
            with np.errstate(invalid='ignore', divide='ignore'):
                avg_mse = np.mean((all_preds_metric - all_targs_metric) ** 2)
                avg_mae = np.mean(np.abs(all_preds_metric - all_targs_metric)) 
                avg_rmse = compute_rmse(all_preds_metric, all_targs_metric)
                avg_mape = compute_mape(all_preds_metric, all_targs_metric)
                
                # Cross-tool R2 for this metric
                if np.var(all_targs_metric) > 1e-8 and len(all_targs_metric) > 1:
                    try:
                        avg_r2 = r2_score(all_targs_metric, all_preds_metric)
                    except Exception:
                        avg_r2 = 0.0
                else:
                    avg_r2 = 0.0
                
                metrics[f'{metric_name}_avg_mse'] = float(avg_mse) if np.isfinite(avg_mse) else 0.0
                metrics[f'{metric_name}_avg_mae'] = float(avg_mae) if np.isfinite(avg_mae) else 0.0
                metrics[f'{metric_name}_avg_r2'] = float(avg_r2) if np.isfinite(avg_r2) else 0.0 
                metrics[f'{metric_name}_avg_rmse'] = avg_rmse
                metrics[f'{metric_name}_avg_mape'] = avg_mape
        
        # Calculate F1 scores (derived from precision and recall)
        try:
            # Extract precision and recall predictions and targets
            precision_preds = preds_reshaped[:, :, 3].ravel()  # Precision is index 3
            recall_preds = preds_reshaped[:, :, 4].ravel()     # Recall is index 4
            precision_targs = targs_reshaped[:, :, 3].ravel()
            recall_targs = targs_reshaped[:, :, 4].ravel()
            
            # Calculate F1 scores
            with np.errstate(invalid='ignore', divide='ignore'):
                f1_preds = 2 * precision_preds * recall_preds / (precision_preds + recall_preds + 1e-8)
                f1_targs = 2 * precision_targs * recall_targs / (precision_targs + recall_targs + 1e-8)
                
                # F1 metrics
                f1_mse = np.mean((f1_preds - f1_targs) ** 2)
                f1_mae = np.mean(np.abs(f1_preds - f1_targs))
                
                metrics['f1_score_avg_mse'] = float(f1_mse) if np.isfinite(f1_mse) else 0.0
                metrics['f1_score_avg_mae'] = float(f1_mae) if np.isfinite(f1_mae) else 0.0
                
        except Exception as e:
            logger.warning(f"F1 score calculation failed: {e}")
            metrics['f1_score_avg_mse'] = 0.0
            metrics['f1_score_avg_mae'] = 0.0
        
        # Add detailed analysis if requested
        if detailed:
            try:
                detailed_errors = {}
                for i, tool_name in enumerate(tool_names_subset):
                    detailed_errors[tool_name] = {}
                    for j, metric_name in enumerate(self.metric_names):
                        pred_metric = preds_reshaped[:, i, j]
                        targ_metric = targs_reshaped[:, i, j]
                        errors = np.abs(pred_metric - targ_metric)
                        detailed_errors[tool_name][metric_name] = errors.tolist()
                
                metrics['detailed_errors'] = detailed_errors
                metrics['prediction_stats'] = self.calculate_prediction_statistics(preds_reshaped)
                metrics['target_stats'] = self.calculate_prediction_statistics(targs_reshaped)
                
            except Exception as e:
                logger.warning(f"Detailed analysis failed: {e}")
        
        # Clean up any NaN or inf values
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isfinite(value):
                metrics[key] = 0.0
        
        return metrics
    
    
    def calculate_prediction_statistics(self, predictions: np.ndarray) -> Dict:
        """Calculate statistics of predictions for analysis."""
        stats = {}
        
        # Global statistics
        stats['global_mean'] = float(np.mean(predictions))
        stats['global_std'] = float(np.std(predictions))
        stats['global_min'] = float(np.min(predictions))
        stats['global_max'] = float(np.max(predictions))
        
        # Per-tool statistics
        for i, tool_name in enumerate(self.tool_names):
            if i >= self.num_tools:
                continue
                
            tool_preds = predictions[:, i, :]
            
            stats[f'{tool_name}_mean'] = np.mean(tool_preds, axis=0).tolist()
            stats[f'{tool_name}_std'] = np.std(tool_preds, axis=0).tolist()
            stats[f'{tool_name}_min'] = np.min(tool_preds, axis=0).tolist()
            stats[f'{tool_name}_max'] = np.max(tool_preds, axis=0).tolist()
        
        # Per-metric statistics
        for j, metric_name in enumerate(self.metric_names):
            metric_preds = predictions[:, :, j]
            
            stats[f'{metric_name}_mean'] = float(np.mean(metric_preds))
            stats[f'{metric_name}_std'] = float(np.std(metric_preds))
            stats[f'{metric_name}_min'] = float(np.min(metric_preds))
            stats[f'{metric_name}_max'] = float(np.max(metric_preds))
        
        return stats
    
    def get_summary_metrics(self, all_metrics: Dict) -> Dict:
        """Get summary metrics for quick evaluation."""
        summary = {}
        
        # Tool performance summary
        summary['overall_mse'] = all_metrics.get('overall_mse', float('inf'))
        summary['overall_mae'] = all_metrics.get('overall_mae', float('inf')) 
        summary['overall_rmse'] = all_metrics.get('overall_rmse', float('inf'))
        summary['overall_mape'] = all_metrics.get('overall_mape', float('inf'))
        
        # Per-metric summaries
        for metric_name in self.metric_names:
            summary[f'{metric_name}_mse'] = all_metrics.get(f'{metric_name}_avg_mse', float('inf'))
            summary[f'{metric_name}_mae'] = all_metrics.get(f'{metric_name}_avg_mae', float('inf'))
        
        # Best and worst tool metrics
        tool_maes = [(tool_name, all_metrics.get(f'{tool_name}_mae', float('inf'))) 
                    for tool_name in self.tool_names]
        
        if tool_maes:
            best_tool = min(tool_maes, key=lambda x: x[1])
            worst_tool = max(tool_maes, key=lambda x: x[1])
            
            summary['best_tool'] = best_tool[0]
            summary['best_tool_mae'] = best_tool[1]
            summary['worst_tool'] = worst_tool[0]
            summary['worst_tool_mae'] = worst_tool[1]
        
        return summary
    
    def analyze_tool_correlations(self, predictions: torch.Tensor, 
                                 targets: torch.Tensor) -> Dict:
        """Analyze correlations between different tools and metrics."""
        # Convert to numpy
        if isinstance(predictions, tuple) and len(predictions) == 2:
            all_preds, metric_preds = predictions
            preds = all_preds.detach().cpu().numpy()
        else:
            preds = predictions.detach().cpu().numpy()
        
        targs = targets.detach().cpu().numpy()
        
        # Reshape to (batch_size, num_tools, num_metrics)
        preds = preds.reshape(-1, self.num_tools, 5)
        targs = targs.reshape(-1, self.num_tools, 5)
        
        analysis = {}
        
        # Tool-to-tool prediction correlation
        tool_corr_matrix = np.zeros((self.num_tools, self.num_tools))
        for i in range(self.num_tools):
            for j in range(self.num_tools):
                # Correlation of the average error per sample
                tool_i_error = np.mean(np.abs(preds[:, i, :] - targs[:, i, :]), axis=1)
                tool_j_error = np.mean(np.abs(preds[:, j, :] - targs[:, j, :]), axis=1)
                
                tool_corr_matrix[i, j] = np.corrcoef(tool_i_error, tool_j_error)[0, 1]
        
        analysis['tool_error_correlation'] = tool_corr_matrix.tolist()
        
        # Metric-to-metric prediction correlation
        metric_corr_matrix = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                # Correlation of the average error per sample across all tools
                metric_i_error = np.mean(np.abs(preds[:, :, i] - targs[:, :, i]), axis=1)
                metric_j_error = np.mean(np.abs(preds[:, :, j] - targs[:, :, j]), axis=1)
                
                metric_corr_matrix[i, j] = np.corrcoef(metric_i_error, metric_j_error)[0, 1]
        
        analysis['metric_error_correlation'] = metric_corr_matrix.tolist()
        
        return analysis
    
    def analyze_error_distribution(self, predictions: torch.Tensor, 
                                  targets: torch.Tensor) -> Dict:
        """Analyze the distribution of prediction errors."""
        # Convert to numpy
        if isinstance(predictions, tuple) and len(predictions) == 2:
            all_preds, metric_preds = predictions
            preds = all_preds.detach().cpu().numpy()
        else:
            preds = predictions.detach().cpu().numpy()
        
        targs = targets.detach().cpu().numpy()
        
        # Reshape to (batch_size, num_tools, num_metrics)
        preds = preds.reshape(-1, self.num_tools, 5)
        targs = targs.reshape(-1, self.num_tools, 5)
        
        analysis = {}
        
        # Calculate raw errors
        errors = preds - targs
        
        # Overall error distribution
        analysis['overall_error_mean'] = float(np.mean(errors))
        analysis['overall_error_std'] = float(np.std(errors))
        analysis['overall_error_percentiles'] = np.percentile(errors, [0, 25, 50, 75, 100]).tolist()
        
        # Per-tool error distribution
        for i, tool_name in enumerate(self.tool_names):
            if i >= self.num_tools:
                continue
                
            tool_errors = errors[:, i, :]
            
            analysis[f'{tool_name}_error_mean'] = np.mean(tool_errors, axis=0).tolist()
            analysis[f'{tool_name}_error_std'] = np.std(tool_errors, axis=0).tolist()
            analysis[f'{tool_name}_error_abs_mean'] = np.mean(np.abs(tool_errors), axis=0).tolist()
        
        # Per-metric error distribution
        for j, metric_name in enumerate(self.metric_names):
            metric_errors = errors[:, :, j]
            
            analysis[f'{metric_name}_error_mean'] = float(np.mean(metric_errors))
            analysis[f'{metric_name}_error_std'] = float(np.std(metric_errors))
            analysis[f'{metric_name}_error_abs_mean'] = float(np.mean(np.abs(metric_errors)))
            
            # Check for systematic bias
            if np.mean(metric_errors) > 0.05:
                analysis[f'{metric_name}_bias'] = "Tends to overestimate"
            elif np.mean(metric_errors) < -0.05:
                analysis[f'{metric_name}_bias'] = "Tends to underestimate"
            else:
                analysis[f'{metric_name}_bias'] = "No significant bias"
        
        return analysis
    
    def create_performance_report(self, metrics: Dict) -> str:
        """Create a human-readable performance report."""
        report = "# Tool Performance Prediction Report\n\n"
        
        # Overall performance
        report += "## Overall Performance\n\n"
        report += f"- Mean Absolute Error: {metrics.get('overall_mae', 'N/A'):.4f}\n"
        report += f"- Mean Squared Error: {metrics.get('overall_mse', 'N/A'):.4f}\n\n"
        
        # Per-metric performance
        report += "## Performance by Metric\n\n"
        report += "| Metric | MAE | MSE |\n"
        report += "|--------|-----|-----|\n"
        
        for metric_name in self.metric_names:
            mae = metrics.get(f'{metric_name}_avg_mae', float('inf'))
            mse = metrics.get(f'{metric_name}_avg_mse', float('inf'))
            report += f"| {metric_name.upper()} | {mae:.4f} | {mse:.4f} |\n"
        
        report += "\n"
        
        # Per-tool performance
        report += "## Performance by Tool\n\n"
        report += "| Tool | MAE | MSE | R^2 |\n"
        report += "|------|-----|-----|----|\n"
        
        for tool_name in self.tool_names:
            mae = metrics.get(f'{tool_name}_mae', float('inf'))
            mse = metrics.get(f'{tool_name}_mse', float('inf'))
            r2 = metrics.get(f'{tool_name}_r2', float('-inf'))
            report += f"| {tool_name.title()} | {mae:.4f} | {mse:.4f} | {r2:.4f} |\n"
        
        report += "\n"
        
        # Detailed tool-metric breakdown
        report += "## Detailed Performance Breakdown\n\n"
        for tool_name in self.tool_names:
            report += f"### {tool_name.title()}\n\n"
            report += "| Metric | MAE | MSE |\n"
            report += "|--------|-----|-----|\n"
            
            for metric_name in self.metric_names:
                mae = metrics.get(f'{tool_name}_{metric_name}_mae', float('inf'))
                mse = metrics.get(f'{tool_name}_{metric_name}_mse', float('inf'))
                report += f"| {metric_name.upper()} | {mae:.4f} | {mse:.4f} |\n"
            
            report += "\n"
        
        return  
    
    
    def calculate_per_vulnerability_metrics(self, detailed_confusion: Dict) -> Dict:
        """Calculate detailed metrics per tool and per vulnerability type."""
        vulnerability_metrics = {}
        
        for tool, tool_data in detailed_confusion.items():
            vulnerability_metrics[tool] = {}
            
            for vuln_type, confusion_data in tool_data.items():
                tp = confusion_data['TP']
                fp = confusion_data['FP']
                fn = confusion_data['FN']
                tn = confusion_data['TN']
                
                # Calculate metrics with epsilon for numerical stability
                epsilon = 1e-8
                precision = tp / (tp + fp + epsilon)
                recall = tp / (tp + fn + epsilon)
                f1_score = 2 * precision * recall / (precision + recall + epsilon)
                accuracy = (tp + tn) / (tp + fp + fn + tn + epsilon)
                specificity = tn / (tn + fp + epsilon)
                
                vulnerability_metrics[tool][vuln_type] = {
                    'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'accuracy': accuracy,
                    'specificity': specificity,
                    'false_positive_rate': fp / (fp + tn + epsilon),
                    'false_negative_rate': fn / (fn + tp + epsilon),
                    'contracts_with_vuln': confusion_data['contracts_with_vuln'],
                    'contracts_detected': confusion_data['contracts_detected'],
                    'false_positive_contracts': confusion_data['false_positive_contracts'],
                    'false_negative_contracts': confusion_data['false_negative_contracts']
                }
        
        return vulnerability_metrics

    def create_vulnerability_performance_report(self, vulnerability_metrics: Dict) -> str:
        """Create a detailed report of FP/FN per tool and vulnerability."""
        report = "# Detailed Tool Performance Analysis: False Positives and False Negatives\n\n"
        
        vulnerability_types = [
            'reentrancy', 'integer_overflow', 'unchecked_return',
            'timestamp_dependency', 'tx_origin', 'unhandled_exception', 'tod'
        ]
        
        tools = ['oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither']
        
        # Overall summary table
        report += "## Summary: False Positive and False Negative Rates\n\n"
        report += "| Tool | Vulnerability | TP | FP | FN | TN | FPR | FNR | Precision | Recall | F1 |\n"
        report += "|------|---------------|----|----|----|----|-----|-----|-----------|--------|----|\n"
        
        for tool in tools:
            if tool not in vulnerability_metrics:
                continue
                
            for vuln_type in vulnerability_types:
                if vuln_type not in vulnerability_metrics[tool]:
                    continue
                    
                metrics = vulnerability_metrics[tool][vuln_type]
                report += f"| {tool.title()} | {vuln_type.replace('_', ' ').title()} | "
                report += f"{metrics['TP']} | {metrics['FP']} | {metrics['FN']} | {metrics['TN']} | "
                report += f"{metrics['false_positive_rate']:.3f} | {metrics['false_negative_rate']:.3f} | "
                report += f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1_score']:.3f} |\n"
        
        report += "\n\n"
        
        # Detailed analysis per tool
        for tool in tools:
            if tool not in vulnerability_metrics:
                continue
                
            report += f"## {tool.title()} - Detailed Analysis\n\n"
            
            # Find worst performing vulnerabilities
            worst_fp = max(vulnerability_metrics[tool].items(), 
                        key=lambda x: x[1]['false_positive_rate'])
            worst_fn = max(vulnerability_metrics[tool].items(), 
                        key=lambda x: x[1]['false_negative_rate'])
            
            report += f"**Highest False Positive Rate:** {worst_fp[0]} ({worst_fp[1]['false_positive_rate']:.3f})\n"
            report += f"**Highest False Negative Rate:** {worst_fn[0]} ({worst_fn[1]['false_negative_rate']:.3f})\n\n"
            
            # List problematic contracts
            for vuln_type in vulnerability_types:
                if vuln_type not in vulnerability_metrics[tool]:
                    continue
                    
                metrics = vulnerability_metrics[tool][vuln_type]
                if metrics['FP'] > 0 or metrics['FN'] > 0:
                    report += f"### {vuln_type.replace('_', ' ').title()}\n\n"
                    
                    if metrics['false_positive_contracts']:
                        report += f"**False Positive Contracts ({len(metrics['false_positive_contracts'])}):**\n"
                        for contract in metrics['false_positive_contracts'][:10]:  # Show first 10
                            report += f"- {contract}\n"
                        if len(metrics['false_positive_contracts']) > 10:
                            report += f"- ... and {len(metrics['false_positive_contracts']) - 10} more\n"
                        report += "\n"
                    
                    if metrics['false_negative_contracts']:
                        report += f"**False Negative Contracts ({len(metrics['false_negative_contracts'])}):**\n"
                        for contract in metrics['false_negative_contracts'][:10]:  # Show first 10
                            report += f"- {contract}\n"
                        if len(metrics['false_negative_contracts']) > 10:
                            report += f"- ... and {len(metrics['false_negative_contracts']) - 10} more\n"
                        report += "\n"
        
        return report
    

def calculate_confidence_intervals(predictions: torch.Tensor, 
                                 targets: torch.Tensor, 
                                 n_bootstrap: int = 1000,
                                 confidence: float = 0.95) -> Dict:
    """Calculate confidence intervals for tool metrics using bootstrapping."""
    # Convert to numpy
    if isinstance(predictions, tuple) and len(predictions) == 2:
        all_preds, metric_preds = predictions
        preds = all_preds.detach().cpu().numpy()
    else:
        preds = predictions.detach().cpu().numpy()
    
    targs = targets.detach().cpu().numpy()
    
    num_samples = preds.shape[0]
    num_tools = preds.shape[1] // 5
    
    # Reshape to (batch_size, num_tools, num_metrics)
    preds = preds.reshape(num_samples, num_tools, 5)
    targs = targs.reshape(num_samples, num_tools, 5)
    
    tool_names = ['oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither']
    metric_names = ['tpr', 'fpr', 'accuracy', 'precision', 'recall']
    
    # Bootstrap samples
    bootstrap_metrics = {
        'overall_mae': [],
        'overall_mse': []
    }
    
    # Initialize per-tool and per-metric bootstrap lists
    for tool_name in tool_names[:num_tools]:
        bootstrap_metrics[f'{tool_name}_mae'] = []
        bootstrap_metrics[f'{tool_name}_mse'] = []
        
        for metric_name in metric_names:
            bootstrap_metrics[f'{tool_name}_{metric_name}_mae'] = []
    
    for metric_name in metric_names:
        bootstrap_metrics[f'{metric_name}_avg_mae'] = []
    
    # Generate bootstrap samples
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(num_samples, num_samples, replace=True)
        bootstrap_preds = preds[indices]
        bootstrap_targs = targs[indices]
        
        # Calculate metrics for this bootstrap sample
        bootstrap_metrics['overall_mae'].append(np.mean(np.abs(bootstrap_preds - bootstrap_targs)))
        bootstrap_metrics['overall_mse'].append(np.mean((bootstrap_preds - bootstrap_targs) ** 2))
        
        # Per-tool metrics
        for i, tool_name in enumerate(tool_names[:num_tools]):
            tool_preds = bootstrap_preds[:, i, :]
            tool_targs = bootstrap_targs[:, i, :]
            
            bootstrap_metrics[f'{tool_name}_mae'].append(np.mean(np.abs(tool_preds - tool_targs)))
            bootstrap_metrics[f'{tool_name}_mse'].append(np.mean((tool_preds - tool_targs) ** 2))
            
            # Per-metric
            for j, metric_name in enumerate(metric_names):
                pred_metric = tool_preds[:, j]
                targ_metric = tool_targs[:, j]
                
                bootstrap_metrics[f'{tool_name}_{metric_name}_mae'].append(
                    np.mean(np.abs(pred_metric - targ_metric))
                )
        
        # Per-metric averages
        for j, metric_name in enumerate(metric_names):
            all_preds = bootstrap_preds[:, :, j].ravel()
            all_targs = bootstrap_targs[:, :, j].ravel()
            
            bootstrap_metrics[f'{metric_name}_avg_mae'].append(
                np.mean(np.abs(all_preds - all_targs))
            )
    
    # Calculate confidence intervals
    confidence_intervals = {}
    alpha = (1 - confidence) / 2
    percentiles = [100 * alpha, 100 * (1 - alpha)]
    
    for metric_name, bootstrap_values in bootstrap_metrics.items():
        lower, upper = np.percentile(bootstrap_values, percentiles)
        confidence_intervals[metric_name] = {
            'lower': float(lower),
            'upper': float(upper),
            'mean': float(np.mean(bootstrap_values))
        }
    
    return confidence_intervals

def compute_tool_correlations(predictions: torch.Tensor, targets: torch.Tensor) -> Dict:
    """Compute correlations between predicted and actual tool performances."""
    if isinstance(predictions, tuple) and len(predictions) == 2:
        all_preds, metric_preds = predictions
        preds = all_preds.detach().cpu().numpy()
    else:
        preds = predictions.detach().cpu().numpy()
    
    targs = targets.detach().cpu().numpy()
    
    # Reshape to (batch_size, num_tools * 5)
    if len(preds.shape) > 2:
        preds = preds.reshape(preds.shape[0], -1)
    if len(targs.shape) > 2:
        targs = targs.reshape(targs.shape[0], -1)
    
    num_tools = preds.shape[1] // 5
    tool_names = ['oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither'][:num_tools]
    metric_names = ['tpr', 'fpr', 'accuracy', 'precision', 'recall']
    
    correlations = {}
    
    # Calculate correlation for each tool-metric combination
    for i, tool_name in enumerate(tool_names):
        tool_correlations = {}
        
        for j, metric_name in enumerate(metric_names):
            idx = i * 5 + j
            pred_values = preds[:, idx]
            targ_values = targs[:, idx]
            
            # Calculate correlation
            corr = np.corrcoef(pred_values, targ_values)[0, 1]
            tool_correlations[metric_name] = float(corr)
        
        correlations[tool_name] = tool_correlations
        
        # Calculate overall correlation for this tool (average across metrics)
        tool_preds = preds[:, i*5:(i+1)*5]
        tool_targs = targs[:, i*5:(i+1)*5]
        
        # Flatten and calculate correlation
        tool_preds_flat = tool_preds.flatten()
        tool_targs_flat = tool_targs.flatten()
        
        overall_corr = np.corrcoef(tool_preds_flat, tool_targs_flat)[0, 1]
        correlations[f'{tool_name}_overall'] = float(overall_corr)
    
    # Overall correlation across all tools and metrics
    overall_corr = np.corrcoef(preds.flatten(), targs.flatten())[0, 1]
    correlations['overall'] = float(overall_corr)
    
    return correlations

def identify_outliers(predictions: torch.Tensor, targets: torch.Tensor, 
                    threshold: float = 3.0) -> Dict:
    """Identify outliers in prediction errors."""
    if isinstance(predictions, tuple) and len(predictions) == 2:
        all_preds, metric_preds = predictions
        preds = all_preds.detach().cpu().numpy()
    else:
        preds = predictions.detach().cpu().numpy()
    
    targs = targets.detach().cpu().numpy()
    
    # Reshape to (batch_size, num_tools, num_metrics)
    batch_size = preds.shape[0]
    num_tools = preds.shape[1] // 5
    
    preds = preds.reshape(batch_size, num_tools, 5)
    targs = targs.reshape(batch_size, num_tools, 5)
    
    # Calculate absolute errors
    abs_errors = np.abs(preds - targs)
    
    # Calculate mean and std of errors
    mean_error = np.mean(abs_errors)
    std_error = np.std(abs_errors)
    
    # Identify outliers (errors > mean + threshold * std)
    outlier_threshold = mean_error + threshold * std_error
    
    outliers = {}
    for i in range(batch_size):
        sample_outliers = {}
        
        for t in range(num_tools):
            tool_outliers = []
            
            for m in range(5):
                if abs_errors[i, t, m] > outlier_threshold:
                    tool_outliers.append({
                        'metric_idx': m,
                        'pred': float(preds[i, t, m]),
                        'target': float(targs[i, t, m]),
                        'error': float(abs_errors[i, t, m])
                    })
            
            if tool_outliers:
                sample_outliers[f'tool_{t}'] = tool_outliers
        
        if sample_outliers:
            outliers[f'sample_{i}'] = sample_outliers
    
    # Summary statistics
    num_outliers = sum(len(sample_outliers) for sample_outliers in outliers.values())
    total_predictions = batch_size * num_tools * 5
    
    summary = {
        'num_outliers': num_outliers,
        'total_predictions': total_predictions,
        'outlier_percentage': num_outliers / total_predictions * 100,
        'outlier_threshold': float(outlier_threshold),
        'mean_error': float(mean_error),
        'std_error': float(std_error)
    }
    
    return {'summary': summary, 'outliers': outliers} 


def analyze_fp_fn_root_causes(self, detailed_confusion: Dict, contracts_data: List[Dict]) -> Dict:
    """Analyze root causes of false positives and false negatives."""
    root_causes = {
        'false_positive_patterns': {},
        'false_negative_patterns': {},
        'recommendations': {}
    }
    
    # Analyze contract characteristics that lead to FPs/FNs
    for tool, tool_data in detailed_confusion.items():
        root_causes['false_positive_patterns'][tool] = {}
        root_causes['false_negative_patterns'][tool] = {}
        root_causes['recommendations'][tool] = []
        
        for vuln_type, confusion_data in tool_data.items():
            # Analyze FP contracts
            fp_contracts = confusion_data['false_positive_contracts']
            fn_contracts = confusion_data['false_negative_contracts']
            
            # Find common patterns in FP contracts
            fp_characteristics = self._analyze_contract_characteristics(fp_contracts, contracts_data)
            fn_characteristics = self._analyze_contract_characteristics(fn_contracts, contracts_data)
            
            root_causes['false_positive_patterns'][tool][vuln_type] = fp_characteristics
            root_causes['false_negative_patterns'][tool][vuln_type] = fn_characteristics
            
            # Generate recommendations
            if confusion_data['FP'] > 5:  # High FP count
                root_causes['recommendations'][tool].append(
                    f"High false positives for {vuln_type}: Consider refining detection rules"
                )
            
            if confusion_data['FN'] > 5:  # High FN count
                root_causes['recommendations'][tool].append(
                    f"High false negatives for {vuln_type}: Consider improving coverage"
                )
    
    return root_causes

def _analyze_contract_characteristics(self, contract_ids: List[str], contracts_data: List[Dict]) -> Dict:
    """Analyze common characteristics of problematic contracts."""
    characteristics = {
        'avg_code_length': 0,
        'common_patterns': [],
        'complexity_metrics': {}
    }
    
    if not contract_ids:
        return characteristics
    
    # Find matching contracts
    matching_contracts = [c for c in contracts_data if c['id'] in contract_ids]
    
    if matching_contracts:
        code_lengths = [len(c.get('source_code', '')) for c in matching_contracts]
        characteristics['avg_code_length'] = sum(code_lengths) / len(code_lengths)
        
        # Find common code patterns
        all_code = ' '.join([c.get('source_code', '') for c in matching_contracts])
        
        # Common vulnerability-related patterns
        patterns = {
            'external_calls': all_code.count('.call('),
            'require_statements': all_code.count('require('),
            'assert_statements': all_code.count('assert('),
            'modifier_usage': all_code.count('modifier'),
            'assembly_blocks': all_code.count('assembly {'),
            'delegatecall_usage': all_code.count('delegatecall')
        }
        
        characteristics['common_patterns'] = [
            f"{pattern}: {count}" for pattern, count in patterns.items() if count > 0
        ]
    
    return characteristics
