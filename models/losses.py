import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional, Union

class ToolPerformanceLoss(nn.Module):
    """Loss function for tool performance prediction."""
    
    def __init__(self, config: Dict):
        super(ToolPerformanceLoss, self).__init__()
        self.num_tools = config['model']['num_tools']
        
        # Base loss functions
        self.mse_loss = nn.MSELoss()
        
        # Metric weights (increased weight for FPR)
        metric_weights = config.get('metric_weights', {
            'tpr': 1.0,
            'fpr': 2.5,  # Increased weight for FPR
            'accuracy': 1.0,
            'precision': 1.0,
            'recall': 1.0
        })
        
        # Tool weights (increased weight for Securify)
        tool_weights = config.get('tool_weights', {
            'oyente': 1.0,
            'securify': 2.0,  # Increased weight for Securify
            'mythril': 1.0,
            'smartcheck': 1.0,
            'manticore': 1.0,
            'slither': 1.0
        })
        
        self.metric_weights = metric_weights
        self.tool_weights = tool_weights
        
        # Tool names for indexing
        self.tool_names = ['oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither']
        
    
    def forward(self, predictions: Union[torch.Tensor, Tuple], targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Fixed loss computation with proper shape handling."""
        
        # Handle different prediction formats
        if isinstance(predictions, tuple) and len(predictions) == 2:
            all_predictions, tool_predictions = predictions
        else:
            all_predictions = predictions
            tool_predictions = None
        
        # Ensure tensors are on the same device
        if all_predictions.device != targets.device:
            targets = targets.to(all_predictions.device)
        
        # CRITICAL FIX: Ensure compatible shapes
        batch_size = min(all_predictions.shape[0], targets.shape[0])
        expected_features = self.num_tools * 5  # 6 tools * 5 metrics = 30
        
        # Crop to matching batch size
        predictions_tensor = all_predictions[:batch_size]
        targets_tensor = targets[:batch_size]
        
        # Handle feature dimension mismatches
        if predictions_tensor.shape[1] != expected_features:
            if predictions_tensor.shape[1] > expected_features:
                predictions_tensor = predictions_tensor[:, :expected_features]
            else:
                # Calculate appropriate padding value from existing data
                padding_size = expected_features - predictions_tensor.shape[1]
                padding = torch.full((batch_size, padding_size), padding_value,      #Replace 0.5 with  padding_value
                                device=predictions_tensor.device, dtype=predictions_tensor.dtype)
                predictions_tensor = torch.cat([predictions_tensor, padding], dim=1) 

        if targets_tensor.shape[1] != expected_features:
            if targets_tensor.shape[1] > expected_features:
                targets_tensor = targets_tensor[:, :expected_features]
            else:
                # Calculate appropriate padding value from existing data
                padding_size = expected_features - targets_tensor.shape[1]
                padding = torch.full((batch_size, padding_size), padding_value,      #Replace 0.5 with  padding_value
                                device=targets_tensor.device, dtype=targets_tensor.dtype)
                targets_tensor = torch.cat([targets_tensor, padding], dim=1)
        
        # Reshape to (batch_size, num_tools, num_metrics)
        preds_reshaped = predictions_tensor.view(batch_size, self.num_tools, 5)
        targets_reshaped = targets_tensor.view(batch_size, self.num_tools, 5)
        
        # Extract metric predictions
        pred_tpr = preds_reshaped[:, :, 0]      # Shape: (batch_size, num_tools)
        pred_fpr = preds_reshaped[:, :, 1]
        pred_accuracy = preds_reshaped[:, :, 2]
        pred_precision = preds_reshaped[:, :, 3]
        pred_recall = preds_reshaped[:, :, 4]
        
        # Extract metric targets
        target_tpr = targets_reshaped[:, :, 0]
        target_fpr = targets_reshaped[:, :, 1]
        target_accuracy = targets_reshaped[:, :, 2]
        target_precision = targets_reshaped[:, :, 3]
        target_recall = targets_reshaped[:, :, 4]
        
        # Compute losses for each metric
        losses = {}
        total_loss = torch.tensor(0.0, device=predictions_tensor.device, requires_grad=True)
        
        # TPR loss
        tpr_loss = self.mse_loss(pred_tpr, target_tpr)
        losses['tpr'] = tpr_loss
        total_loss = total_loss + self.metric_weights['tpr'] * tpr_loss
        
        # FPR loss (higher weight)
        fpr_loss = self.mse_loss(pred_fpr, target_fpr)
        losses['fpr'] = fpr_loss
        total_loss = total_loss + self.metric_weights['fpr'] * fpr_loss
        
        # Accuracy loss
        accuracy_loss = self.mse_loss(pred_accuracy, target_accuracy)
        losses['accuracy'] = accuracy_loss
        total_loss = total_loss + self.metric_weights['accuracy'] * accuracy_loss
        
        # Precision loss
        precision_loss = self.mse_loss(pred_precision, target_precision)
        losses['precision'] = precision_loss
        total_loss = total_loss + self.metric_weights['precision'] * precision_loss
        
        # Recall loss
        recall_loss = self.mse_loss(pred_recall, target_recall)
        losses['recall'] = recall_loss
        total_loss = total_loss + self.metric_weights['recall'] * recall_loss
        
        # Tool-specific losses
        for tool_idx, tool_name in enumerate(self.tool_names):
            if tool_idx >= self.num_tools:
                continue
            
            tool_weight = self.tool_weights.get(tool_name, 1.0)
            
            # Extract predictions and targets for this tool
            tool_pred = preds_reshaped[:, tool_idx, :]  # Shape: (batch_size, 5)
            tool_target = targets_reshaped[:, tool_idx, :]
            
            # Compute tool-specific loss
            tool_loss = self.mse_loss(tool_pred, tool_target)
            tool_loss_weighted = tool_weight * tool_loss
            
            total_loss = total_loss + tool_loss_weighted
            losses[f'{tool_name}_loss'] = tool_loss_weighted
        
        # Special emphasis on Securify FPR
        securify_idx = 1  # Assuming Securify is index 1
        if securify_idx < self.num_tools:
            securify_fpr_pred = pred_fpr[:, securify_idx]
            securify_fpr_target = target_fpr[:, securify_idx]
            
            securify_fpr_loss = self.mse_loss(securify_fpr_pred, securify_fpr_target)
            extra_securify_loss = 1.5 * securify_fpr_loss  # Extra emphasis
            
            total_loss = total_loss + extra_securify_loss
            losses['extra_securify_fpr'] = extra_securify_loss
        
        # Add regularization to prevent extreme predictions
        pred_mean = torch.mean(predictions_tensor)
        pred_std = torch.std(predictions_tensor)
        
        # Encourage predictions to stay in reasonable range [0, 1]
        range_penalty = torch.mean(torch.clamp(predictions_tensor - 1.0, min=0.0)) + \
                    torch.mean(torch.clamp(-predictions_tensor, min=0.0))
        
        total_loss = total_loss + 0.1 * range_penalty
        losses['range_penalty'] = range_penalty
        
        losses['total'] = total_loss
        return total_loss, losses    

    
    def compute_per_tool_losses(self, predictions: Dict, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute per-tool losses for analysis."""
        tool_losses = {}
        
        # Extract targets for each metric
        batch_size = targets.shape[0]
        targets_reshaped = targets.view(batch_size, self.num_tools, 5) 
        # targets_reshaped = targets.view(-1, self.num_tools, 5)
        
        for metric, pred in predictions.items():
            metric_idx = {'tpr': 0, 'fpr': 1, 'accuracy': 2, 'precision': 3, 'recall': 4}
            if metric in metric_idx:
                target = targets_reshaped[:, :, metric_idx[metric]]
                
                # Compute per-tool losses
                for tool_idx in range(self.num_tools):
                    tool_pred = pred[:, tool_idx]
                    tool_target = target[:, tool_idx]
                    
                    # Ensure compatible shapes
                    if tool_pred.shape != tool_target.shape:
                        min_size = min(tool_pred.shape[0], tool_target.shape[0])
                        tool_pred = tool_pred[:min_size]
                        tool_target = tool_target[:min_size]
                        
                    tool_loss = F.mse_loss(tool_pred, tool_target, reduction='mean')
                    
                    tool_name = self.tool_names[tool_idx] if tool_idx < len(self.tool_names) else f'tool{tool_idx}'
                    tool_losses[f'{metric}_{tool_name}'] = tool_loss
                    
                    # Special handling for Securify's FPR
                    if metric == 'fpr' and tool_name == 'securify':
                        # Additional analysis metrics for Securify's FPR
                        securify_fpr_under_pred = torch.mean((tool_target - tool_pred).clamp(min=0))
                        securify_fpr_over_pred = torch.mean((tool_pred - tool_target).clamp(min=0))
                        
                        tool_losses['securify_fpr_under_prediction'] = securify_fpr_under_pred
                        tool_losses['securify_fpr_over_prediction'] = securify_fpr_over_pred
        
        return tool_losses