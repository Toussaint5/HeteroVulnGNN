# -*- coding: utf-8 -*-

import os
import sys
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.data import HeteroData
import wandb
from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import json 
import pandas as pd
import seaborn as sns
from torchinfo import summary

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.loader import SolidiFIDataLoader
from data.processor import SolidityCodeProcessor
from data.graph_builder import HeterogeneousGraphBuilder
from models.heterognn import HeteroToolGNN
from models.losses import ToolPerformanceLoss
from utils.metrics import ToolMetricsCalculator
from utils.helpers import EarlyStopping, ModelCheckpoint, set_seed, count_parameters 
from utils.visualization import plot_tool_confusion_matrix
# from config import config

# Load config first to check debug settings
if len(sys.argv) > 1:
    try:
        with open(sys.argv[2] if '--config' in sys.argv else 'config/config.yaml', 'r') as f:
            temp_config = yaml.safe_load(f)
        if temp_config.get('advanced', {}).get('enable_gradient_anomaly_detection', False):
            torch.autograd.set_detect_anomaly(True)
    except:
        pass  # If config loading fails, continue without anomaly detection


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeteroToolDataset(torch.utils.data.Dataset):
    """Fixed dataset for tool performance prediction."""
    
    def __init__(self, contracts: List[Dict], processor: SolidityCodeProcessor,
                 graph_builder: HeterogeneousGraphBuilder, tool_results: Dict,
                 injected_bugs: Dict, performance_metrics: Dict):
        self.contracts = contracts
        self.processor = processor
        self.graph_builder = graph_builder
        self.tool_results = tool_results
        self.injected_bugs = injected_bugs
        self.performance_metrics = performance_metrics 
        
        # Filter valid contracts - FIXED: more lenient filtering
        self.valid_contracts = []
        for contract in contracts:
            contract_id = contract['id']
            # More lenient check - just need the contract to exist in our data
            has_tool_results = contract_id in tool_results
            has_injected_bugs = contract_id in injected_bugs
            
            # Debug logging
            if not has_tool_results:
                logger.debug(f"Contract {contract_id} missing tool results")
            if not has_injected_bugs:
                logger.debug(f"Contract {contract_id} missing injected bugs")
            
            # Accept contracts if they have either tool results OR injected bugs
            # (we can create synthetic data for missing parts)
            if has_tool_results or has_injected_bugs:
                self.valid_contracts.append(contract)
            else:
                logger.debug(f"Skipping contract {contract_id} - no data available")
        
        logger.info(f"Dataset created with {len(self.valid_contracts)} valid contracts out of {len(contracts)}")
        
        # If still no valid contracts, use all contracts and create synthetic data
        if len(self.valid_contracts) == 0:
            logger.warning("No contracts passed validation, using all contracts with synthetic data")
            self.valid_contracts = contracts
        
    def __len__(self):
        return len(self.valid_contracts)
    
    def __getitem__(self, idx):
        contract = self.valid_contracts[idx]
        contract_id = contract['id']
        
        try:
            # Extract AST features
            ast_features = self.processor.extract_ast_features(contract.get('ast', {}))
            
            # Build heterogeneous graph
            hetero_data = self.graph_builder.build_heterogeneous_graph(
                contract, ast_features, self.processor
            )
            
            # Create tool performance labels 
            try:
                tool_labels = self.processor.create_tool_performance_labels(
                    contract_id, 
                    self.injected_bugs, 
                    self.tool_results, 
                    self.performance_metrics
                )
            except Exception as e:
                logger.debug(f"Error creating labels for {contract_id}: {e}")
                # Create default labels
                tool_labels = torch.full((30,), 0.5)  # Default values
            
            # Ensure tool_labels is the right shape (30,) not (6,5)
            if tool_labels.dim() == 2:
                tool_labels = tool_labels.flatten()
            elif tool_labels.dim() == 0:
                tool_labels = tool_labels.unsqueeze(0).expand(30)
            
            # Ensure we have exactly 30 features
            if tool_labels.shape[0] < 30:
                padding = torch.full((30 - tool_labels.shape[0],), 0.5)
                tool_labels = torch.cat([tool_labels, padding])
            elif tool_labels.shape[0] > 30:
                tool_labels = tool_labels[:30]
            
            # Add labels to graph
            hetero_data['contract'].y_tool = tool_labels.unsqueeze(0)  # Shape: (1, 30)
            
            return hetero_data
            
        except Exception as e:
            logger.error(f"Error processing contract {contract_id}: {e}")
            return self._create_fallback_graph(contract)
    
    def _create_fallback_graph(self, contract: Dict) -> HeteroData:
        """Create a minimal valid graph as fallback."""
        hetero_data = HeteroData()
        
        # Create minimal nodes
        hetero_data['function'].x = torch.zeros(1, 21)
        hetero_data['function'].node_ids = torch.tensor([0])
        
        # Contract features
        hetero_data['contract'].x = torch.zeros(1, 32)
        
        # Default tool performance labels (30 values)
        # hetero_data['contract'].y_tool = torch.full((1, 30), 0.5) 
        
        # Use actual empirical metrics instead of hardcoded 0.5
        try:
            tool_labels = self.processor.create_tool_performance_labels(
                contract.get('id', 'unknown'),
                self.injected_bugs,
                self.tool_results,
                self.performance_metrics
            )
        except Exception as e:
            logger.warning(f"Failed to create empirical labels, using neutral defaults: {e}")
            tool_labels = torch.full((30,), 0.5)
        
        hetero_data['contract'].y_tool = tool_labels.unsqueeze(0)
        
        return hetero_data

class ToolPerformanceTrainer:
    """Trainer for tool performance prediction with F1 score tracking."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Set random seed for reproducibility
        set_seed(42)
        
        # Initialize components
        self.data_loader = SolidiFIDataLoader(config)
        self.processor = SolidityCodeProcessor(config)
        self.graph_builder = HeterogeneousGraphBuilder(config)
        self.metrics_calculator = ToolMetricsCalculator()
        
        # Initialize model
        self.model = HeteroToolGNN(config).to(self.device)
        self.criterion = ToolPerformanceLoss(config) 
            
        # Try to get torchinfo summary with sample input
        try:
            # Create a sample heterogeneous graph input
            sample_batch = self._create_sample_input()
            if sample_batch is not None:
                # Move to device
                sample_batch = sample_batch.to(self.device)
                
                # Test forward pass
                with torch.no_grad():
                    output = self.model(sample_batch)
                    logger.info(f"Model output shape: {output[0].shape if isinstance(output, tuple) else output.shape}")
                
        except Exception as e:
            logger.warning(f"Could not create sample input for detailed analysis: {e}")
        
        # Log model parameters
        logger.info(f"Model parameters: {count_parameters(self.model):,}") 
        
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Use cosine annealing with warm restarts if configured
        if config['training'].get('use_scheduler', False):
            scheduler_type = config['training'].get('scheduler_type', 'reduce_on_plateau')
            
            if scheduler_type == 'cosine_warmup':
                from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
                self.scheduler = CosineAnnealingWarmRestarts(
                    self.optimizer, 
                    T_0=config['training'].get('warmup_epochs', 10),
                    T_mult=2,
                    eta_min=config['training'].get('min_lr', 1e-5)
                )
            else:
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', patience=5, factor=0.5
                )
        else:
            self.scheduler = None
        
        # Initialize utilities
        self.early_stopping = EarlyStopping(
            patience=config['training']['early_stopping_patience'],
            min_delta=0.001
        )
        self.model_checkpoint = ModelCheckpoint(save_dir='checkpoints')
        
        # Training history tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        # Logging
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(
                project=config['logging'].get('wandb_project', 'heterotoolgnn'),
                config=config,
                name=f"tool_performance_{config['training']['learning_rate']}"
            )
        
        # Data loaders (to be initialized)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None  
        
    def _print_model_summary(self):
        """Print and save detailed model summary"""
        import pandas as pd
        from pathlib import Path
        import time
        
        logger.info("Generating Heterogeneous Tool GNN Model Summary...")
        
        # Create results directory
        results_dir = Path('model_analysis')
        results_dir.mkdir(exist_ok=True)
        
        summary_data = []
        total_params = 0
        total_memory = 0
        
        print("\n" + "="*100)
        print("HETEROGENEOUS TOOL GNN MODEL SUMMARY")
        print("="*100)
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {next(self.model.parameters()).device}")
        print(f"Configuration: {self.config['model']}")
        print("-"*100)
        
        # Component analysis
        component_groups = {
            'Node Projections': [],
            'Heterogeneous GNN Layers': [],
            'Normalization Layers': [],
            'Cross Attention': [],
            'Contract Pooling': [],
            'Tool Attention': [],
            'Prediction Heads': [],
            'Loss Function': []
        }
        
        # Analyze model components
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                params = sum(p.numel() for p in module.parameters())
                trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                param_size_mb = sum(p.numel() * p.element_size() for p in module.parameters()) / (1024 * 1024)
                
                module_info = {
                    'Component': name if name else 'root',
                    'Type': module.__class__.__name__,
                    'Parameters': params,
                    'Trainable_Params': trainable_params,
                    'Memory_MB': param_size_mb,
                    'Input_Features': getattr(module, 'in_features', 'N/A'),
                    'Output_Features': getattr(module, 'out_features', 'N/A'),
                    'Hidden_Dim': getattr(module, 'hidden_dim', 'N/A')
                }
                
                # Categorize components
                if 'node_projections' in name:
                    component_groups['Node Projections'].append(module_info)
                elif 'hetero_layers' in name:
                    component_groups['Heterogeneous GNN Layers'].append(module_info)
                elif 'norms' in name or 'LayerNorm' in module.__class__.__name__:
                    component_groups['Normalization Layers'].append(module_info)
                elif 'cross_attention' in name:
                    component_groups['Cross Attention'].append(module_info)
                elif 'contract_pooling' in name:
                    component_groups['Contract Pooling'].append(module_info)
                elif 'tool_attention' in name:
                    component_groups['Tool Attention'].append(module_info)
                elif any(head in name for head in ['tpr_head', 'fpr_head', 'accuracy_head', 'precision_head', 'recall_head', 'securify_heads']):
                    component_groups['Prediction Heads'].append(module_info)
                
                summary_data.append(module_info)
                total_params += params
                total_memory += param_size_mb
        
        # Print categorized summary
        for category, components in component_groups.items():
            if components:
                print(f"\n{category.upper()}:")
                print(f"{'Component':<35} {'Type':<20} {'Params':<10} {'Memory(MB)':<12} {'In/Out Features':<20}")
                print("-"*100)
                
                for comp in components:
                    in_out = f"{comp['Input_Features']}/{comp['Output_Features']}"
                    if comp['Hidden_Dim'] != 'N/A':
                        in_out = f"Hidden: {comp['Hidden_Dim']}"
                    
                    print(f"{comp['Component']:<35} {comp['Type']:<20} {comp['Parameters']:<10,} {comp['Memory_MB']:<12.3f} {in_out:<20}")
        
        # Model architecture summary
        print(f"\n{'='*100}")
        print("MODEL ARCHITECTURE SUMMARY")
        print(f"{'='*100}")
        print(f"Total Layers: {len([m for m in self.model.modules() if len(list(m.children())) == 0])}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Non-trainable Parameters: {sum(p.numel() for p in self.model.parameters() if not p.requires_grad):,}")
        print(f"Total Memory Usage: {total_memory:.2f} MB")
        print(f"Model Size: {sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024*1024):.2f} MB")
        
        # Architecture details
        print(f"\nARCHITECTURE DETAILS:")
        print(f"Hidden Dimension: {self.config['model']['hidden_dim']}")
        print(f"Number of Layers: {self.config['model']['num_layers']}")
        print(f"Number of Attention Heads: {self.config['model']['num_heads']}")
        print(f"Dropout Rate: {self.config['model']['dropout']}")
        print(f"Node Types: {self.config['model']['node_types']}")
        print(f"Edge Types: {self.config['model']['edge_types']}")
        print(f"Number of Tools: {self.config['model']['num_tools']}")
        
        # Test model with sample input
        print(f"\nMODEL INPUT/OUTPUT VERIFICATION:")
        try:
            sample_input = self._create_minimal_sample_input()
            if sample_input is not None:
                with torch.no_grad():
                    self.model.eval()
                    output = self.model(sample_input)
                    if isinstance(output, tuple):
                        print(f"Model Output Shapes: {[o.shape for o in output]}")
                        print(f"Main Prediction Shape: {output[0].shape}")
                    else:
                        print(f"Model Output Shape: {output.shape}")
                    self.model.train()
        except Exception as e:
            print(f"Could not verify model I/O: {e}")
        
        # Save to file
        self._save_model_summary(summary_data, total_params, total_memory, results_dir)
        
        print(f"{'='*100}")
        logger.info("Model summary generation completed!")
    
    def _create_minimal_sample_input(self):
        """Create minimal sample input for model verification"""
        try:
            from torch_geometric.data import HeteroData
            
            sample_data = HeteroData()
            
            # Add minimal node features
            sample_data['function'].x = torch.randn(1, 21, device=self.device)
            sample_data['contract'].x = torch.randn(1, 32, device=self.device)
            
            # Add minimal tool labels
            sample_data['contract'].y_tool = torch.randn(1, 30, device=self.device)
            
            return sample_data
        except Exception as e:
            logger.warning(f"Could not create sample input: {e}")
            return None
    
    def _save_model_summary(self, summary_data, total_params, total_memory, results_dir):
        """Save model summary to files"""
        import pandas as pd
        import time
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save as CSV
        try:
            df = pd.DataFrame(summary_data)
            csv_path = results_dir / f'heterognn_model_summary_{timestamp}.csv'
            df.to_csv(csv_path, index=False)
            
            # Save detailed text report
            txt_path = results_dir / f'heterognn_model_summary_{timestamp}.txt'
            with open(txt_path, 'w') as f:
                f.write("HETEROGENEOUS TOOL GNN MODEL DETAILED SUMMARY\n")
                f.write("="*80 + "\n\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {self.model.__class__.__name__}\n")
                f.write(f"Device: {next(self.model.parameters()).device}\n\n")
                
                f.write("Configuration:\n")
                for key, value in self.config['model'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
                
                f.write("Component Details:\n")
                f.write(f"{'Component':<35} {'Type':<20} {'Parameters':<12} {'Memory(MB)':<12}\n")
                f.write("-"*84 + "\n")
                
                for comp in summary_data:
                    f.write(f"{comp['Component']:<35} {comp['Type']:<20} {comp['Parameters']:<12,} {comp['Memory_MB']:<12.3f}\n")
                
                f.write(f"\nSummary Statistics:\n")
                f.write(f"Total Parameters: {total_params:,}\n")
                f.write(f"Total Memory: {total_memory:.2f} MB\n")
            
            print(f"\nModel summary saved to:")
            print(f"  📄 CSV: {csv_path}")
            print(f"  📄 TXT: {txt_path}")
            
        except Exception as e:
            logger.warning(f"Could not save model summary files: {e}") 
            
    
    def prepare_data(self):
        """Prepare datasets for training."""
        logger.info("Loading contracts and tool results...")
        
        try:
            # Load data using the data loader properties (FIXED)
            contracts = self.data_loader.contracts  # Use property instead of method
            tool_results = self.data_loader.tool_results  # Use property instead of method
            injected_bugs = self.data_loader.injected_bugs  # Use property instead of method
            performance_metrics = self.data_loader.performance_metrics  # Use property instead of method
            
            logger.info(f"Loaded data: {len(contracts)} contracts, "
                    f"{len(tool_results)} tool result sets, "
                    f"{len(injected_bugs)} injected bug entries")
            
            if not contracts:
                raise ValueError("No contracts loaded!")
            
            # Split data
            train_contracts, val_contracts, test_contracts = self.data_loader.get_dataset_split(contracts)
            
            logger.info(f"Train: {len(train_contracts)}, Val: {len(val_contracts)}, Test: {len(test_contracts)}")
            
            # Ensure we have at least some data for each split
            if len(train_contracts) == 0:
                raise ValueError("No training contracts available!")
            
            # If validation set is empty, use a small portion of training set
            if len(val_contracts) == 0:
                logger.warning("No validation contracts, using 10% of training set")
                split_idx = max(1, len(train_contracts) // 10)
                val_contracts = train_contracts[:split_idx]
                train_contracts = train_contracts[split_idx:]
            
            # Create datasets with FIXED parameters
            train_dataset = HeteroToolDataset(
                train_contracts, 
                self.processor, 
                self.graph_builder,
                tool_results,  # Direct pass
                injected_bugs,  # Direct pass
                performance_metrics  # Direct pass
            )
            
            val_dataset = HeteroToolDataset(
                val_contracts, 
                self.processor, 
                self.graph_builder,
                tool_results,  # Direct pass
                injected_bugs,  # Direct pass
                performance_metrics  # Direct pass
            )
            
            test_dataset = HeteroToolDataset(
                test_contracts, 
                self.processor, 
                self.graph_builder,
                tool_results,  # Direct pass
                injected_bugs,  # Direct pass
                performance_metrics  # Direct pass
            ) if test_contracts else None
            
            # Check dataset sizes after filtering
            if len(train_dataset) == 0:
                raise ValueError("Training dataset is empty after filtering!")
            
            if len(val_dataset) == 0:
                logger.warning("Validation dataset is empty, creating minimal validation set")
                # Create a minimal validation set from training data
                val_dataset = HeteroToolDataset(
                    train_contracts[:2],  # Use first 2 training contracts
                    self.processor, 
                    self.graph_builder,
                    tool_results,
                    injected_bugs,
                    performance_metrics
                )
            
            # Create data loaders
            batch_size = self.config['training']['batch_size']
            num_workers = 0  # Avoid multiprocessing issues
            
            self.train_loader = GeometricDataLoader(
                train_dataset, 
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=True  # Drop last incomplete batch
            )
            
            self.val_loader = GeometricDataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
            
            if test_dataset and len(test_dataset) > 0:
                self.test_loader = GeometricDataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers
                )
            else:
                self.test_loader = None
            
            logger.info("Data preparation completed successfully.")
            logger.info(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
            
            # Test loading one batch to verify everything works
            try:
                test_batch = next(iter(self.train_loader))
                logger.info(f"Test batch loaded successfully:")
                logger.info(f"  Batch keys: {list(test_batch.keys())}")
                if 'contract' in test_batch:
                    logger.info(f"  Contract features shape: {test_batch['contract'].x.shape}")
                    if hasattr(test_batch['contract'], 'y_tool'):
                        logger.info(f"  Tool labels shape: {test_batch['contract'].y_tool.shape}")
            except Exception as e:
                logger.error(f"Error loading test batch: {e}")
                raise
            
        except Exception as e:
            logger.error(f"Error in data preparation: {e}")
            logger.error("Please check:")
            logger.error("1. Data source path is correct")
            logger.error("2. Data has the expected format")
            logger.error("3. Sufficient disk space is available")
            raise
    
        # Add this call to the train() method at the beginning (right after prepare_data())
        if self.config.get('debug', {}).get('check_data_consistency', False):
            self.verify_data_consistency() 
    
    
    def train_epoch(self, epoch: int) -> Dict:
        """Fixed training epoch with proper shape handling."""
        self.model.train()
        total_loss = 0.0
        total_losses = {}
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            try:
                batch = batch.to(self.device)
                
                # Forward pass
                raw_predictions = self.model(batch)
                
                # Handle tuple output from model
                if isinstance(raw_predictions, tuple):
                    predictions = raw_predictions[0]  # Take the main prediction tensor
                else:
                    predictions = raw_predictions
                
                # Get targets
                targets = batch['contract'].y_tool  
                
                # targets[targets == 0.5] = 0.0
                
                # Ensure shapes match
                batch_size = targets.shape[0]
                expected_features = 30 
                
                # Reshape predictions if needed
                if predictions.dim() == 1:
                    predictions = predictions.unsqueeze(0)
                
                if predictions.shape[1] != expected_features:
                    logger.warning(f"Prediction shape mismatch: {predictions.shape} vs expected (*, {expected_features})")
                    # Take only what we need or pad with zeros
                    if predictions.shape[1] > expected_features:
                        predictions = predictions[:, :expected_features]
                    else:
                        padding = torch.zeros(predictions.shape[0], 
                                            expected_features - predictions.shape[1], 
                                            0.5, device=predictions.device)
                        predictions = torch.cat([predictions, padding], dim=1)
                
                # Ensure targets have correct shape
                if targets.shape[1] != expected_features:
                    if targets.shape[1] > expected_features:
                        targets = targets[:, :expected_features]
                    else:
                        padding = torch.full((targets.shape[0], 
                                            expected_features - targets.shape[1]), 
                                            0.5, device=targets.device)
                        targets = torch.cat([targets, padding], dim=1)
                
                # Ensure batch sizes match
                min_batch = min(predictions.shape[0], targets.shape[0])
                predictions = predictions[:min_batch]
                targets = targets[:min_batch]
                
                # Compute loss
                loss, losses = self.criterion(predictions, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clip']
                )
                
                self.optimizer.step()
                
                # Accumulate losses
                total_loss += loss.item()
                for key, value in losses.items():
                    if key not in total_losses:
                        total_losses[key] = 0.0
                    total_losses[key] += value.item()
                
                # Store for metrics
                all_predictions.append(predictions.detach())
                all_targets.append(targets.detach())
                
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Calculate average losses
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        else:
            avg_loss = float('inf')
            avg_losses = {}
        
        # Calculate training metrics
        train_metrics = {}
        if all_predictions and all_targets:
            try:
                all_preds = torch.cat(all_predictions, dim=0)
                all_targs = torch.cat(all_targets, dim=0)
                
                train_metrics = self.metrics_calculator.calculate_tool_metrics(
                    all_preds.cpu(), all_targs.cpu()
                )
                
                # Safe logging
                def safe_format(value):
                    try:
                        return f"{float(value):.4f}"
                    except (ValueError, TypeError):
                        return str(value)
                
                logger.info(f"Train - TPR MAE: {safe_format(train_metrics.get('tpr_avg_mae', 'N/A'))}, "
                        f"FPR MAE: {safe_format(train_metrics.get('fpr_avg_mae', 'N/A'))}")
                        
            except Exception as e:
                logger.error(f"Error calculating training metrics: {e}") 
                
        # In train_epoch method, after calculating train_metrics, add:
        if self.config.get('logging', {}).get('log_confusion_matrices', False) and epoch % 5 == 0:
            # Log confusion matrix details every 5 epochs
            if all_predictions and all_targets:
                all_preds = torch.cat(all_predictions, dim=0)
                all_targs = torch.cat(all_targets, dim=0)
                self.log_confusion_matrix_details(epoch, all_preds, all_targs, "train")
        
        return {'loss': avg_loss, **avg_losses, **train_metrics} 
    
    
    def compute_confusion_per_tool(self, preds: torch.Tensor, targets: torch.Tensor, tools: List[str], threshold: float = 0.5):
        """
        Returns a dictionary: tool -> dict(tp, fn, fp, tn)
        """
        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        tool_confusion = {}
        num_metrics_per_tool = 5

        for i, tool in enumerate(tools):
            tp = fn = fp = tn = 0
            for row in range(preds_np.shape[0]):
                for m in range(num_metrics_per_tool):
                    col = i * num_metrics_per_tool + m
                    pred = preds_np[row][col] >= threshold
                    true = targets_np[row][col] >= threshold
                    if true and pred:
                        tp += 1
                    elif true and not pred:
                        fn += 1
                    elif not true and pred:
                        fp += 1
                    elif not true and not pred:
                        tn += 1
            tool_confusion[tool] = {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn}
        return tool_confusion
       
    
    def validate(self, epoch: int = None) -> Dict:
        """Validate the model with enhanced confusion matrix tracking and F1 score monitoring."""
        
        # Safe logging utility function
        def safe_float(value, precision=4):
            try:
                return f"{float(value):.{precision}f}"
            except (ValueError, TypeError):
                return str(value)
        
        self.model.eval()
        total_loss = 0.0
        total_losses = {}
        num_batches = 0
        num_samples = 0
        
        all_predictions = []
        all_targets = []
        
        logger.info("Starting validation...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='Validation')):
                try:
                    batch = batch.to(self.device)
                    batch_size = batch['contract'].x.shape[0]
                    num_samples += batch_size
                    
                    # Forward pass with proper tuple handling
                    raw_predictions = self.model(batch)
                    if isinstance(raw_predictions, tuple):
                        predictions = raw_predictions[0]
                        tool_predictions = raw_predictions[1] if len(raw_predictions) > 1 else None
                    else:
                        predictions = raw_predictions
                        tool_predictions = None
                    
                    # Get targets
                    targets = batch['contract'].y_tool 
                    
                    # targets[targets == 0.5] = 0.0
                    
                    # Debug logging for shape tracking
                    if batch_idx == 0:  # Log shapes for first batch
                        logger.debug(f"Validation batch shapes - Predictions: {predictions.shape}, Targets: {targets.shape}")
                    
                    # Ensure compatible shapes before loss calculation
                    batch_size_actual = min(predictions.shape[0], targets.shape[0])
                    expected_features = 30 
                    
                    # Handle prediction shape
                    if predictions.dim() == 1:
                        predictions = predictions.unsqueeze(0)
                    
                    if predictions.shape[1] != expected_features:
                        if predictions.shape[1] > expected_features:
                            predictions = predictions[:, :expected_features]
                        else:
                            # Pad with default values
                            padding = torch.full((predictions.shape[0], 
                                                expected_features - predictions.shape[1]), 
                                            0.5, device=predictions.device, dtype=predictions.dtype)
                            predictions = torch.cat([predictions, padding], dim=1)
                    
                    # Handle target shape
                    if targets.shape[1] != expected_features:
                        if targets.shape[1] > expected_features:
                            targets = targets[:, :expected_features]
                        else:
                            padding = torch.full((targets.shape[0], 
                                                expected_features - targets.shape[1]), 
                                            0.5, device=targets.device, dtype=targets.dtype)
                            targets = torch.cat([targets, padding], dim=1)
                    
                    # Ensure batch sizes match
                    predictions = predictions[:batch_size_actual]
                    targets = targets[:batch_size_actual]
                    
                    # Compute loss with shape-aligned tensors
                    loss, losses = self.criterion(predictions, targets)
                    
                    # Accumulate losses
                    total_loss += loss.item()
                    for key, value in losses.items():
                        if key not in total_losses:
                            total_losses[key] = 0.0
                        total_losses[key] += value.item()
                    
                    # Store predictions and targets for metrics calculation
                    pred_tensor = predictions.detach().cpu()
                    target_tensor = targets.detach().cpu()
                    
                    # Ensure 2D tensors for consistency
                    if pred_tensor.dim() == 1:
                        pred_tensor = pred_tensor.unsqueeze(0)
                    if target_tensor.dim() == 1:
                        target_tensor = target_tensor.unsqueeze(0)
                    
                    all_predictions.append(pred_tensor)
                    all_targets.append(target_tensor)
                    
                    num_batches += 1
                    
                    # Log progress for long validation
                    if batch_idx % 50 == 0 and batch_idx > 0:
                        logger.debug(f"Validation progress: {batch_idx}/{len(self.val_loader)} batches")
                        
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    # Log more details for debugging
                    if batch_idx < 5:  # Only log details for first few batches to avoid spam
                        logger.error(f"  Batch details - Contract features: {batch.get('contract', {}).get('x', 'missing').shape if hasattr(batch.get('contract', {}), 'x') else 'no x'}")
                    continue
        
        logger.info(f"Validation completed: {num_batches} batches, {num_samples} samples")
        
        # Calculate average losses
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        else:
            logger.warning("No validation batches processed successfully!")
            return {'loss': float('inf'), 'error': 'No valid batches'}
        
        # Calculate comprehensive metrics including F1
        tool_metrics = {}
        
        if all_predictions and all_targets:
            try:
                # Concatenate all predictions and targets
                all_preds = torch.cat(all_predictions, dim=0)
                all_targs = torch.cat(all_targets, dim=0)
                
                logger.info(f"Final validation shapes - Predictions: {all_preds.shape}, Targets: {all_targs.shape}")
                
                # Calculate comprehensive metrics using the enhanced metrics calculator
                tool_metrics = self.metrics_calculator.calculate_tool_metrics(
                    all_preds, all_targs, detailed=True
                )
                
                # Log key metrics with safe formatting
                logger.info("=== Validation Metrics ===")
                logger.info(f"Overall MSE: {safe_float(tool_metrics.get('overall_mse', 'N/A'))}")
                logger.info(f"Overall MAE: {safe_float(tool_metrics.get('overall_mae', 'N/A'))}")
                logger.info(f"Overall RMSE: {safe_float(tool_metrics.get('overall_rmse', 'N/A'))}")
                logger.info(f"Overall MAPE: {safe_float(tool_metrics.get('overall_mape', 'N/A'))}")
                
                # Log per-metric performance including F1
                metric_names = ['tpr', 'fpr', 'accuracy', 'precision', 'recall', 'f1_score']
                logger.info("Per-Metric Performance:")
                for metric in metric_names:
                    mae = tool_metrics.get(f'{metric}_avg_mae', 'N/A')
                    mse = tool_metrics.get(f'{metric}_avg_mse', 'N/A')
                    r2 = tool_metrics.get(f'{metric}_avg_r2', 'N/A') 
                    rmse = tool_metrics.get(f'{metric}_avg_rmse', 'N/A')
                    mape = tool_metrics.get(f'{metric}_avg_mape', 'N/A')
                    
                    if mae != 'N/A':
                        logger.info(f"  {metric.upper()}: MAE={safe_float(mae)}, MSE={safe_float(mse)}, R²={safe_float(r2)}, RMSE={safe_float(rmse)}, MAPE={safe_float(mape)}")
                    else:
                        logger.info(f"  {metric.upper()}: N/A")
                
                # Log per-tool performance summary
                tool_names = ['oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither']
                logger.info("Per-Tool Performance:")
                for tool in tool_names:
                    mae = tool_metrics.get(f'{tool}_mae', 'N/A')
                    mse = tool_metrics.get(f'{tool}_mse', 'N/A')
                    r2 = tool_metrics.get(f'{tool}_r2', 'N/A') 
                    rmse = tool_metrics.get(f'{tool}_rmse', 'N/A')
                    mape = tool_metrics.get(f'{tool}_mape', 'N/A')
                    
                    if mae != 'N/A':
                        logger.info(f"  {tool.title()}: MAE={safe_float(mae)}, MSE={safe_float(mse)}, R²={safe_float(r2)}, RMSE={safe_float(rmse)}, MAPE={safe_float(mape)}")
                
                # Enhanced confusion matrix logging if enabled
                if (epoch is not None and 
                    self.config.get('logging', {}).get('log_confusion_matrices', False) and 
                    epoch % 5 == 0):  # Log every 5 epochs
                    
                    self.log_confusion_matrix_details(epoch, all_preds, all_targs, "validation")
                
                # Validate metric ranges for sanity check
                # self._validate_metric_ranges(tool_metrics, "validation")
                        
            except Exception as e:
                logger.error(f"Error calculating validation metrics: {e}")
                import traceback
                logger.error(f"Metrics calculation traceback: {traceback.format_exc()}")
                tool_metrics = {'metrics_error': str(e)}
        else:
            logger.warning("No predictions or targets available for metrics calculation")
            tool_metrics = {'warning': 'No data for metrics'}
        
        # Create comprehensive validation results
        validation_results = {
            'loss': avg_loss,
            **avg_losses,
            **tool_metrics,
            'num_samples': num_samples,
            'num_batches': num_batches
        } 
        
        # tp, fn, fp, tn = self.compute_confusion(all_preds, all_targs)
        # validation_results.update({
        #     'tp': tp,
        #     'fp': fp,
        #     'fn': fn,
        #     'tn': tn
        # })
        # logger.info(f"[Validation Epoch {epoch}] TP={tp}, FN={fn}, FP={fp}, TN={tn}") 
        
        print("Target value range:", torch.unique(all_targs))
        
        tool_names = ['oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither']
        tool_confusions = self.compute_confusion_per_tool(all_preds, all_targs, tool_names) 
        
        # plot_tool_confusion_matrix(tool_confusions, save_path=f'training_plots/confusion_matrix_epoch_{epoch}.png')

        # Add to validation results
        for tool, values in tool_confusions.items():
            for k, v in values.items():
                validation_results[f'{tool}_{k}'] = v

        # logger.info("=== Confusion Matrix Per Tool ===")
        # for tool, values in tool_confusions.items():
        #     logger.info(f"  {tool.title()}: TP={values['tp']}, FN={values['fn']}, FP={values['fp']}, TN={values['tn']}")
        
        # Additional validation checks if debugging is enabled
        if self.config.get('debug', {}).get('validate_confusion_matrices', False):
            validation_results.update(self._perform_validation_sanity_checks(all_preds, all_targs)) 
            

        if epoch is not None and epoch % 10 == 0:
            try:
                detailed_confusion = self.data_loader.get_detailed_confusion_matrix_per_vulnerability()
                vulnerability_metrics = self.metrics_calculator.calculate_per_vulnerability_metrics(detailed_confusion)
                
                # Log summary statistics
                logger.info(f"\n=== FP/FN Analysis (Epoch {epoch}) ===")
                for tool in ['oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither']:
                    if tool in vulnerability_metrics:
                        total_fp = sum(metrics['FP'] for metrics in vulnerability_metrics[tool].values())
                        total_fn = sum(metrics['FN'] for metrics in vulnerability_metrics[tool].values())
                        logger.info(f"{tool.title()}: Total FP={total_fp}, Total FN={total_fn}")
                
                validation_results.update({
                    'detailed_fp_fn': {
                        'total_false_positives': {
                            tool: sum(metrics['FP'] for metrics in tool_metrics.values())
                            for tool, tool_metrics in vulnerability_metrics.items()
                        },
                        'total_false_negatives': {
                            tool: sum(metrics['FN'] for metrics in tool_metrics.values())
                            for tool, tool_metrics in vulnerability_metrics.items()
                        }
                    }
                })
                
            except Exception as e:
                logger.warning(f"Error in detailed FP/FN analysis: {e}")
        
        return validation_results
        
    
    def train(self):
        """Main training loop with enhanced tracking and visualization."""
        logger.info("Starting training for tool performance prediction...")

        # Prepare data
        self.prepare_data() 


        best_val_loss = float('inf')
        best_f1_mae = float('inf')

        for epoch in range(self.config['training']['num_epochs']):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate() 

            # Learning rate scheduling
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('loss', 0.0))
                else:
                    self.scheduler.step()

            # Store history
            self.history['train_loss'].append(train_metrics.get('loss', float('inf')))
            self.history['val_loss'].append(val_metrics.get('loss', float('inf')))
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['learning_rates'].append(current_lr)

            # === Safe F1 retrieval ===
            train_f1_mae = train_metrics.get('f1_score_avg_mae', float('inf'))
            val_f1_mae = val_metrics.get('f1_score_avg_mae', float('inf'))

            try:
                if not isinstance(train_f1_mae, (float, int)):
                    train_f1_mae = float(train_f1_mae)
                if not isinstance(val_f1_mae, (float, int)):
                    val_f1_mae = float(val_f1_mae)
            except Exception as e:
                logger.warning(f"Could not convert F1 MAE to float: {e}")
                train_f1_mae, val_f1_mae = float('inf'), float('inf')

            logger.info(f"Epoch {epoch}: Train Loss: {train_metrics.get('loss', float('nan')):.4f}, "
                        f"Val Loss: {val_metrics.get('loss', float('nan')):.4f}, LR: {current_lr:.6f}")
            logger.info(f"Train - TPR MAE: {train_metrics.get('tpr_avg_mae', 'N/A')}, "
                        f"FPR MAE: {train_metrics.get('fpr_avg_mae', 'N/A')}, "
                        f"F1 MAE: {train_f1_mae if train_f1_mae != float('inf') else 'N/A'}")

            logger.info(f"Val - TPR MAE: {val_metrics.get('tpr_avg_mae', 'N/A')}, "
                        f"FPR MAE: {val_metrics.get('fpr_avg_mae', 'N/A')}, "
                        f"F1 MAE: {val_f1_mae if val_f1_mae != float('inf') else 'N/A'}")

            # Log to wandb if enabled
            if self.use_wandb:
                wandb_log = {
                    'epoch': epoch,
                    'train_loss': train_metrics.get('loss', float('nan')),
                    'val_loss': val_metrics.get('loss', float('nan')),
                    'learning_rate': current_lr,
                    **{f'train_{k}': v for k, v in train_metrics.items() if k != 'loss'},
                    **{f'val_{k}': v for k, v in val_metrics.items() if k != 'loss'}
                }
                wandb.log(wandb_log)

            # Check for best models
            is_best_loss = val_metrics.get('loss', float('inf')) < best_val_loss
            is_best_f1 = val_f1_mae < best_f1_mae

            if is_best_loss:
                best_val_loss = val_metrics.get('loss', float('inf'))
            if is_best_f1:
                best_f1_mae = val_f1_mae

            # Save checkpoints
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'val_loss': val_metrics.get('loss', float('inf')),
                'val_f1_mae': val_f1_mae,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'config': self.config
            }

            self.model_checkpoint.save_checkpoint(checkpoint_data, is_best_loss)

            # Save best F1 model separately
            if is_best_f1:
                f1_checkpoint_path = Path('checkpoints') / 'best_f1_model.pt'
                torch.save(checkpoint_data, f1_checkpoint_path)
                logger.info(f"Saved best F1 model with MAE: {val_f1_mae:.4f}")

            # Early stopping
            if self.early_stopping(val_metrics.get('loss', float('inf'))):
                logger.info(f"Early stopping at epoch {epoch}")
                break

            # Periodic visualization
            if epoch % self.config['logging'].get('checkpoint_frequency', 10) == 0:
                self.plot_training_history(epoch)

        # Final plots
        self.plot_training_history(self.config['training']['num_epochs'] - 1)
        self.save_training_history()
        logger.info("Training completed.") 
        
        
    def log_confusion_matrix_details(self, epoch: int, predictions: torch.Tensor, 
                                    targets: torch.Tensor, split: str = "train"):
        """Log detailed confusion matrix information for verification."""
        if not self.config.get('logging', {}).get('log_confusion_matrices', False):
            return
        
        try:
            # Convert to numpy
            if isinstance(predictions, tuple):
                preds = predictions[0].detach().cpu().numpy()
            else:
                preds = predictions.detach().cpu().numpy()
            
            targs = targets.detach().cpu().numpy()
            
            # Reshape to (batch_size, num_tools, 5)
            batch_size = preds.shape[0]
            num_tools = 6
            
            if preds.shape[1] >= 30:  # Ensure we have enough features
                preds_reshaped = preds[:, :30].reshape(batch_size, num_tools, 5)
                targs_reshaped = targs[:, :30].reshape(batch_size, num_tools, 5)
                
                # Calculate average metrics per tool
                tool_names = ['oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither']
                
                logger.info(f"\n=== {split.title()} Confusion Matrix Details (Epoch {epoch}) ===")
                
                for tool_idx, tool_name in enumerate(tool_names):
                    if tool_idx >= num_tools:
                        continue
                    
                    tool_preds = preds_reshaped[:, tool_idx, :]  # Shape: (batch_size, 5)
                    tool_targs = targs_reshaped[:, tool_idx, :]
                    
                    # Calculate average predicted metrics
                    avg_pred_tpr = np.mean(tool_preds[:, 0])
                    avg_pred_fpr = np.mean(tool_preds[:, 1])
                    avg_pred_acc = np.mean(tool_preds[:, 2])
                    
                    # Calculate average target metrics
                    avg_targ_tpr = np.mean(tool_targs[:, 0])
                    avg_targ_fpr = np.mean(tool_targs[:, 1])
                    avg_targ_acc = np.mean(tool_targs[:, 2])
                    
                    # Calculate errors
                    tpr_error = np.mean(np.abs(tool_preds[:, 0] - tool_targs[:, 0]))
                    fpr_error = np.mean(np.abs(tool_preds[:, 1] - tool_targs[:, 1]))
                    acc_error = np.mean(np.abs(tool_preds[:, 2] - tool_targs[:, 2]))
                    
                    logger.info(f"{tool_name.title():<12}: "
                            f"TPR={avg_pred_tpr:.3f}({avg_targ_tpr:.3f}) "
                            f"FPR={avg_pred_fpr:.3f}({avg_targ_fpr:.3f}) "
                            f"Acc={avg_pred_acc:.3f}({avg_targ_acc:.3f}) | "
                            f"Errors: {tpr_error:.3f}, {fpr_error:.3f}, {acc_error:.3f}")
                
        except Exception as e:
            logger.warning(f"Error logging confusion matrix details: {e}")

        
        # Check if we have data
        contracts = self.data_loader.contracts
        tool_results = self.data_loader.tool_results
        injected_bugs = self.data_loader.injected_bugs
        performance_metrics = self.data_loader.performance_metrics
        
        logger.info(f"Contracts: {len(contracts)}")
        logger.info(f"Tool results: {len(tool_results)}")
        logger.info(f"Injected bugs: {len(injected_bugs)}")
        logger.info(f"Performance metrics: {len(performance_metrics)} tools")
        
        # Check for contracts without tool results
        contracts_without_results = 0
        for contract in contracts:
            if contract['id'] not in tool_results:
                contracts_without_results += 1
        
        logger.info(f"Contracts without tool results: {contracts_without_results}/{len(contracts)}")
        
        # Check tool coverage
        tool_coverage = {}
        for contract_id, results in tool_results.items():
            for tool_name in results:
                if tool_name not in tool_coverage:
                    tool_coverage[tool_name] = 0
                tool_coverage[tool_name] += 1
        
        logger.info("Tool coverage:")
        for tool, count in sorted(tool_coverage.items()):
            percentage = (count / len(contracts) * 100) if contracts else 0
            logger.info(f"  {tool}: {count} contracts ({percentage:.1f}%)")
        
        # Check performance metrics sanity
        logger.info("\nPerformance metrics sanity check:")
        for tool, metrics in performance_metrics.items():
            if isinstance(metrics, dict):
                tpr = metrics.get('tpr', 0)
                fpr = metrics.get('fpr', 0)
                accuracy = metrics.get('accuracy', 0)
                
                # Sanity checks
                issues = []
                if not (0 <= tpr <= 1):
                    issues.append(f"TPR out of range: {tpr}")
                if not (0 <= fpr <= 1):
                    issues.append(f"FPR out of range: {fpr}")
                if not (0 <= accuracy <= 1):
                    issues.append(f"Accuracy out of range: {accuracy}")
                
                if issues:
                    logger.warning(f"  {tool}: {', '.join(issues)}")
                else:
                    logger.info(f"  {tool}: TPR={tpr:.3f}, FPR={fpr:.3f}, Acc={accuracy:.3f} ✓")
    
    
    def plot_training_history(self, epoch: int):
        """Plot comprehensive training history including F1 scores."""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15)) 
        
        # Set global font properties
        plt.rcParams.update({
            'font.family': 'serif',          # or 'sans-serif', 'monospace'
            'font.serif': 'Times New Roman', # or 'Computer Modern', 'DejaVu Serif'
            'font.size': 12,
            'font.weight': 'normal',         # or 'bold'
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10
        }) 
        
        # Prepare plotting data dictionary
        plotting_data = {
            'epoch': epoch,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'train_loss': self.history['train_loss'].copy(),
            'val_loss': self.history['val_loss'].copy(),
            'learning_rates': self.history['learning_rates'].copy(),
            'metrics': {},
            'summary_stats': {}
        }
        
        # Training and validation loss
        ax = axes[0, 0]
        ax.plot(self.history['train_loss'], label='Train', linewidth=2)
        ax.plot(self.history['val_loss'], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3) 
        
        # Store loss data
        plotting_data['loss_data'] = {
            'train_loss': self.history['train_loss'].copy(),
            'val_loss': self.history['val_loss'].copy()
        }
                       
         
        # Per-metric MAE
        ax = axes[0, 1]
        metrics_to_plot = ['tpr', 'fpr', 'recall', 'precision']    # 'accuracy', 'f1_score' 
        metric_mae_data = {} 
        for metric in metrics_to_plot:
            val_maes = [m.get(f'{metric}_avg_mae', np.nan) for m in self.history['val_metrics']] 
            ax.plot(val_maes, label=metric.upper(), linewidth=2)
            metric_mae_data[metric] = val_maes 
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE')
        ax.set_title('Validation MAE by Metric')
        ax.legend()
        ax.grid(True, alpha=0.3) 
        
        # Store metric MAE data
        plotting_data['metric_mae_data'] = metric_mae_data
        
        # F1 Score specific
        ax = axes[0, 2]
        train_f1 = [m.get('f1_score_avg_mae', np.nan) for m in self.history['train_metrics']]
        val_f1 = [m.get('f1_score_avg_mae', np.nan) for m in self.history['val_metrics']]
        ax.plot(train_f1, label='Train F1 MAE', linewidth=2)
        ax.plot(val_f1, label='Val F1 MAE', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score MAE')
        ax.set_title('F1 Score Performance')
        ax.legend()
        ax.grid(True, alpha=0.3) 
        
        # Store F1 data
        plotting_data['f1_data'] = {
            'train_f1_mae': train_f1,
            'val_f1_mae': val_f1
        }
        
        # Per-tool performance
        ax = axes[1, 0]
        tool_names = ['oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither'] 
        tool_mae_data = {}
        for tool in tool_names:
            tool_maes = [m.get(f'{tool}_mae', np.nan) for m in self.history['val_metrics']]
            ax.plot(tool_maes, label=tool.title(), linewidth=2) 
            tool_mae_data[tool] = tool_maes
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE')
        ax.set_title('Validation MAE by Tool')
        ax.legend()
        ax.grid(True, alpha=0.3) 
        
        # Store tool MAE data
        plotting_data['tool_mae_data'] = tool_mae_data
    
        
        # Per-metric RMSE
        ax = axes[1, 1]
        metrics_to_plot = ['tpr', 'fpr', 'recall', 'precision']    # 'accuracy', 'f1_score' 
        metric_rmse_data = {} 
        for metric in metrics_to_plot:
            val_rmses = [m.get(f'{metric}_avg_rmse', np.nan) for m in self.history['val_metrics']] 
            ax.plot(val_rmses, label=metric.upper(), linewidth=2)
            metric_rmse_data[metric] = val_rmses 
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('Validation RMSE by Metric')
        ax.legend()
        ax.grid(True, alpha=0.3) 
        
        # Store metric MAE data
        plotting_data['metric_rmse_data'] = metric_rmse_data
        
        # Validation MAPE per tool
        ax = axes[1, 2] 
        tool_names = ['oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither'] 
        tool_mape_data = {}
        for tool in tool_names:
            tool_mapes = [m.get(f'{tool}_mape', np.nan) for m in self.history['val_metrics']]
            ax.plot(tool_mapes, label=tool.title(), linewidth=2) 
            tool_mape_data[tool] = tool_mapes
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAPE (%)')
        ax.set_title('Validation MAPE by Tool')
        ax.legend()
        ax.grid(True, alpha=0.3) 
        
        # Store tool MAPE data
        plotting_data['tool_mape_data'] = tool_mape_data

        # Validation RMSE by Tool
        ax = axes[2, 0]
        tool_names = ['oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither'] 
        tool_rmse_data = {} 
        for tool in tool_names:
            tool_rmses = [m.get(f'{tool}_rmse', np.nan) for m in self.history['val_metrics']]
            ax.plot(tool_rmses, label=tool.title(), linewidth=2) 
            tool_rmse_data[tool] = tool_rmses
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('Validation RMSE by Tool')
        ax.legend()
        ax.grid(True, alpha=0.3) 

        # Store tool RMSE data
        plotting_data['tool_rmse_data'] = tool_rmse_data  
        
           
        # Summary of final MAPE and RMSE
        ax = axes[2, 1]
        latest_val = self.history['val_metrics'][-1] if self.history['val_metrics'] else {}
        summary_names = ['MAPE', 'RMSE']
        summary_vals = [
            latest_val.get('overall_mape', 0),
            latest_val.get('overall_rmse', 0)
        ]

        bars = ax.bar(summary_names, summary_vals, alpha=0.8, color=['darkorange', 'royalblue'])
        ax.set_ylabel('Value')
        ax.set_title('Final Validation MAPE and RMSE')

        # Add value labels on bars
        for bar, value in zip(bars, summary_vals):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{value:.3f}', ha='center', va='bottom')

        ax.grid(axis='y', alpha=0.3) 
        
        # Store summary data
        plotting_data['final_summary'] = {
            'summary_names': summary_names,
            'summary_values': summary_vals
        }
        
        # Extract additional metrics for comprehensive data saving
        plotting_data['metrics'] = {
            'all_train_metrics': self.history['train_metrics'].copy(),
            'all_val_metrics': self.history['val_metrics'].copy()
        }
        
        # Calculate summary statistics
        plotting_data['summary_stats'] = {
            'best_val_loss': min(self.history['val_loss']) if self.history['val_loss'] else float('inf'),
            'best_val_loss_epoch': self.history['val_loss'].index(min(self.history['val_loss'])) if self.history['val_loss'] else -1,
            'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else float('inf'),
            'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else float('inf'),
            'final_learning_rate': self.history['learning_rates'][-1] if self.history['learning_rates'] else 0.0,
            'total_epochs_trained': len(self.history['train_loss'])
        }
        
        # Add tool-specific summary statistics
        for tool in tool_names:
            tool_maes = [m.get(f'{tool}_mae', float('inf')) for m in self.history['val_metrics']]
            if tool_maes and not all(mae == float('inf') for mae in tool_maes):
                plotting_data['summary_stats'][f'{tool}_best_mae'] = min(mae for mae in tool_maes if mae != float('inf'))
                plotting_data['summary_stats'][f'{tool}_final_mae'] = tool_maes[-1] if tool_maes else float('inf')
        
        # Add metric-specific summary statistics
        for metric in ['tpr', 'fpr', 'accuracy', 'precision', 'recall', 'f1_score']:
            metric_maes = [m.get(f'{metric}_avg_mae', float('inf')) for m in self.history['val_metrics']]
            if metric_maes and not all(mae == float('inf') for mae in metric_maes):
                plotting_data['summary_stats'][f'{metric}_best_avg_mae'] = min(mae for mae in metric_maes if mae != float('inf'))
                plotting_data['summary_stats'][f'{metric}_final_avg_mae'] = metric_maes[-1] if metric_maes else float('inf')
        
        
        plt.tight_layout() 
        
        # Save plot
        plot_dir = Path('training_plots')
        plot_dir.mkdir(exist_ok=True)
        plt.savefig(plot_dir / f'training_history_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close() 
        
        
        # Convert numpy types to native Python types for YAML serialization
        def convert_for_yaml(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_for_yaml(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_yaml(item) for item in obj]
            elif isinstance(obj, tuple):
                return [convert_for_yaml(item) for item in obj]
            elif pd.isna(obj) if hasattr(pd, 'isna') else np.isnan(obj):
                return None
            else:
                return obj
        
        # Clean the plotting data for YAML serialization
        plotting_data_clean = convert_for_yaml(plotting_data)
        
        # Save to YAML file
        results_plot_path = Path('training_plots') / 'results_plot.yaml' 
        try:
            with open(results_plot_path, 'w') as f:
                yaml.dump(plotting_data_clean, f, default_flow_style=False, indent=2, sort_keys=False)
            logger.info(f"Plotting data saved to {results_plot_path}")
        except Exception as e:
            logger.warning(f"Failed to save plotting data to YAML: {e}")
        
        
        # Save a simple loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        plt.plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_dir / 'loss_curve.png', dpi=300, bbox_inches='tight')
        plt.close() 
        
        # Save a simple MAE by Metric plot
        plt.figure(figsize=(10, 6))
        metrics_to_plot = ['tpr', 'fpr', 'precision', 'recall']          # 'recall', 'f1_score'
        for metric in metrics_to_plot:
            val_maes = [m.get(f'{metric}_avg_mae', np.nan) for m in self.history['val_metrics']]
            plt.plot(val_maes, label=metric.upper(), linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('Validation MAE by Metric')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_dir / 'MAE_by_Metric_curve.png', dpi=300, bbox_inches='tight')
        plt.close()  
        
        # Save a simple RMSE by Metric plot
        plt.figure(figsize=(10, 6))
        metrics_to_plot = ['tpr', 'fpr', 'precision', 'recall']          # 'recall', 'f1_score'
        for metric in metrics_to_plot:
            val_maes = [m.get(f'{metric}_avg_rmse', np.nan) for m in self.history['val_metrics']]
            plt.plot(val_maes, label=metric.upper(), linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('Validation RMSE by Metric')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_dir / 'RMSE_by_Metric_curve.png', dpi=300, bbox_inches='tight')
        plt.close() 
        
        # Save a simple MAE by Tool plot
        plt.figure(figsize=(10, 6))
        tool_names = ['oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither']
        for tool in tool_names:
            tool_maes = [m.get(f'{tool}_mae', np.nan) for m in self.history['val_metrics']]
            plt.plot(tool_maes, label=tool.title(), linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('Validation MAE by Tool')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_dir / 'MAE_by_Tool_curve.png', dpi=300, bbox_inches='tight')
        plt.close() 
        
        # Save a simple MSE by Tool plot
        plt.figure(figsize=(10, 6))
        tool_names = ['oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither']
        for tool in tool_names:
            tool_mses = [m.get(f'{tool}_mse', np.nan) for m in self.history['val_metrics']]
            plt.plot(tool_mses, label=tool.title(), linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Validation MSE by Tool')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_dir / 'MSE_by_Tool_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save a simple MAPE by Tool plot
        plt.figure(figsize=(10, 6))
        tool_names = ['oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither']
        for tool in tool_names:
            tool_mapes = [m.get(f'{tool}_mape', np.nan) for m in self.history['val_metrics']]
            plt.plot(tool_mapes, label=tool.title(), linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('MAPE (%)')
        plt.title('Validation MAPE by Tool')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_dir / 'MAPE_by_Tool_curve.png', dpi=300, bbox_inches='tight')
        plt.close() 
        
        # Save a simple RMSE by Tool plot
        plt.figure(figsize=(10, 6))
        tool_names = ['oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither']
        for tool in tool_names: 
            tool_rmses = [m.get(f'{tool}_rmse', np.nan) for m in self.history['val_metrics']]
            plt.plot(tool_rmses, label=tool.title(), linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('Validation RMSE by Tool')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_dir / 'RMSE_by_Tool_curve.png', dpi=300, bbox_inches='tight')
        plt.close() 
        
        # Save a simple Current Validation Performance plot
        plt.figure(figsize=(10, 6))
        latest_val = self.history['val_metrics'][-1] if self.history['val_metrics'] else {}

        metric_names = ['TPR', 'FPR', 'Precision', 'Recall']     # 'Rec', 'F1'
        metric_values = [latest_val.get(f'{m.lower()}_avg_mae', 0) for m in metric_names]

        bars = plt.bar(metric_names, metric_values, alpha=0.8, color='steelblue') 
        plt.xlabel('Metric')
        plt.ylabel('MAE')
        plt.title(f'Current Validation Performance (Epoch {epoch})')

        # # Add value labels to bars
        # for bar, value in zip(bars, metric_values):
        #     if value > 0:
        #         plt.text(bar.get_x() + bar.get_width()/2., value,
        #                 f'{value:.4f}', ha='center', va='bottom')

        plt.grid(True, axis='y', alpha=0.3)

        # Avoid using spaces in file names
        plt.savefig(plot_dir / 'current_validation_performance_curve.png', dpi=300, bbox_inches='tight')
        plt.close() 
        
        # Save a simple Summary of final MAPE and RMSE plot
        plt.figure(figsize=(10, 6))
        latest_val = self.history['val_metrics'][-1] if self.history['val_metrics'] else {}
        summary_names = ['MAPE', 'RMSE']
        summary_vals = [
            latest_val.get('overall_mape', 0),
            latest_val.get('overall_rmse', 0)
        ]

        bars = plt.bar(summary_names, summary_vals, alpha=0.8, color=['darkorange', 'royalblue'])
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.title(f'Final Validation MAPE and RMSE (Epoch {epoch})')

        # Add value labels to bars
        for bar, value in zip(bars, summary_vals):
            if value > 0:
                plt.text(bar.get_x() + bar.get_width()/2., value,
                        f'{value:.3f}', ha='center', va='bottom')

        plt.grid(True, axis='y', alpha=0.3)

        # Avoid using spaces in file names
        plt.savefig(plot_dir / 'Summary of final MAPE and RMSE_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

    
    def save_training_history(self):
        """Save complete training history to file."""
        history_path = Path('checkpoints') / 'training_history.json'
        
        # Convert numpy values to Python native types
        def convert_to_native(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj
        
        with open(history_path, 'w') as f:
            json.dump(convert_to_native(self.history), f, indent=2)
        
        logger.info(f"Training history saved to {history_path}")

    
    # SAFE LOGGING 
    def safe_float(value, precision=4):
        try:
            return f"{float(value):.{precision}f}"
        except (ValueError, TypeError):
            return str(value)

    
    def test(self) -> Dict:
        
            # SAFE LOGGING 
        def safe_float(value, precision=4):
            try:
                return f"{float(value):.{precision}f}"
            except (ValueError, TypeError):
                return str(value)
        
        """Test the model on the test set with F1 score evaluation."""
        if not self.test_loader:
            logger.info("No test data available.")
            return {}
        
        logger.info("Testing model for tool performance prediction...")

        # Load best model
        try:
            checkpoint = self.model_checkpoint.load_best_checkpoint()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded best model checkpoint for testing")
        except FileNotFoundError:
            logger.warning("No best model checkpoint found, using current model")

        self.model.eval()
        num_samples = 0
        num_batches = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc='Testing')):
                try:
                    batch = batch.to(self.device)
                    batch_size = batch['contract'].x.shape[0]
                    num_samples += batch_size

                    # Forward pass and tuple unpacking (Fix 2)
                    # predictions = self.model(batch)
                    # if isinstance(predictions, tuple):
                    #     predictions = predictions[0]  # Use only the relevant output
                    
                    raw_predictions = self.model(batch)
                    if isinstance(raw_predictions, tuple):
                        predictions = raw_predictions[0]
                    else:
                        predictions = raw_predictions


                    # Get targets
                    targets = batch['contract'].y_tool

                    # Move to CPU
                    pred_tensor = predictions.detach().cpu()
                    target_tensor = targets.detach().cpu()

                    # Ensure 2D shape
                    if pred_tensor.dim() == 1:
                        pred_tensor = pred_tensor.unsqueeze(0)
                    if target_tensor.dim() == 1:
                        target_tensor = target_tensor.unsqueeze(0)

                    # Align batch size (Fix 1)
                    if pred_tensor.shape[0] != target_tensor.shape[0]:
                        logger.warning(
                            f"[Test] Shape mismatch: preds={pred_tensor.shape}, targets={target_tensor.shape}"
                        )
                        min_len = min(pred_tensor.shape[0], target_tensor.shape[0])
                        pred_tensor = pred_tensor[:min_len]
                        target_tensor = target_tensor[:min_len]

                    # Align feature count if necessary
                    if pred_tensor.shape[1] != target_tensor.shape[1]:
                        min_features = min(pred_tensor.shape[1], target_tensor.shape[1])
                        pred_tensor = pred_tensor[:, :min_features]
                        target_tensor = target_tensor[:, :min_features]

                    all_predictions.append(pred_tensor)
                    all_targets.append(target_tensor)

                    logger.debug(f"[Test] Batch {batch_idx} - preds: {pred_tensor.shape}, targets: {target_tensor.shape}")
                    num_batches += 1

                except Exception as e:
                    logger.error(f"Error in test batch {batch_idx}: {e}")
                    continue

        logger.info(f"Test completed: {num_batches} batches, {num_samples} samples")
        test_metrics = {}

        if all_predictions and all_targets:
            try:
                all_preds = torch.cat(all_predictions, dim=0)
                all_targs = torch.cat(all_targets, dim=0)

                logger.info(f"Test shapes - Predictions: {all_preds.shape}, Targets: {all_targs.shape}")

                test_metrics = self.metrics_calculator.calculate_tool_metrics(all_preds, all_targs, detailed=True)

                logger.info("Test Results:")
                # logger.info(f"Overall MSE: {test_metrics.get('overall_mse', 'N/A'):.4f}")
                # logger.info(f"Overall MAE: {test_metrics.get('overall_mae', 'N/A'):.4f}")
                logger.info(f"Overall MSE: {test_metrics.get('overall_mse', 'N/A')}")
                logger.info(f"Overall MAE: {test_metrics.get('overall_mae', 'N/A')}") 
                logger.info(f"Overall RMSE: {test_metrics.get('overall_rmse', 'N/A')}")
                logger.info(f"Overall MAPE: {test_metrics.get('overall_mape', 'N/A')}")

                # Per-metric
                metric_names = ['tpr', 'fpr', 'accuracy', 'precision', 'recall', 'f1_score']
                for metric in metric_names:
                    mae = test_metrics.get(f'{metric}_avg_mae', 'N/A')
                    mse = test_metrics.get(f'{metric}_avg_mse', 'N/A')
                    r2 = test_metrics.get(f'{metric}_avg_r2', 'N/A') 
                    rmse = test_metrics.get(f'{metric}_avg_rmse', 'N/A')
                    mape = test_metrics.get(f'{metric}_avg_mape', 'N/A')
                    
                    if mae != 'N/A':
                        # logger.info(f"{metric.upper()} - MAE: {mae:.4f}, MSE: {mse:.4f}, R^2: {r2:.4f}")
                        logger.info(f"{metric.upper()} - MAE: {safe_float(mae)}, MSE: {safe_float(mse)}, R^2: {safe_float(r2)}, RMSE: {safe_float(rmse)}, MAPE: {safe_float(mape)}")

                # Per-tool
                tool_names = ['oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither']
                logger.info("\nPer-Tool Performance:")
                for tool in tool_names:
                    mae = test_metrics.get(f'{tool}_mae', 'N/A')
                    mse = test_metrics.get(f'{tool}_mse', 'N/A')
                    r2 = test_metrics.get(f'{tool}_r2', 'N/A')
                    rmse = test_metrics.get(f'{tool}_rmse', 'N/A')
                    mape = test_metrics.get(f'{tool}_mape', 'N/A')
                    
                    if mae != 'N/A':
                        # logger.info(f"{tool.title()} - MAE: {mae:.4f}, MSE: {mse:.4f}, R^2: {r2:.4f}")
                        logger.info(f"{tool.title()} - MAE: {safe_float(mae)}, MSE: {safe_float(mse)}, R^2: {safe_float(r2)}, RMSE: {safe_float(rmse)}, MAPE: {safe_float(mape)}")

            except Exception as e:
                logger.error(f"Error calculating test metrics: {e}")
                test_metrics = {}

        return test_metrics
    

def main():
    parser = argparse.ArgumentParser(description='Train HeteroToolGNN for Tool Performance Prediction')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Set debug logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        sys.exit(1)
    
    config['use_wandb'] = args.wandb
    
    # Create necessary directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('training_plots', exist_ok=True)
    os.makedirs(config['data'].get('cache_dir', 'cache'), exist_ok=True)
    
    try:
        # Initialize trainer
        trainer = ToolPerformanceTrainer(config)
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if trainer.scheduler and checkpoint.get('scheduler_state_dict'):
                trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            logger.info(f"Resumed from epoch {start_epoch}")
        
        # Train model
        trainer.train()
        
        # Test model
        test_metrics = trainer.test()
        
        # Save final results
        if test_metrics:
            results_dir = Path('results')
            results_dir.mkdir(exist_ok=True)
            
            # Save as YAML
            with open(results_dir / 'tool_performance_results.yaml', 'w') as f:
                yaml.dump(test_metrics, f)
            
            # Save as JSON with proper formatting
            def convert_to_native(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_to_native(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(item) for item in obj]
                else:
                    return obj
            
            with open(results_dir / 'test_results.json', 'w') as f:
                json.dump(convert_to_native(test_metrics), f, indent=2)
            
            # Create a summary report
            report = trainer.metrics_calculator.create_performance_report(test_metrics)
            with open(results_dir / 'test_report.md', 'w') as f:
                f.write(report)
            
            logger.info(f"Test results saved to {results_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()