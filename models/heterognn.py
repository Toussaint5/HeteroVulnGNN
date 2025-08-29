import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from .layers import HeteroConv, EdgeTypeConv, MultiHeadAttention, ContractPooling

class HeteroToolGNN(nn.Module):
    """Heterogeneous Graph Neural Network for Smart Contract Tool Performance Prediction."""
    
    def __init__(self, config: Dict):
        super(HeteroToolGNN, self).__init__()
        self.config = config
        self.hidden_dim = config['model']['hidden_dim']
        self.num_layers = config['model']['num_layers']
        self.num_heads = config['model']['num_heads']
        self.dropout = config['model']['dropout']
        self.node_types = config['model']['node_types']
        self.edge_types = config['model']['edge_types']
        self.num_tools = config['model']['num_tools']
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Input projections for each node type
        self.node_projections = nn.ModuleDict({
            node_type: nn.Linear(self._get_input_dim(node_type), self.hidden_dim)
            for node_type in self.node_types
        })
        
        # Contract-level features projection
        # Increased input dimension to accommodate the enhanced features
        self.contract_projection = nn.Linear(
            self._get_contract_input_dim(), self.hidden_dim
        )
        
        # Heterogeneous GNN layers
        self.hetero_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            convs = {}
            for src_type in self.node_types:
                for dst_type in self.node_types:
                    for edge_type in self.edge_types:
                        edge_key = (src_type, edge_type, dst_type)
                        convs[edge_key] = EdgeTypeConv(
                            self.hidden_dim, self.hidden_dim
                        )
            self.hetero_layers.append(HeteroConv(convs))
        
        # Normalization layers
        self.norms = nn.ModuleList([
            nn.ModuleDict({
                node_type: nn.LayerNorm(self.hidden_dim)
                for node_type in self.node_types
            }) for _ in range(self.num_layers)
        ])
        
        # Multi-head attention for cross-type interactions
        self.cross_attention = MultiHeadAttention(
            self.hidden_dim, self.num_heads, self.dropout
        )
        
        # Contract-level pooling
        self.contract_pooling = ContractPooling(
            self.hidden_dim, pool_type='attention'
        )
        
        # Tool-specific attention mechanisms (especially for Securify)
        self.tool_attention = nn.ModuleDict({
            'securify': nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            ),
            'fpr_focus': nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
        })
        
        # Tool performance prediction heads
        self._build_tool_head()
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def _get_input_dim(self, node_type: str) -> int:
        """Get input dimension for a node type."""
        return {
            'function': 21,  # Based on _extract_node_features
            'statement': 21,
            'expression': 21,
            'variable': 21
        }[node_type]
    
    def _get_contract_input_dim(self) -> int:
        """Get input dimension for contract-level features."""
        # Original + FPR-specific features + Securify-specific features
        return 22 + 5 + 5  # Now includes additional feature sets
    
    def _build_tool_head(self):
        """Build tool performance prediction head."""
        # Enhanced TPR prediction head
        self.tpr_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.num_tools),
            nn.Sigmoid()  # TPR is bounded [0,1]
        )
        
        # Enhanced FPR prediction head with extra layers
        self.fpr_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.num_tools),
            nn.Sigmoid()  # FPR is bounded [0,1]
        )
        
        # Accuracy head
        self.accuracy_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.num_tools),
            nn.Sigmoid()  # Accuracy is bounded [0,1]
        )
        
        # Precision head
        self.precision_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.num_tools),
            nn.Sigmoid()  # Precision is bounded [0,1]
        )
        
        # Recall head
        self.recall_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.num_tools),
            nn.Sigmoid()  # Recall is bounded [0,1]
        )
        
        # Securify-specific prediction heads
        self.securify_heads = nn.ModuleDict({
            'tpr': nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim // 2, 1),
                nn.Sigmoid()
            ),
            'fpr': nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim // 2, 1),
                nn.Sigmoid()
            ),
            'accuracy': nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim // 2, 1),
                nn.Sigmoid()
            ),
            'precision': nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim // 2, 1),
                nn.Sigmoid()
            ),
            'recall': nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim // 2, 1),
                nn.Sigmoid()
            ),
        })
            
        
    def forward(self, batch_data, return_embeddings: bool = False):
        """Forward pass of the model."""
        
        # Extract node features and edge indices
        x_dict = {}
        edge_index_dict = {}
        
        # Project input features - only for node types that exist in the batch
        for node_type in self.node_types:
            if node_type in batch_data and hasattr(batch_data[node_type], 'x'):
                if batch_data[node_type].x.size(0) > 0:  # Check if we have nodes
                    x_dict[node_type] = self.node_projections[node_type](
                        batch_data[node_type].x
                    )
        
        # If no node types found, create a default representation
        if not x_dict:
            # Create a minimal representation with function nodes
            x_dict['function'] = self.node_projections['function'](
                torch.zeros(1, self._get_input_dim('function')).to(self.device)
            )
        
        # Extract edge indices - only if they exist
        if hasattr(batch_data, 'edge_types'):
            for edge_type_key in batch_data.edge_types:
                if hasattr(batch_data[edge_type_key], 'edge_index'):
                    edge_index_dict[edge_type_key] = batch_data[edge_type_key].edge_index
        
        # Apply heterogeneous GNN layers
        for i, (hetero_layer, norm_dict) in enumerate(zip(self.hetero_layers, self.norms)):
            # Only apply if we have edges
            if edge_index_dict:
                # Graph convolution
                out_dict = hetero_layer(x_dict, edge_index_dict)
                
                # Add residual connections and normalization (avoid in-place operations)
                # for node_type in out_dict:
                #     if node_type in x_dict:
                #         # Use addition instead of in-place addition
                #         out_dict[node_type] = out_dict[node_type] + x_dict[node_type]
                #     if node_type in norm_dict:
                #         out_dict[node_type] = norm_dict[node_type](out_dict[node_type])
                #     out_dict[node_type] = F.relu(out_dict[node_type])
                #     out_dict[node_type] = self.dropout_layer(out_dict[node_type])
                
                # x_dict = out_dict 
                
                new_x_dict = {}
                for node_type in out_dict:
                    # Residual connection
                    if node_type in x_dict:
                        new_features = out_dict[node_type] + x_dict[node_type]
                    else:
                        new_features = out_dict[node_type]
                    
                    # Normalization
                    if node_type in norm_dict:
                        new_features = norm_dict[node_type](new_features)
                    
                    # Activation and dropout
                    new_features = F.relu(new_features)
                    new_features = self.dropout_layer(new_features)
                    
                    new_x_dict[node_type] = new_features
                
                x_dict = new_x_dict
                
            else:
                # If no edges, just apply normalization and activation
                new_x_dict = {}
                # for node_type in x_dict:
                #     if node_type in norm_dict:
                #         new_x_dict[node_type] = norm_dict[node_type](x_dict[node_type])
                #     else:
                #         new_x_dict[node_type] = x_dict[node_type]
                #     new_x_dict[node_type] = F.relu(new_x_dict[node_type])
                #     new_x_dict[node_type] = self.dropout_layer(new_x_dict[node_type])
                # x_dict = new_x_dict 
                for node_type in x_dict:
                    features = x_dict[node_type]
                    
                    if node_type in norm_dict:
                        features = norm_dict[node_type](features)
                    
                    features = F.relu(features)
                    features = self.dropout_layer(features)
                    
                    new_x_dict[node_type] = features
                
                x_dict = new_x_dict
        
        # Cross-type attention - only if we have multiple node types
        if len(x_dict) > 1:
            # Concatenate all node features
            all_features = []
            node_type_indices = {}
            current_idx = 0
            
            for node_type, features in x_dict.items():
                all_features.append(features)
                node_type_indices[node_type] = (current_idx, current_idx + features.size(0))
                current_idx += features.size(0)
            
            if all_features:
                combined_features = torch.cat(all_features, dim=0)
                attended_features, _ = self.cross_attention(
                    combined_features, combined_features, combined_features
                )
                
                # Split back to node types
                new_x_dict = {}
                for node_type, (start_idx, end_idx) in node_type_indices.items():
                    new_x_dict[node_type] = attended_features[start_idx:end_idx]
                x_dict = new_x_dict
        
        # Contract-level pooling
        contract_embedding = self.contract_pooling(x_dict)
        
        # Add contract-level features if available
        if 'contract' in batch_data and hasattr(batch_data['contract'], 'x'):
            contract_features = self.contract_projection(batch_data['contract'].x)
            contract_embedding = contract_embedding + contract_features
        
        # Apply tool-specific attention for Securify
        securify_embedding = None
        if 'securify' in self.tool_attention:
            attention_weights = torch.sigmoid(self.tool_attention['securify'](contract_embedding))
            securify_embedding = contract_embedding * attention_weights
        
        # Apply FPR-focused attention
        fpr_embedding = None
        if 'fpr_focus' in self.tool_attention:
            attention_weights = torch.sigmoid(self.tool_attention['fpr_focus'](contract_embedding))
            fpr_embedding = contract_embedding * attention_weights
        
        # Tool performance predictions with separate heads
        tool_predictions = {}
        
        # Standard predictions
        tpr_pred = self.tpr_head(contract_embedding)
        fpr_pred = self.fpr_head(fpr_embedding if fpr_embedding is not None else contract_embedding)
        accuracy_pred = self.accuracy_head(contract_embedding)
        precision_pred = self.precision_head(contract_embedding)
        recall_pred = self.recall_head(contract_embedding)
        
        # Apply Securify-specific predictions (create new tensors instead of modifying existing ones)
        if securify_embedding is not None:
            securify_idx = 1  # Assuming Securify is the second tool (index 1)
            
            # Get Securify-specific predictions
            securify_tpr = self.securify_heads['tpr'](securify_embedding)
            securify_fpr = self.securify_heads['fpr'](securify_embedding)
            securify_accuracy = self.securify_heads['accuracy'](securify_embedding)
            securify_precision = self.securify_heads['precision'](securify_embedding)
            securify_recall = self.securify_heads['recall'](securify_embedding)
            
            # Create new tensors with Securify replacements (avoid in-place operations)
            tpr_pred = self._replace_tool_prediction(tpr_pred, securify_tpr, securify_idx)
            fpr_pred = self._replace_tool_prediction(fpr_pred, securify_fpr, securify_idx)
            accuracy_pred = self._replace_tool_prediction(accuracy_pred, securify_accuracy, securify_idx)
            precision_pred = self._replace_tool_prediction(precision_pred, securify_precision, securify_idx)
            recall_pred = self._replace_tool_prediction(recall_pred, securify_recall, securify_idx)
        
        # Store predictions
        tool_predictions = {
            'tpr': tpr_pred,
            'fpr': fpr_pred,
            'accuracy': accuracy_pred,
            'precision': precision_pred,
            'recall': recall_pred
        } 
        
        # FIXED: Combine predictions in the correct order
        # Each tool has 5 metrics: [tpr, fpr, accuracy, precision, recall]
        batch_size = tpr_pred.shape[0]
        num_tools = tpr_pred.shape[1]
        
        # Reshape to interleave metrics properly
        all_predictions = torch.zeros(batch_size, num_tools * 5, device=tpr_pred.device)
        
        for tool_idx in range(num_tools):
            base_idx = tool_idx * 5
            all_predictions[:, base_idx] = tpr_pred[:, tool_idx]
            all_predictions[:, base_idx + 1] = fpr_pred[:, tool_idx]
            all_predictions[:, base_idx + 2] = accuracy_pred[:, tool_idx]
            all_predictions[:, base_idx + 3] = precision_pred[:, tool_idx]
            all_predictions[:, base_idx + 4] = recall_pred[:, tool_idx]
        
        if return_embeddings:
            embeddings = {
                'contract_embedding': contract_embedding,
                'node_embeddings': x_dict
            }
            if securify_embedding is not None:
                embeddings['securify_embedding'] = securify_embedding
            if fpr_embedding is not None:
                embeddings['fpr_embedding'] = fpr_embedding
                
            return all_predictions, tool_predictions, embeddings
        
        return all_predictions, tool_predictions
    
    
    def _replace_tool_prediction(self, original_pred: torch.Tensor, 
                            tool_specific_pred: torch.Tensor, 
                            tool_idx: int) -> torch.Tensor:
        """Replace prediction for a specific tool without in-place operations."""
        # Create a new tensor instead of modifying in-place
        new_pred = original_pred.clone()
        if tool_specific_pred.dim() == 2 and tool_specific_pred.shape[1] == 1:
            new_pred[:, tool_idx] = tool_specific_pred.squeeze(-1)
        else:
            new_pred[:, tool_idx] = tool_specific_pred
        return new_pred
    
    def get_attention_weights(self, batch_data):
        """Extract attention weights for visualization."""
        attention_weights = {}
        
        # Hook to capture attention weights
        def hook_fn(module, input, output):
            if len(output) == 2:  # (output, attention_weights)
                attention_weights['cross_attention'] = output[1]
        
        # Register hook
        handle = self.cross_attention.register_forward_hook(hook_fn)
        
        # Forward pass
        _ = self.forward(batch_data)
        
        # Remove hook
        handle.remove()
        
        return attention_weights
    
    def reset_parameters(self):
        """Reset all model parameters."""
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()