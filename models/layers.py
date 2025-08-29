import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, degree
from typing import Dict, List, Tuple, Optional, Union

class HeteroConv(nn.Module):
    """Heterogeneous graph convolution layer."""
    
    def __init__(self, convs: Dict[Tuple[str, str, str], nn.Module]):
        super().__init__()
        self.convs = nn.ModuleDict()
        for edge_type, conv in convs.items():
            self.convs['__'.join(edge_type)] = conv
    
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                edge_attr_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None):
        
        out_dict = {}
        
        for edge_type, edge_index in edge_index_dict.items():
            src_type, rel_type, dst_type = edge_type
            
            if src_type not in x_dict or dst_type not in x_dict:
                continue
            
            conv_key = '__'.join(edge_type)
            if conv_key not in self.convs:
                continue
            
            # Get edge attributes if available
            edge_attr = edge_attr_dict.get(edge_type) if edge_attr_dict else None
            
            # Apply convolution
            if edge_attr is not None:
                out = self.convs[conv_key](
                    (x_dict[src_type], x_dict[dst_type]), 
                    edge_index, 
                    edge_attr
                )
            else:
                out = self.convs[conv_key](
                    (x_dict[src_type], x_dict[dst_type]), 
                    edge_index
                )
            
            # Aggregate outputs for destination type (avoid in-place operations)
            if dst_type not in out_dict:
                out_dict[dst_type] = []
            out_dict[dst_type].append(out)
        
        # Combine multiple messages for each node type (avoid in-place operations)
        final_out_dict = {}
        for node_type in out_dict:
            if len(out_dict[node_type]) == 1:
                final_out_dict[node_type] = out_dict[node_type][0]
            else:
                final_out_dict[node_type] = torch.stack(out_dict[node_type]).mean(0)
        
        return final_out_dict

class EdgeTypeConv(MessagePassing):
    """Edge-type specific graph convolution."""
    
    def __init__(self, in_channels: Union[int, Tuple[int, int]], 
                 out_channels: int, edge_dim: Optional[int] = None):
        super().__init__(aggr='mean', node_dim=0)
        
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        
        # Linear transformations
        self.lin_src = nn.Linear(in_channels[0], out_channels)
        self.lin_dst = nn.Linear(in_channels[1], out_channels)
        
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, out_channels)
        else:
            self.lin_edge = None
        
        # Attention mechanism
        self.att = nn.Linear(out_channels, 1)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        self.att.reset_parameters()
    
    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], 
                edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None):
        
        if isinstance(x, torch.Tensor):
            x = (x, x)
        
        # Transform source and destination features
        x_src = self.lin_src(x[0])
        x_dst = self.lin_dst(x[1])
        
        # Propagate messages
        out = self.propagate(edge_index, x=(x_src, x_dst), edge_attr=edge_attr)
        
        return out
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None):
        # x_i: destination nodes, x_j: source nodes
        
        # Combine source and destination features (avoid in-place operations)
        msg = x_j + x_i
        
        # Add edge features if available
        if edge_attr is not None and self.lin_edge is not None:
            edge_feat = self.lin_edge(edge_attr)
            msg = msg + edge_feat
        
        # Apply attention
        alpha = torch.sigmoid(self.att(msg))
        return alpha * msg
        
class MultiHeadAttention(nn.Module):
    """Multi-head attention for heterogeneous graphs."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size = query.size(0)
        
        # Transform inputs
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.hidden_dim
        )
        
        # Final linear transformation
        output = self.out_linear(context)
        
        return output, attention_weights

class ToolAwareAttention(nn.Module):
    """Tool-aware attention mechanism for performance prediction."""
    
    def __init__(self, hidden_dim: int, num_tools: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tools = num_tools
        
        # Tool-specific attention weights
        self.tool_attention = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_tools)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, node_features: torch.Tensor, 
                tool_idx: Optional[torch.Tensor] = None):
        
        if tool_idx is not None:
            # Apply tool-specific attention
            attended_features = []
            for i, tool_attn in enumerate(self.tool_attention):
                mask = (tool_idx == i)
                if mask.any():
                    tool_features = tool_attn(node_features[mask])
                    attended_features.append(tool_features)
                else:
                    attended_features.append(torch.empty(0, self.hidden_dim, device=node_features.device))
            
            # Combine features
            output = torch.cat(attended_features, dim=0)
        else:
            # Apply all tool attentions and average
            tool_outputs = [attn(node_features) for attn in self.tool_attention]
            output = torch.stack(tool_outputs).mean(0)
        
        return self.output_proj(output)

class ContractPooling(nn.Module):
    """Contract-level pooling for heterogeneous graphs."""
    
    def __init__(self, hidden_dim: int, pool_type: str = 'attention'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pool_type = pool_type
        
        if pool_type == 'attention':
            self.attention_weights = nn.Linear(hidden_dim, 1)
        elif pool_type == 'hierarchical':
            self.node_type_weights = nn.Parameter(torch.randn(4, 1))  # 4 node types
            
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                batch_dict: Optional[Dict[str, torch.Tensor]] = None):
        
        if self.pool_type == 'mean':
            pooled_features = []
            for node_type, features in x_dict.items():
                if batch_dict and node_type in batch_dict:
                    pooled = global_mean_pool(features, batch_dict[node_type])
                else:
                    pooled = features.mean(0, keepdim=True)
                pooled_features.append(pooled)
            
        elif self.pool_type == 'max':
            pooled_features = []
            for node_type, features in x_dict.items():
                if batch_dict and node_type in batch_dict:
                    pooled = global_max_pool(features, batch_dict[node_type])
                else:
                    pooled = features.max(0, keepdim=True)[0]
                pooled_features.append(pooled)
                
        elif self.pool_type == 'attention':
            pooled_features = []
            for node_type, features in x_dict.items():
                # Compute attention weights
                attn_weights = F.softmax(self.attention_weights(features), dim=0)
                # Weighted sum
                pooled = (attn_weights * features).sum(0, keepdim=True)
                pooled_features.append(pooled)
                
        elif self.pool_type == 'hierarchical':
            # Weight different node types differently
            node_type_order = ['function', 'statement', 'expression', 'variable']
            weighted_features = []
            
            for i, node_type in enumerate(node_type_order):
                if node_type in x_dict:
                    features = x_dict[node_type]
                    if batch_dict and node_type in batch_dict:
                        pooled = global_mean_pool(features, batch_dict[node_type])
                    else:
                        pooled = features.mean(0, keepdim=True)
                    
                    weighted = self.node_type_weights[i] * pooled
                    weighted_features.append(weighted)
            
            pooled_features = weighted_features
        
        # Combine all node type features (avoid in-place operations)
        if pooled_features:
            combined = torch.cat(pooled_features, dim=-1)
            # Project back to hidden_dim if necessary
            if combined.size(-1) != self.hidden_dim:
                # Add a projection layer if dimensions don't match
                if not hasattr(self, 'feature_proj'):
                    self.feature_proj = nn.Linear(combined.size(-1), self.hidden_dim).to(combined.device)
                combined = self.feature_proj(combined)
        else:
            combined = torch.zeros(1, self.hidden_dim, device=next(iter(x_dict.values())).device)
        
        return combined