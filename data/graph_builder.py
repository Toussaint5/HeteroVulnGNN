import torch
import numpy as np
from torch_geometric.data import Data, HeteroData
from typing import Dict, List, Tuple, Optional
import networkx as nx
from collections import defaultdict
import re
import logging

logger = logging.getLogger(__name__)

# Set logging level to reduce edge validation warnings
logger.setLevel(logging.INFO)

class HeterogeneousGraphBuilder:
    def __init__(self, config: Dict):
        self.config = config
        self.node_types = config['model']['node_types']
        self.edge_types = config['model']['edge_types']
        
    def build_heterogeneous_graph(self, contract_data: Dict, 
                                ast_features: Dict,
                                processor) -> HeteroData:
        """Build heterogeneous graph from contract data for tool performance prediction."""
        
        # Initialize heterogeneous data object
        hetero_data = HeteroData()
        
        # Handle empty AST features
        if not ast_features:
            return self._create_minimal_graph(contract_data, processor)
        
        # Build nodes for each type
        node_mappings = {}
        for node_type in self.node_types:
            nodes, features, mapping = self._build_nodes_by_type(
                ast_features, node_type, processor
            )
            
            if len(nodes) > 0:
                hetero_data[node_type].x = features
                hetero_data[node_type].node_ids = torch.tensor(nodes)
                node_mappings[node_type] = mapping
                logger.debug(f"Created {len(nodes)} nodes of type '{node_type}'")
        
        # Ensure at least one node type exists
        if not node_mappings:
            return self._create_minimal_graph(contract_data, processor)
        
        # Build edges for each type - with validation
        for edge_type in self.edge_types:
            edges = self._build_edges_by_type(
                ast_features, edge_type, node_mappings
            )
            
            for (src_type, dst_type), edge_index in edges.items():
                if edge_index.shape[1] > 0:
                    # Validate edge indices
                    if self._validate_edge_indices(edge_index, src_type, dst_type, hetero_data):
                        edge_name = (src_type, edge_type, dst_type)
                        hetero_data[edge_name].edge_index = edge_index
                        logger.debug(f"Added {edge_index.shape[1]} edges of type '{edge_name}'")
                    else:
                        # Only log at DEBUG level to reduce console noise
                        logger.debug(f"Invalid edge indices for {src_type} -> {dst_type}, skipping")
        
        # Extract source code for additional features
        source_code = contract_data.get('source_code', '')
        
        # Get FPR-specific features from the processor
        fpr_specific_features = self._extract_fpr_features(source_code)
        
        # Get Securify-specific features from the processor
        securify_specific_features = processor.compute_securify_specific_features(source_code)
        
        # Add global contract features with enhanced information
        base_features = processor.create_global_contract_features(contract_data)
        
        # Create combined feature tensor including FPR and Securify-specific features
        contract_features = base_features  # This already includes the new features from processor
        
        # Add the contract features
        hetero_data['contract'].x = contract_features.unsqueeze(0)
        
        return hetero_data
        
    def _validate_edge_indices(self, edge_index: torch.Tensor, src_type: str, dst_type: str, hetero_data: HeteroData) -> bool:
        """Validate that edge indices point to existing nodes."""
        if src_type not in hetero_data or dst_type not in hetero_data:
            return False
        
        src_nodes = hetero_data[src_type].x.shape[0] if hasattr(hetero_data[src_type], 'x') else 0
        dst_nodes = hetero_data[dst_type].x.shape[0] if hasattr(hetero_data[dst_type], 'x') else 0
        
        if src_nodes == 0 or dst_nodes == 0:
            return False
        
        # Check if all source indices are valid
        if edge_index[0].max() >= src_nodes:
            if logger.level <= logging.DEBUG:  # Only log at DEBUG level
                logger.debug(f"Source edge index {edge_index[0].max()} >= {src_nodes} nodes")
            return False
        
        # Check if all destination indices are valid
        if edge_index[1].max() >= dst_nodes:
            if logger.level <= logging.DEBUG:  # Only log at DEBUG level
                logger.debug(f"Destination edge index {edge_index[1].max()} >= {dst_nodes} nodes")
            return False
        
        return True
    
    def _create_minimal_graph(self, contract_data: Dict, processor) -> HeteroData:
        """Create a minimal graph when AST features are not available."""
        hetero_data = HeteroData()
        
        # Create a single function node with default features
        hetero_data['function'].x = torch.zeros(1, 21)  # 21 features as expected
        hetero_data['function'].node_ids = torch.tensor([0])
        
        # Add contract-level features with enhanced information for FPR and Securify
        source_code = contract_data.get('source_code', '')
        contract_features = processor.create_global_contract_features(contract_data)
        
        # Add contract-level features
        hetero_data['contract'].x = contract_features.unsqueeze(0)
        
        return hetero_data
    
    def _build_nodes_by_type(self, ast_features: Dict, node_type: str, 
                           processor) -> Tuple[List, torch.Tensor, Dict]:
        """Build nodes of a specific type."""
        nodes = []
        features = []
        mapping = {}
        
        for node_id, node_data in ast_features.items():
            if self._classify_node_type(node_data) == node_type:
                nodes.append(node_id)
                features.append(node_data['features'])
                # Map original node ID to index in the feature tensor
                mapping[node_id] = len(mapping)
        
        if features:
            # Stack features and ensure they're the right size
            feature_array = np.vstack(features)
            if feature_array.shape[1] < 21:
                # Pad with zeros if needed
                padding = np.zeros((feature_array.shape[0], 21 - feature_array.shape[1]))
                feature_array = np.hstack([feature_array, padding])
            elif feature_array.shape[1] > 21:
                # Truncate if too many features
                feature_array = feature_array[:, :21]
            
            feature_tensor = torch.from_numpy(feature_array).float()
        else:
            # Create empty feature tensor with correct dimensions
            feature_tensor = torch.empty(0, 21)
        
        logger.debug(f"Built {len(nodes)} nodes of type '{node_type}' with mapping: {len(mapping)} entries")
        return nodes, feature_tensor, mapping
    
    def _classify_node_type(self, node_data: Dict) -> str:
        """Classify AST node into our predefined types."""
        ast_type = node_data['type'].lower()
        
        if any(func_type in ast_type for func_type in 
               ['function', 'constructor', 'modifier']):
            return 'function'
        elif any(stmt_type in ast_type for stmt_type in 
                ['statement', 'block', 'if', 'for', 'while', 'return']):
            return 'statement'
        elif any(expr_type in ast_type for expr_type in 
                ['expression', 'call', 'assignment', 'binary', 'unary']):
            return 'expression'
        elif any(var_type in ast_type for var_type in 
                ['variable', 'parameter', 'declaration']):
            return 'variable'
        else:
            return 'statement'  # Default classification
    
    def _build_edges_by_type(self, ast_features: Dict, edge_type: str,
                           node_mappings: Dict) -> Dict[Tuple[str, str], torch.Tensor]:
        """Build edges of a specific type with proper validation."""
        edges = defaultdict(list)
        
        if edge_type == 'control_flow':
            self._add_control_flow_edges(ast_features, node_mappings, edges)
        elif edge_type == 'data_flow':
            self._add_data_flow_edges(ast_features, node_mappings, edges)
        elif edge_type == 'call':
            self._add_call_edges(ast_features, node_mappings, edges)
        elif edge_type == 'dependency':
            self._add_dependency_edges(ast_features, node_mappings, edges)
        
        # Convert edge lists to tensors with validation
        edge_tensors = {}
        for (src_type, dst_type), edge_list in edges.items():
            if edge_list:
                # Validate edge indices before creating tensor
                valid_edges = []
                src_max = len(node_mappings.get(src_type, {}))
                dst_max = len(node_mappings.get(dst_type, {}))
                
                for edge in edge_list:
                    src_idx, dst_idx = edge
                    if 0 <= src_idx < src_max and 0 <= dst_idx < dst_max:
                        valid_edges.append(edge)
                    else:
                        # Only log at DEBUG level
                        if logger.level <= logging.DEBUG:
                            logger.debug(f"Skipping invalid edge {edge}: src_max={src_max}, dst_max={dst_max}")
                
                if valid_edges:
                    edge_tensors[(src_type, dst_type)] = torch.tensor(
                        valid_edges, dtype=torch.long
                    ).t()
                else:
                    edge_tensors[(src_type, dst_type)] = torch.empty(2, 0, dtype=torch.long)
            else:
                edge_tensors[(src_type, dst_type)] = torch.empty(2, 0, dtype=torch.long)
        
        return edge_tensors
    
    def _add_control_flow_edges(self, ast_features: Dict, 
                              node_mappings: Dict, edges: Dict):
        """Add control flow edges based on AST structure."""
        for node_id, node_data in ast_features.items():
            src_type = self._classify_node_type(node_data)
            
            if src_type not in node_mappings:
                continue
                
            src_idx = node_mappings[src_type].get(node_id)
            if src_idx is None:
                continue
            
            # Add edges to children (control flow)
            for child_id in node_data.get('children', []):
                if child_id in ast_features:
                    child_data = ast_features[child_id]
                    dst_type = self._classify_node_type(child_data)
                    
                    if dst_type in node_mappings:
                        dst_idx = node_mappings[dst_type].get(child_id)
                        if dst_idx is not None:
                            edges[(src_type, dst_type)].append([src_idx, dst_idx])
    
    def _add_data_flow_edges(self, ast_features: Dict, 
                           node_mappings: Dict, edges: Dict):
        """Add data flow edges based on variable usage."""
        variables = {}
        
        # First pass: collect variable definitions
        for node_id, node_data in ast_features.items():
            if 'variable' in node_data['type'].lower():
                var_name = node_data.get('code', '').strip()
                if var_name and len(var_name) > 1:  # Ignore single characters
                    variables[var_name] = node_id
        
        # Second pass: add edges for variable usage
        for node_id, node_data in ast_features.items():
            node_type = self._classify_node_type(node_data)
            code = node_data.get('code', '').strip()
            
            # Check if this node uses any variables
            for var_name, var_id in variables.items():
                if var_name and var_name in code and var_id != node_id and len(var_name) > 2:
                    var_type = self._classify_node_type(ast_features[var_id])
                    
                    # Add data flow edge from variable to usage
                    if (var_type in node_mappings and 
                        node_type in node_mappings):
                        
                        src_idx = node_mappings[var_type].get(var_id)
                        dst_idx = node_mappings[node_type].get(node_id)
                        
                        if src_idx is not None and dst_idx is not None:
                            edges[(var_type, node_type)].append([src_idx, dst_idx])
    
    def _add_call_edges(self, ast_features: Dict, 
                       node_mappings: Dict, edges: Dict):
        """Add function call edges."""
        functions = {}
        
        # Collect function definitions
        for node_id, node_data in ast_features.items():
            if 'function' in node_data['type'].lower():
                func_name = node_data.get('code', '').strip()
                if func_name and len(func_name) > 1:
                    functions[func_name] = node_id
        
        # Add call edges
        for node_id, node_data in ast_features.items():
            if 'call' in node_data['type'].lower() or 'expression' in node_data['type'].lower():
                node_type = self._classify_node_type(node_data)
                code = node_data.get('code', '').strip()
                
                # Try to match function calls
                for func_name, func_id in functions.items():
                    if func_name and func_name in code and len(func_name) > 2:
                        func_type = self._classify_node_type(ast_features[func_id])
                        
                        if (node_type in node_mappings and 
                            func_type in node_mappings):
                            
                            src_idx = node_mappings[node_type].get(node_id)
                            dst_idx = node_mappings[func_type].get(func_id)
                            
                            if src_idx is not None and dst_idx is not None:
                                edges[(node_type, func_type)].append([src_idx, dst_idx])
    
    def _add_dependency_edges(self, ast_features: Dict, 
                            node_mappings: Dict, edges: Dict):
        """Add dependency edges for patterns relevant to tool analysis."""
        # Patterns that may be relevant for tool performance
        tool_patterns = {
            'complex_computation': ['uint', 'int', '+', '-', '*', '/', '%'],
            'external_calls': ['call', 'transfer', 'send', 'address'],
            'storage_access': ['storage', 'memory', 'mapping'],
            'control_structures': ['require', 'assert', 'if', 'for', 'while'],
            # Additional patterns for FPR and Securify
            'fpr_triggers': ['complex', 'assembly', 'delegate', 'loop'],
            'securify_patterns': ['tx.origin', 'block.timestamp', 'now', 'selfdestruct']
        }
        
        pattern_nodes = defaultdict(list)
        
        # Classify nodes by tool-relevant patterns
        for node_id, node_data in ast_features.items():
            code = node_data.get('code', '').lower()
            node_type = self._classify_node_type(node_data)
            
            for pattern_name, keywords in tool_patterns.items():
                if any(keyword.lower() in code for keyword in keywords):
                    pattern_nodes[pattern_name].append((node_id, node_type))
        
        # Add edges between related patterns - limit to prevent explosion
        pattern_relations = [
            ('complex_computation', 'control_structures'),   # Computing might need validation
            ('external_calls', 'control_structures'),        # External calls often validated
            ('storage_access', 'complex_computation'),       # Storage often involves calculation
            ('fpr_triggers', 'control_structures'),          # FPR often related to control
            ('fpr_triggers', 'external_calls'),              # FPR often related to external calls
            ('securify_patterns', 'control_structures'),     # Securify patterns and control
            ('securify_patterns', 'external_calls')          # Securify patterns and external calls
        ]
        
        for pattern1, pattern2 in pattern_relations:
            # Limit number of edges to prevent graph explosion
            pattern1_nodes = pattern_nodes[pattern1][:5]  # Limit to 5 nodes
            pattern2_nodes = pattern_nodes[pattern2][:5]  # Limit to 5 nodes
            
            for node1_id, type1 in pattern1_nodes:
                for node2_id, type2 in pattern2_nodes:
                    if (type1 in node_mappings and type2 in node_mappings):
                        src_idx = node_mappings[type1].get(node1_id)
                        dst_idx = node_mappings[type2].get(node2_id)
                        
                        if src_idx is not None and dst_idx is not None:
                            edges[(type1, type2)].append([src_idx, dst_idx])
    
    def _extract_fpr_features(self, source_code: str) -> List[float]:
        """Extract features specifically related to False Positive Rate prediction."""
        # Patterns that often lead to false positives
        features = {
            'complex_conditions': len(re.findall(r'if\s*\(.+?&&|\|\|.+?\)', source_code)),
            'nested_mappings': len(re.findall(r'mapping\s*\(\s*\w+\s*=>\s*mapping', source_code)),
            'inline_assembly': len(re.findall(r'assembly\s*{', source_code)),
            'complex_loops': len(re.findall(r'for\s*\(.+?\)\s*{.+?for\s*\(', source_code, re.DOTALL)),
            'modifiers_with_args': len(re.findall(r'modifier\s+\w+\s*\([^\)]+\)', source_code))
        }
        
        # Normalize features
        normalized_features = [
            features.get('complex_conditions', 0) / 10,
            features.get('nested_mappings', 0) / 5,
            features.get('inline_assembly', 0) / 2,
            features.get('complex_loops', 0) / 3,
            features.get('modifiers_with_args', 0) / 5
        ]
        
        return normalized_features
    
    def add_tool_performance_labels(self, hetero_data: HeteroData,
                                  tool_labels: torch.Tensor) -> HeteroData:
        """Add tool performance labels to the graph."""
        # Ensure labels have the right shape [1, num_tools * 5] for single sample
        if tool_labels.dim() == 1:
            tool_labels = tool_labels.unsqueeze(0)
        hetero_data['contract'].y_tool = tool_labels
        return hetero_data
    
    def create_simple_graph_from_source(self, source_code: str, processor) -> HeteroData:
        """Create a simple graph directly from source code when AST is not available."""
        hetero_data = HeteroData()
        
        # Tokenize the source code
        tokens = processor.tokenizer.tokenize(source_code[:512])
        token_ids = processor.tokenizer.convert_tokens_to_ids(tokens)
        
        if not token_ids:
            # Create minimal graph with default features
            hetero_data['function'].x = torch.zeros(1, 21)
            hetero_data['function'].node_ids = torch.tensor([0])
            
            # Add contract features with enhanced FPR and Securify information
            contract_features = self._create_minimal_contract_features(source_code, processor)
            hetero_data['contract'].x = contract_features.unsqueeze(0)
            
            return hetero_data
        
        # Create function nodes (one per function keyword)
        function_positions = [i for i, token in enumerate(tokens) if 'function' in token.lower()]
        if not function_positions:
            function_positions = [0]  # At least one function node
        
        # Create function features
        function_features = []
        for i, pos in enumerate(function_positions):
            # Use tokens around function declaration
            start_idx = max(0, pos - 10)
            end_idx = min(len(token_ids), pos + 10)
            local_tokens = token_ids[start_idx:end_idx]
            
            # Pad or truncate to 21 features
            features = local_tokens + [0] * (21 - len(local_tokens))
            features = features[:21]
            function_features.append(features)
        
        hetero_data['function'].x = torch.tensor(function_features, dtype=torch.float32)
        hetero_data['function'].node_ids = torch.arange(len(function_features))
        
        # Create statement nodes with emphasis on FPR and Securify relevant statements
        statement_keywords = [
            'require', 'assert', 'if', 'for', 'while', 
            'tx.origin', 'block.timestamp', 'now',  # Securify-relevant
            'assembly', 'delegatecall'             # FPR-relevant
        ]
        statement_positions = []
        for keyword in statement_keywords:
            positions = [i for i, token in enumerate(tokens) if keyword in token.lower()]
            statement_positions.extend(positions)
        
        if statement_positions:
            statement_features = []
            for pos in statement_positions[:10]:  # Limit to 10 statements
                start_idx = max(0, pos - 5)
                end_idx = min(len(token_ids), pos + 5)
                local_tokens = token_ids[start_idx:end_idx]
                
                features = local_tokens + [0] * (21 - len(local_tokens))
                features = features[:21]
                statement_features.append(features)
            
            hetero_data['statement'].x = torch.tensor(statement_features, dtype=torch.float32)
            hetero_data['statement'].node_ids = torch.arange(len(statement_features))
            
            # Add control flow edges from functions to statements
            num_functions = len(function_features)
            num_statements = len(statement_features)
            
            # Create valid edges (ensure indices are within bounds)
            edges = []
            for i in range(min(num_functions, num_statements)):
                edges.append([i, i])
            
            # Add some additional edges between consecutive statements
            for i in range(min(num_statements - 1, 5)):  # Limit edges
                edges.append([min(i, num_functions - 1), i])
            
            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t()
                hetero_data['function', 'control_flow', 'statement'].edge_index = edge_index
        
        # Add enhanced contract features
        contract_features = processor.create_global_contract_features({
            'source_code': source_code
        })
        hetero_data['contract'].x = contract_features.unsqueeze(0)
        
        return hetero_data
    
    def _create_minimal_contract_features(self, source_code: str, processor) -> torch.Tensor:
        """Create minimal contract features when AST is not available."""
        # Use the processor to create features
        contract_data = {'source_code': source_code}
        return processor.create_global_contract_features(contract_data)