import logging
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re
from transformers import AutoTokenizer, AutoModel

from utils import metrics

SOLIDIFI_BUG_TYPES = [
    'Re-entrancy',
    'Timestamp-Dependency',
    'Unchecked-Send',
    'Unhandled-Exceptions',
    'TOD',  # Transaction Order Dependence
    'Integer-Overflow-Underflow',
    'tx-origin'
] 

logger = logging.getLogger(__name__)

class SolidityCodeProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        self.max_length = 512 
        self.tool_names = ['oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither']
        
        # Initialize encoders
        self.node_type_encoder = LabelEncoder()
        self.edge_type_encoder = LabelEncoder()
        
        # Fit encoders with known types
        self.node_type_encoder.fit(config['model']['node_types'])
        self.edge_type_encoder.fit(config['model']['edge_types'])
        
    def _count_xpath_ambiguous_patterns(self, source_code: str) -> int:
        # Placeholder or actual pattern count logic for ambiguities in source code
        return len(re.findall(r'(\[\@|\/\/)[^\]]+\]', source_code))

    def _estimate_path_complexity(self, source_code: str) -> int:
        # Count branching constructs as a proxy for symbolic execution complexity
        count_if = source_code.count('if(')
        count_for = source_code.count('for(')
        count_while = source_code.count('while(')
        count_require = source_code.count('require(')
        return count_if + count_for + count_while + count_require
        
    def extract_solidifi_specific_features(self, source_code: str) -> Dict:
        """Extract features related to SolidiFI bug patterns and common static analysis pitfalls."""

        # Count specific Solidity constructs
        num_call_value = source_code.count('call.value')
        num_send = source_code.count('.send(')
        num_transfer = source_code.count('.transfer(')
        num_complex_ifs = len(re.findall(r'if\s*\(.*?(&&|\|\|).*?\)', source_code))

        features = {
            # Known false negative causes
            'has_call_value': num_call_value > 0,
            'has_send_function': num_send > 0,
            'has_transfer_function': num_transfer > 0,
            'complex_conditions': num_complex_ifs,

            # Additional patterns from SMARTSCOPY's observations
            'xpath_ambiguous_patterns': self._count_xpath_ambiguous_patterns(source_code),
            'symbolic_execution_complexity': self._estimate_path_complexity(source_code),

            # Optional: Add numeric counts as secondary features
            'count_call_value': num_call_value,
            'count_send': num_send,
            'count_transfer': num_transfer,
            'count_complex_ifs': num_complex_ifs,
        }

        return features 

        
    def extract_ast_features(self, ast: Dict) -> Dict:
        """Extract features from AST nodes for tool performance prediction."""
        if not ast:
            return {}
        
        features = {}
        
        def traverse_ast(node, parent_id=None):
            if not isinstance(node, dict):
                return
            
            node_id = node.get('id', len(features))
            node_type = node.get('nodeType', 'unknown')
            
            # Extract node features
            feature_vector = self._extract_node_features(node)
            
            # Get children list properly
            children_list = []
            if 'children' in node:
                children_list = [child.get('id', len(features) + i) 
                               for i, child in enumerate(node['children']) 
                               if isinstance(child, dict)]
            
            features[node_id] = {
                'type': node_type,
                'features': feature_vector,
                'parent': parent_id,
                'children': children_list,
                'source_location': node.get('src', ''),
                'code': node.get('name', '') or str(node_id)
            }
            
            if parent_id is not None and parent_id in features:
                features[parent_id]['children'].append(node_id)
            
            # Recursively process children
            if 'children' in node:
                for child in node['children']:
                    if isinstance(child, dict):
                        traverse_ast(child, node_id)
        
        traverse_ast(ast)
        return features
    
    def _extract_node_features(self, node: Dict) -> np.ndarray:
        """Extract numerical features from an AST node focused on tool performance."""
        features = []
        
        # Node type encoding
        node_type = node.get('nodeType', 'unknown')
        # Map unknown types to a default value
        if node_type in self.node_type_encoder.classes_:
            type_encoded = self.node_type_encoder.transform([node_type])[0]
        else:
            # Assign to one of the known types based on keywords
            if 'function' in node_type.lower():
                type_encoded = self.node_type_encoder.transform(['function'])[0]
            elif 'statement' in node_type.lower() or 'block' in node_type.lower():
                type_encoded = self.node_type_encoder.transform(['statement'])[0]
            elif 'expression' in node_type.lower():
                type_encoded = self.node_type_encoder.transform(['expression'])[0]
            elif 'variable' in node_type.lower():
                type_encoded = self.node_type_encoder.transform(['variable'])[0]
            else:
                type_encoded = 0  # Default to function
        features.append(type_encoded)
        
        # Boolean features - safely get with defaults
        # Feature selection focused on what might affect tool performance
        features.extend([
            int(node.get('constant', False)),
            int(node.get('payable', False)),
            int(node.get('stateMutability', '') == 'pure'),
            int(node.get('stateMutability', '') == 'view'),
            int(node.get('visibility', '') == 'public'),
            int(node.get('visibility', '') == 'private'),
            int(node.get('visibility', '') == 'internal'),
            int(node.get('visibility', '') == 'external'),
        ])
        
        # Numerical features - safely extract with defaults
        parameters = node.get('parameters', {})
        if isinstance(parameters, dict):
            param_count = len(parameters.get('parameters', []))
        else:
            param_count = 0
        
        return_params = node.get('returnParameters', {})
        if isinstance(return_params, dict):
            return_count = len(return_params.get('parameters', []))
        else:
            return_count = 0
        
        modifiers = node.get('modifiers', [])
        modifier_count = len(modifiers) if isinstance(modifiers, list) else 0
        
        body = node.get('body', {})
        if isinstance(body, dict):
            statement_count = len(body.get('statements', []))
        else:
            statement_count = 0
        
        features.extend([
            param_count,
            return_count,
            modifier_count,
            statement_count,
        ])
        
        # Code complexity features - focus on metrics that affect tool performance
        code = str(node.get('name', '')) + str(node.get('operator', ''))
        
        # Count occurrences of various keywords important for tools
        features.extend([
            len(code),
            code.lower().count('if'),
            code.lower().count('for'),
            code.lower().count('while'),
            code.lower().count('require'),
            code.lower().count('assert'),
            code.lower().count('msg.sender'),
            code.lower().count('now') + code.lower().count('block.timestamp'),
            code.lower().count('send') + code.lower().count('transfer') + code.lower().count('call'),
        ])
        
        # Ensure we have exactly 21 features
        while len(features) < 21:
            features.append(0.0)
        
        return np.array(features[:21], dtype=np.float32)
        
    def encode_source_code(self, source_code: str) -> torch.Tensor:
        """Encode source code using CodeBERT."""
        # Clean and truncate source code
        cleaned_code = re.sub(r'/\*.*?\*/', '', source_code, flags=re.DOTALL)
        cleaned_code = re.sub(r'//.*', '', cleaned_code)
        cleaned_code = re.sub(r'\s+', ' ', cleaned_code).strip()
        
        # Tokenize and encode
        inputs = self.tokenizer(
            cleaned_code,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return inputs['input_ids'].squeeze()
    
    
    def create_tool_performance_labels(self, contract_id: str,
                                    injected_bugs: Dict,
                                    tool_results: Dict,
                                    performance_metrics: Dict) -> torch.Tensor:
        """
        Create tool performance labels following the exact instructions:
        For each contract and tool, calculate TP, FP, FN, TN based on
        injected bugs vs detected bugs.
        """
        num_tools = len(self.tool_names)
        num_metrics = 5  # tpr, fpr, accuracy, precision, recall
        
        # Initialize flat tensor
        labels = torch.zeros(num_tools * num_metrics, dtype=torch.float32)
        
        # Get contract information to determine injected bugs
        contract_injected_bugs = self._get_contract_injected_bugs(contract_id, injected_bugs)
        
        # Process each tool
        for tool_idx, tool_name in enumerate(self.tool_names):
            base_idx = tool_idx * num_metrics
            
            # Get detected bugs for this contract by this tool
            contract_detected_bugs = self._get_contract_detected_bugs(contract_id, tool_name, tool_results)
            
            # Calculate confusion matrix for this contract-tool pair
            tp, fp, fn, tn = self._calculate_contract_confusion_matrix(
                contract_injected_bugs, contract_detected_bugs
            )
            
            # Calculate performance metrics with epsilon for numerical stability
            epsilon = 1e-6
            # tpr = tp / (tp + fn + epsilon)      # True Positive Rate (Recall)
            # fpr = fp / (fp + tn + epsilon)      # False Positive Rate  
            # accuracy = (tp + tn) / (tp + fp + tn + fn + epsilon)
            # precision = tp / (tp + fp + epsilon)
            # recall = tpr  # Same as TPR 
            
            tpr = min(1.0, max(0.0, tp / (tp + fn + epsilon)))      # True Positive Rate (Recall) 
            fpr = min(1.0, max(0.0, fp / (fp + tn + epsilon)))      # False Positive Rate 
            accuracy = min(1.0, max(0.0, (tp + tn) / (tp + fp + tn + fn + epsilon))) 
            precision = min(1.0, max(0.0, tp / (tp + fp + epsilon))) 
            recall = tpr  # Same as TPR 
            
            
            # Store in flat tensor: [tpr, fpr, accuracy, precision, recall]
            labels[base_idx:base_idx + num_metrics] = torch.tensor([
                tpr, fpr, accuracy, precision, recall
            ])
            
            # Debug logging for first few contracts
            if contract_id.endswith('_1.sol') or 'buggy_1' in contract_id:
                logger.debug(f"Contract {contract_id} - Tool {tool_name}:")
                logger.debug(f"  Injected: {contract_injected_bugs}")
                logger.debug(f"  Detected: {contract_detected_bugs}")
                logger.debug(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
                logger.debug(f"  TPR: {tpr:.3f}, FPR: {fpr:.3f}, Acc: {accuracy:.3f}")    
                
            if not all(k in metrics for k in ["tpr", "fpr", "accuracy"]):
                print(f"Missing tool metrics for contract: {contract_id}, tool: {tool_name}") 
                
            # Example normalization
            # labels = (labels - labels.mean()) / labels.std()
        
        return labels
    
    def _get_contract_injected_bugs(self, contract_id: str, injected_bugs: Dict) -> set:
        """Get injected vulnerabilities for a specific contract following SolidiFI structure."""
        contract_injected_bugs = set()
        
        # Method 1: Parse from contract ID (e.g., "Re-entrancy/buggy_1.sol")
        if '/' in contract_id:
            vulnerability_category = contract_id.split('/')[0]
            normalized_vuln = self._normalize_vulnerability_type(vulnerability_category)
            if normalized_vuln:
                contract_injected_bugs.add(normalized_vuln)
        
        # Method 2: Use injected_bugs mapping if available
        if contract_id in injected_bugs:
            for bug in injected_bugs[contract_id]:
                if isinstance(bug, dict) and 'type' in bug:
                    bug_type = self._normalize_vulnerability_type(bug['type'])
                    if bug_type:
                        contract_injected_bugs.add(bug_type)
        
        return contract_injected_bugs

    def _get_contract_detected_bugs(self, contract_id: str, tool_name: str, tool_results: Dict) -> set:
        """Get detected vulnerabilities for a specific contract by a specific tool."""
        detected_bugs = set()
        
        if contract_id not in tool_results:
            return detected_bugs
        
        if tool_name not in tool_results[contract_id]:
            return detected_bugs
        
        tool_result = tool_results[contract_id][tool_name]
        issues = tool_result.get('issues', [])
        
        for issue in issues:
            if isinstance(issue, dict):
                issue_type = issue.get('type', '')
            else:
                issue_type = str(issue)
            
            # Map the detected issue to standard vulnerability type
            mapped_type = self._map_detected_issue_to_standard_type(issue_type)
            if mapped_type:
                detected_bugs.add(mapped_type)
        
        return detected_bugs

    def _calculate_contract_confusion_matrix(self, injected_bugs: set, detected_bugs: set) -> Tuple[int, int, int, int]:
        """
        Calculate TP, FP, FN, TN for a single contract following exact instructions:
        - TP: Correctly detected injected bugs
        - FN: Missed injected bugs
        - FP: Reported bugs not in injected set  
        - TN: Correctly not reporting non-bugs
        """ 
        
        # True Negatives: vulnerability types that were correctly not flagged
        all_possible_vulns = {
            'reentrancy', 'integer_overflow', 'unchecked_return',
            'timestamp_dependency', 'tx_origin', 'unhandled_exception', 'tod'
        }
        
        # True Positives: injected bugs that were detected
        tp = len(injected_bugs.intersection(detected_bugs))
        
        # False Negatives: injected bugs that were NOT detected
        fn = len(injected_bugs - detected_bugs)
        
        # False Positives: detected bugs that were NOT injected
        fp = len(detected_bugs - injected_bugs) 
        
        # True Negatives: detected bugs that were injected
        tn = len(all_possible_vulns - injected_bugs - detected_bugs)
     
        return tp, fp, fn, tn

    def _normalize_vulnerability_type(self, vuln_type: str) -> Optional[str]:
        """Normalize vulnerability type to standard format."""
        if not vuln_type:
            return None
        
        vuln_lower = vuln_type.lower().strip()
        
        # Comprehensive mapping dictionary
        mappings = {
            're-entrancy': 'reentrancy',
            'reentrancy': 'reentrancy',
            'integer-overflow-underflow': 'integer_overflow',
            'overflow-underflow': 'integer_overflow',
            'integer_overflow': 'integer_overflow',
            'unchecked-send': 'unchecked_return',
            'unchecked_return_value': 'unchecked_return',
            'unchecked-return-value': 'unchecked_return',
            'timestamp-dependency': 'timestamp_dependency',
            'timestamp_dependency': 'timestamp_dependency',
            'tx-origin': 'tx_origin',
            'tx.origin': 'tx_origin',
            'authorization_through_tx_origin': 'tx_origin',
            'authorization-through-tx-origin': 'tx_origin',
            'unhandled-exceptions': 'unhandled_exception',
            'unhandled_exception': 'unhandled_exception',
            'tod': 'tod',
            'business_logic_error': 'business_logic',
            'business-logic-error': 'business_logic'
        }
        
        if vuln_lower in mappings:
            return mappings[vuln_lower]
        
        # Pattern-based fallback
        if 'reentr' in vuln_lower:
            return 'reentrancy'
        elif 'overflow' in vuln_lower or 'underflow' in vuln_lower:
            return 'integer_overflow'
        elif 'unchecked' in vuln_lower:
            return 'unchecked_return'
        elif 'timestamp' in vuln_lower or 'time' in vuln_lower:
            return 'timestamp_dependency'
        elif 'origin' in vuln_lower:
            return 'tx_origin'
        elif 'exception' in vuln_lower or 'unhandled' in vuln_lower:
            return 'unhandled_exception'
        elif 'tod' in vuln_lower:
            return 'tod'
        
        return None

    def _map_detected_issue_to_standard_type(self, issue_type: str) -> Optional[str]:
        """Map detected issue type to standard vulnerability type."""
        if not issue_type:
            return None
        
        issue_lower = issue_type.lower().strip()
        
        # Tool-specific detection mappings
        detection_mappings = {
            # Mythril
            'external call to user-supplied address': 'unchecked_return',
            'unchecked call return value': 'unchecked_return',
            'delegatecall to user-supplied address': 'reentrancy',
            'dependence on predictable environment variable': 'timestamp_dependency',
            'use of tx.origin': 'tx_origin',
            'integer overflow': 'integer_overflow',
            'integer underflow': 'integer_overflow',
            'exception disorder': 'unhandled_exception',
            
            # Slither
            'reentrancy-eth': 'reentrancy',
            'reentrancy-no-eth': 'reentrancy',
            'reentrancy-benign': 'reentrancy',
            'reentrancy-events': 'reentrancy',
            'timestamp': 'timestamp_dependency',
            'block-timestamp': 'timestamp_dependency',
            'tx-origin': 'tx_origin',
            'unchecked-lowlevel': 'unchecked_return',
            'unchecked-send': 'unchecked_return',
            
            # Oyente
            'callstack': 'reentrancy',
            'money_concurrency': 'reentrancy',
            'time_dependency': 'timestamp_dependency',
            'assertion_failure': 'unhandled_exception',
            
            # Securify
            'dao': 'reentrancy',
            'tod': 'tod',
            'todreceiver': 'tod',
            'todtransfer': 'tod',
            'unhandledexception': 'unhandled_exception',
            'lockedether': 'unchecked_return',
            
            # SmartCheck
            'reentrancy': 'reentrancy',
            'timestamp-dependency': 'timestamp_dependency',
            'tx.origin': 'tx_origin',
            'unchecked-send': 'unchecked_return',
        }
        
        # Direct mapping
        if issue_lower in detection_mappings:
            return detection_mappings[issue_lower]
        
        # Substring matching
        for pattern, vuln_type in detection_mappings.items():
            if pattern in issue_lower:
                return vuln_type
        
        # Fallback to general normalization
        return self._normalize_vulnerability_type(issue_type)


    
    def normalize_features(self, features: List[np.ndarray]) -> List[np.ndarray]:
        """Normalize numerical features."""
        if not features:
            return features
        
        scaler = StandardScaler()
        stacked_features = np.vstack(features)
        normalized = scaler.fit_transform(stacked_features)
        
        return [normalized[i] for i in range(len(features))]
    
    def extract_code_metrics(self, source_code: str) -> Dict:
        """Extract features that might affect tool performance."""
        metrics = {}
        
        # Basic metrics
        metrics['lines_of_code'] = len(source_code.split('\n'))
        metrics['characters'] = len(source_code)
        
        # Function metrics
        metrics['function_count'] = source_code.count('function ')
        metrics['modifier_count'] = source_code.count('modifier ')
        metrics['event_count'] = source_code.count('event ')
        
        # Structure metrics
        metrics['contract_count'] = source_code.count('contract ')
        metrics['interface_count'] = source_code.count('interface ')
        metrics['library_count'] = source_code.count('library ')
        
        # Tool-relevant patterns
        metrics['require_count'] = source_code.count('require(')
        metrics['assert_count'] = source_code.count('assert(')
        metrics['msg_sender_count'] = source_code.count('msg.sender')
        metrics['tx_origin_count'] = source_code.count('tx.origin')
        metrics['now_count'] = source_code.count('now')
        metrics['timestamp_count'] = source_code.count('block.timestamp')
        metrics['send_count'] = source_code.count('.send(')
        metrics['transfer_count'] = source_code.count('.transfer(')
        metrics['call_count'] = source_code.count('.call(')
        
        # Visibility metrics
        metrics['public_count'] = source_code.count(' public ')
        metrics['private_count'] = source_code.count(' private ')
        metrics['internal_count'] = source_code.count(' internal ')
        metrics['external_count'] = source_code.count(' external ')
        
        # Gas-related patterns that tools often analyze
        metrics['mapping_count'] = source_code.count('mapping(')
        metrics['storage_count'] = source_code.count(' storage ')
        metrics['memory_count'] = source_code.count(' memory ')
        metrics['loop_count'] = (source_code.count('for ') + 
                               source_code.count('while '))
        
        # FPR-specific metrics (patterns that often lead to false positives)
        metrics['complex_conditions'] = len(re.findall(r'if\s*\(.+?&&|\|\|.+?\)', source_code))
        metrics['nested_mappings'] = len(re.findall(r'mapping\s*\(\s*\w+\s*=>\s*mapping', source_code))
        metrics['inline_assembly'] = len(re.findall(r'assembly\s*{', source_code))
        metrics['complex_loops'] = len(re.findall(r'for\s*\(.+?\)\s*{.+?for\s*\(', source_code, re.DOTALL))
        metrics['modifiers_with_args'] = len(re.findall(r'modifier\s+\w+\s*\([^\)]+\)', source_code))
        
        return metrics
    
    def compute_securify_specific_features(self, source_code: str) -> Dict:
        """Extract features specifically relevant to Securify's analysis patterns."""
        securify_features = {
            # Based on Securify's known vulnerability patterns
            'reentrancy_patterns': len(re.findall(r'\.call{.*?}\(.*?\).*?\.transfer\(', source_code, re.DOTALL)),
            'unchecked_calls': len(re.findall(r'\.call\(.*?\)[^;]*?[^require|assert]', source_code)),
            'locked_ether': len(re.findall(r'(payable|receive|fallback).*?{.*?}', source_code, re.DOTALL)) and 
                           not len(re.findall(r'selfdestruct|suicide', source_code)),
            'tx_origin': len(re.findall(r'tx\.origin', source_code)),
            'timestamp_dependence': len(re.findall(r'block\.timestamp|now', source_code)),
        }
        
        return securify_features
    
    def create_global_contract_features(self, contract: Dict) -> torch.Tensor:
        """Create global features for contract-level tool performance prediction."""
        source_code = contract.get('source_code', '')
        
        # Extract code metrics with focus on tool performance
        code_metrics = self.extract_code_metrics(source_code)
        securify_features = self.compute_securify_specific_features(source_code)
        
        # Feature selection that might affect tool performance
        features = [
            code_metrics.get('lines_of_code', 0) / 1000,  # Normalized
            code_metrics.get('function_count', 0) / 10,
            code_metrics.get('modifier_count', 0) / 5,
            code_metrics.get('event_count', 0) / 5,
            
            # Features that often trigger tool warnings
            code_metrics.get('require_count', 0) / 10,
            code_metrics.get('assert_count', 0) / 5,
            code_metrics.get('msg_sender_count', 0) / 5,
            code_metrics.get('tx_origin_count', 0) / 2,
            code_metrics.get('now_count', 0) + code_metrics.get('timestamp_count', 0) / 3,
            code_metrics.get('send_count', 0) / 2,
            code_metrics.get('transfer_count', 0) / 2,
            code_metrics.get('call_count', 0) / 5,
            
            # Visibility metrics that affect tool coverage
            code_metrics.get('public_count', 0) / 10,
            code_metrics.get('private_count', 0) / 5,
            code_metrics.get('internal_count', 0) / 10,
            code_metrics.get('external_count', 0) / 5,
            
            # Code complexity metrics that affect tool performance
            code_metrics.get('mapping_count', 0) / 5,
            code_metrics.get('storage_count', 0) / 5, 
            code_metrics.get('memory_count', 0) / 10,
            code_metrics.get('loop_count', 0) / 5,
            len(source_code) / 10000,  # Code length normalized
            max(0, min(1, len(source_code) / 50000))  # Size factor (0-1)
        ]
        
        # Extract FPR-specific features
        fpr_features = [
            code_metrics.get('complex_conditions', 0) / 10,
            code_metrics.get('nested_mappings', 0) / 5,
            code_metrics.get('inline_assembly', 0) / 2,
            code_metrics.get('complex_loops', 0) / 3,
            code_metrics.get('modifiers_with_args', 0) / 5
        ]
        
        # Extract Securify-specific features
        securify_specific_features = [
            securify_features.get('reentrancy_patterns', 0) / 2,
            securify_features.get('unchecked_calls', 0) / 5,
            securify_features.get('locked_ether', 0) / 1,
            securify_features.get('tx_origin', 0) / 2,
            securify_features.get('timestamp_dependence', 0) / 3
        ]
        
        # Combine all features
        all_features = features + fpr_features + securify_specific_features
        
        return torch.tensor(all_features, dtype=torch.float32)
    
    def compute_tool_specific_features(self, source_code: str) -> Dict:
        """Extract features tailored to specific tool characteristics."""
        tool_features = {}
        
        # Mythril-specific features
        tool_features['mythril'] = {
            'symbolic_execution_points': source_code.count('if') + source_code.count('require('),
            'complex_math': source_code.count('*') + source_code.count('/'),
            'delegatecall_usage': source_code.count('delegatecall'),
            'storage_access': source_code.count('SSTORE') + source_code.count('SLOAD')
        }
        
        # Slither-specific features
        tool_features['slither'] = {
            'inheritance_depth': source_code.count('is '),
            'external_calls': source_code.count('.call('),
            'constant_modifiers': source_code.count(' constant '),
            'constructor_count': source_code.count('constructor')
        }
        
        # Oyente-specific features
        tool_features['oyente'] = {
            'exception_handling': source_code.count('try') + source_code.count('catch'),
            'transaction_order': source_code.count('block.number'),
            'reentrancy_guards': source_code.count('nonReentrant')
        }
        
        # Securify-specific features (enhanced)
        tool_features['securify'] = {
            'access_control': source_code.count('onlyOwner') + source_code.count('require(msg.sender'),
            'locked_ether': source_code.count('selfdestruct') + source_code.count('suicide'),
            'transaction_ordering': source_code.count('tx.origin'),
            'reentrancy_patterns': len(re.findall(r'\.call{.*?}\(.*?\).*?\.transfer\(', source_code, re.DOTALL)),
            'unchecked_calls': len(re.findall(r'\.call\(.*?\)[^;]*?[^require|assert]', source_code)),
            'timestamp_dependence': source_code.count('block.timestamp') + source_code.count('now'),
            'multiple_sends': len(re.findall(r'send\(.*?\).*?send\(', source_code, re.DOTALL)),
            'complex_fallback': len(re.findall(r'function\s+(\(\)|fallback)', source_code)) and source_code.count('.transfer('),
            'delegatecall_in_loop': len(re.findall(r'(for|while).*?delegatecall', source_code, re.DOTALL))
        }
        
        # Manticore-specific features
        tool_features['manticore'] = {
            'state_exploration': source_code.count('if') + source_code.count('else'),
            'symbolic_variables': source_code.count('uint') + source_code.count('int'),
            'memory_operations': source_code.count('memory') + source_code.count('calldata')
        }
        
        # Smartcheck-specific features
        tool_features['smartcheck'] = {
            'style_issues': source_code.count('var '),
            'pragma_directives': source_code.count('pragma'),
            'deprecated_features': source_code.count('throw') + source_code.count('sha3')
        }
        
        return tool_features  
    
    
