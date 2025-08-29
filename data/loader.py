import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from torch_geometric.data import Data, HeteroData
from transformers import RobertaTokenizer
from sklearn.model_selection import train_test_split
import re
from collections import Counter, defaultdict
import copy
import random
import numpy as np
import yaml

# SolidiFI bug types - comprehensive list with normalized versions
SOLIDIFI_BUG_TYPES = [
    'Re-entrancy',
    'Timestamp-Dependency', 
    'Unchecked-Send',
    'Unhandled-Exceptions',
    'TOD',  # Transaction Order Dependence
    'Integer-Overflow-Underflow',
    'tx-origin'
]

# Normalized versions for internal processing
NORMALIZED_BUG_TYPES = [
    'reentrancy',
    'timestamp_dependency',
    'unchecked_return', 
    'unhandled_exception',
    'tod',
    'integer_overflow',
    'tx_origin'
]

# Comprehensive vulnerability type mapping
VULNERABILITY_TYPE_MAPPINGS = {
    # Original SolidiFI names
    'Re-entrancy': 'reentrancy',
    'Reentrancy': 'reentrancy',
    'Integer-Overflow-Underflow': 'integer_overflow',
    'Overflow-Underflow': 'integer_overflow',
    'Integer_Overflow': 'integer_overflow',
    'Unchecked-Send': 'unchecked_return',
    'Unchecked_Return_Value': 'unchecked_return',
    'Unchecked-Return-Value': 'unchecked_return',
    'Timestamp-Dependency': 'timestamp_dependency',
    'Timestamp_Dependency': 'timestamp_dependency',
    'Authorization-Through-tx-origin': 'tx_origin',
    'Authorization_Through_tx_origin': 'tx_origin',
    'tx.origin': 'tx_origin',
    'tx-origin': 'tx_origin',
    'Unhandled-Exceptions': 'unhandled_exception',
    'Unhandled_Exception': 'unhandled_exception',
    'Business_Logic_Error': 'business_logic',
    'Business-Logic-Error': 'business_logic',
    'TOD': 'tod',
    
    # Lowercase versions
    're-entrancy': 'reentrancy',
    'reentrancy': 'reentrancy',
    'integer-overflow-underflow': 'integer_overflow',
    'overflow-underflow': 'integer_overflow',
    'integer_overflow': 'integer_overflow',
    'unchecked-send': 'unchecked_return',
    'unchecked_return_value': 'unchecked_return',
    'timestamp-dependency': 'timestamp_dependency',
    'timestamp_dependency': 'timestamp_dependency',
    'authorization-through-tx-origin': 'tx_origin',
    'authorization_through_tx_origin': 'tx_origin',
    'unhandled-exceptions': 'unhandled_exception',
    'unhandled_exception': 'unhandled_exception',
    'business_logic_error': 'business_logic',
    'tod': 'tod',
} 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SolidiFIDataLoader:
    """Fixed SolidiFI data loader with proper architecture and no circular dependencies."""
    
    def __init__(self, config):
        self.config = config
        self.processed_data_path = config.get('processed_data_path', 'results')
        self.solidifi_path = Path(config['data']['solidifi_path'])
        self.contracts_path = config['data'].get('contracts_path', 'buggy_contracts')
        self.results_path = self.solidifi_path / config['data'].get('results_path', 'results')
        self.cache_dir = Path(config['data'].get('cache_dir', 'cache'))
        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
        
        # Tool names - get from config
        self.tool_names = config.get('tools', [
            'oyente', 'securify', 'mythril', 'smartcheck', 'manticore', 'slither'
        ])
        
        # Create cache directory
        self.cache_dir.mkdir(exist_ok=True)
        
        # Vulnerability type mappings for SolidiFI
        self.vuln_type_mapping = {
            'reentrancy': ['Reentrancy', 'reentrancy', 'Re-entrancy'],
            'integer_overflow': ['Integer_Overflow', 'integer overflow', 'overflow', 'Integer-Overflow', 'Overflow-Underflow'],
            'unchecked_return': ['Unchecked_Return_Value', 'unchecked return', 'unchecked-send', 'Unchecked-Return-Value', 'Unchecked-Send'],
            'timestamp_dependency': ['Timestamp_Dependency', 'timestamp', 'block timestamp', 'Timestamp-Dependency'],
            'tx_origin': ['Authorization_Through_tx_origin', 'tx.origin', 'tx-origin', 'Authorization-Through-tx-origin'],
            'unhandled_exception': ['Unhandled_Exception', 'unhandled exception', 'exception', 'Unhandled-Exception', 'Unhandled-Exceptions'],
            'business_logic': ['Business_Logic_Error', 'business logic', 'logic error', 'Business-Logic-Error'],
            'tod': ['TOD', 'Transaction Order Dependency', 'transaction order']
        }
        
        # Initialize data containers to avoid circular dependencies
        self._contracts = None
        self._tool_results = None
        self._injected_bugs = None
        self._performance_metrics = None
        
        # Verify paths exist and explore structure
        self._verify_and_explore_paths()
        
        # Load all data in correct order
        self._load_all_data()
    
    def _verify_and_explore_paths(self):
        """Verify that SolidiFI path exists and explore its structure."""
        logger.info(f"Verifying SolidiFI path: {self.solidifi_path}")
        
        if not self.solidifi_path.exists():
            raise FileNotFoundError(
                f"SolidiFI benchmark not found at {self.solidifi_path}. "
                f"Please run: git clone https://github.com/DependableSystemsLab/SolidiFI-benchmark.git"
            )
        
        # Explore the actual structure
        logger.info("Exploring SolidiFI directory structure...")
        subdirs = [item.name for item in self.solidifi_path.iterdir() if item.is_dir()]
        logger.info(f"Found subdirectories: {subdirs}")
        
        # Check for buggy_contracts directory
        self.buggy_contracts_path = self.solidifi_path / self.contracts_path
        if not self.buggy_contracts_path.exists():
            logger.warning(f"buggy_contracts directory not found at {self.buggy_contracts_path}")
            # Try alternative locations
            for alt_path in ['contracts', 'buggy_contracts']:
                alt_full_path = self.solidifi_path / alt_path
                if alt_full_path.exists():
                    self.buggy_contracts_path = alt_full_path
                    logger.info(f"Using contracts directory: {self.buggy_contracts_path}")
                    break
        
        # Check results path
        if not self.results_path.exists():
            logger.warning(f"Results directory not found at {self.results_path}")
            # Try alternative location
            alt_results_path = self.solidifi_path / 'results'
            if alt_results_path.exists():
                self.results_path = alt_results_path
                logger.info(f"Using results directory: {self.results_path}")
        
        # Find vulnerability directories from the results structure
        self.vuln_dirs = []
        if self.results_path.exists():
            # Get vulnerability types from any tool's results
            for tool_name in ['Oyente', 'Securify', 'Mythril', 'Smartcheck', 'Manticore', 'Slither']:
                tool_path = self.results_path / tool_name / 'analyzed_buggy_contracts'
                if tool_path.exists():
                    for vuln_dir in tool_path.iterdir():
                        if vuln_dir.is_dir() and vuln_dir.name not in self.vuln_dirs:
                            self.vuln_dirs.append(vuln_dir.name)
                            logger.debug(f"Found vulnerability type: {vuln_dir.name}")
                    # if self.vuln_dirs:
                    #     break
        
        logger.info(f"Found {len(self.vuln_dirs)} vulnerability types: {self.vuln_dirs}")
    
    def _load_all_data(self):
        """Load all data in the correct order to avoid circular dependencies."""
        logger.info("Loading SolidiFI data in correct order...")
        
        # Step 1: Load contracts from directory structure
        logger.info("Step 1: Loading contracts from results structure...")
        self._contracts = self._load_contracts_from_results()
        
        # Step 2: Load tool results
        logger.info("Step 2: Loading tool results...")
        self._tool_results = self._load_tool_results_from_files()
        
        # Step 3: Create injected bugs mapping
        logger.info("Step 3: Creating injected bugs mapping...")
        self._injected_bugs = self._create_injected_bugs_mapping()
        
        # Step 4: Calculate performance metrics
        logger.info("Step 4: Calculating performance metrics...")
        self._performance_metrics = self._calculate_performance_metrics()
        
        logger.info(f"Data loading complete: {len(self._contracts)} contracts, "
                   f"{len(self._tool_results)} tool result sets, "
                   f"{len(self._injected_bugs)} bug mappings")
    
    def _load_contracts_from_results(self) -> List[Dict]:
        """Load contracts based on available tool results."""
        contracts = []
        contracts_seen = set()
        
        # Check cache first
        cache_file = self.cache_dir / 'contracts_cache.json'
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_contracts = json.load(f)
                logger.info(f"Loaded {len(cached_contracts)} contracts from cache")
                return cached_contracts
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        # Scan results directory to find available contracts
        for tool_name in self.tool_names:
            # Try both capitalized and lowercase tool names
            for tool_variant in [tool_name.title(), tool_name.capitalize(), tool_name.upper()]:
                tool_path = self.results_path / tool_variant / 'analyzed_buggy_contracts'
                
                if not tool_path.exists():
                    continue
                
                logger.debug(f"Processing {tool_variant} results from {tool_path}")
                
                # Process each vulnerability type
                for vuln_dir in tool_path.iterdir():
                    if not vuln_dir.is_dir():
                        continue
                    
                    vuln_name = vuln_dir.name
                    results_dir = vuln_dir / 'results'
                    
                    if not results_dir.exists():
                        # Sometimes results are directly in vuln_dir
                        results_dir = vuln_dir
                    
                    # Extract contract names from result files
                    for result_file in results_dir.iterdir():
                        if not result_file.is_file():
                            continue
                        
                        # Extract contract name (e.g., "buggy_1" from various file formats)
                        contract_name = self._extract_contract_name_from_filename(result_file.name)
                        if not contract_name:
                            continue
                        
                        contract_id = f"{vuln_name}/{contract_name}.sol"
                        
                        if contract_id not in contracts_seen:
                            contracts_seen.add(contract_id)
                            
                            # Load source code if available
                            source_code = self._load_contract_source(vuln_name, contract_name)
                            
                            contracts.append({
                                'id': contract_id,
                                'path': None,  # Will be set if source file found
                                'contract_name': contract_name,
                                'category_name': vuln_name,
                                'vuln_type': self._map_category_to_vuln_type(vuln_name),
                                'category_id': self._get_category_id(vuln_name),
                                'is_buggy': True,  # All contracts in SolidiFI are buggy
                                'base_name': contract_name.replace('buggy_', ''),
                                'source_code': source_code,
                                'ast': self._create_simple_ast(source_code),
                                'bugs': [vuln_name]  # This contract has this vulnerability
                            })
                
                # Break after finding the first valid tool directory
                if contracts:
                    break
        
        logger.info(f"Found {len(contracts)} unique contracts")
        
        # Cache the results
        try:
            with open(cache_file, 'w') as f:
                json.dump(contracts, f, indent=2)
            logger.info(f"Cached contracts to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache contracts: {e}")
        
        return contracts
    
    def _extract_contract_name_from_filename(self, filename: str) -> Optional[str]:
        """Extract contract name from result filename."""
        # Try different patterns
        patterns = [
            r'(buggy_\d+)\.sol',      # buggy_1.sol
            r'(buggy_\d+)\.txt',      # buggy_1.txt
            r'(buggy_\d+)\.json',     # buggy_1.json
            r'(buggy_\d+)\.sol\..*',  # buggy_1.sol.json
            r'(buggy_\d+)\..*',       # buggy_1.anything
            r'(buggy_\d+)',           # just buggy_1
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1)
        
        return None
    
    def _load_contract_source(self, vuln_name: str, contract_name: str) -> str:
        """Load source code for a contract."""
        # Try multiple possible locations
        possible_paths = [
            self.buggy_contracts_path / vuln_name / f"{contract_name}.sol",
            self.solidifi_path / 'contracts' / vuln_name / f"{contract_name}.sol",
            self.solidifi_path / vuln_name / f"{contract_name}.sol",
            self.solidifi_path / 'buggy_contracts' / f"{contract_name}.sol",
            # Also try in results directory (some tools copy source)
            self.results_path / 'Oyente' / 'analyzed_buggy_contracts' / vuln_name / f"{contract_name}.sol",
        ]
        
        for path in possible_paths:
            if path.exists():
                try:
                    return self._parse_solidity_file(path)
                except Exception as e:
                    logger.warning(f"Error reading {path}: {e}")
                    continue
        
        # Return minimal source if not found
        logger.debug(f"Source not found for {contract_name}, creating minimal source")
        return self._create_minimal_source(contract_name)
    
    
    def _get_category_id(self, vuln_name: str) -> int:
        """Get numeric category ID for vulnerability type."""
        if not hasattr(self, '_category_mapping'):
            self._category_mapping = {name: i for i, name in enumerate(sorted(set(self.vuln_dirs)))}
        return self._category_mapping.get(vuln_name, 0)
    
    def _load_tool_results_from_files(self) -> Dict:
        """Load tool analysis results from the SolidiFI-benchmark results directory."""
        tool_results = {}
        
        # Check cache first
        cache_file = self.cache_dir / 'tool_results_cache.json'
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_results = json.load(f)
                logger.info(f"Loaded tool results from cache")
                return cached_results
            except Exception as e:
                logger.warning(f"Failed to load tool results cache: {e}")
        
        for tool_name in self.tool_names:
            # Try different capitalizations
            for tool_variant in [tool_name.title(), tool_name.capitalize(), tool_name.upper()]:
                tool_path = self.results_path / tool_variant / 'analyzed_buggy_contracts'
                
                if not tool_path.exists():
                    continue
                
                logger.info(f"Loading {tool_name} results from {tool_path}")
                
                for vuln_dir in tool_path.iterdir():
                    if not vuln_dir.is_dir():
                        continue
                    
                    vuln_name = vuln_dir.name
                    results_dir = vuln_dir / 'results'
                    
                    if not results_dir.exists():
                        results_dir = vuln_dir  # Sometimes results are in the main directory
                    
                    for result_file in results_dir.iterdir():
                        if not result_file.is_file():
                            continue
                        
                        contract_name = self._extract_contract_name_from_filename(result_file.name)
                        if not contract_name:
                            continue
                        
                        contract_id = f"{vuln_name}/{contract_name}.sol"
                        
                        # Initialize contract entry if needed
                        if contract_id not in tool_results:
                            tool_results[contract_id] = {}
                        
                        # Parse tool output
                        try:
                            tool_output = self._parse_tool_output(result_file, tool_name)
                            tool_results[contract_id][tool_name] = tool_output
                        except Exception as e:
                            logger.error(f"Failed to parse {tool_name} output for {result_file}: {e}")
                            tool_results[contract_id][tool_name] = {'issues': [], 'raw': None}
                
                # Break after finding the first valid tool directory
                break
        
        # Cache results
        try:
            with open(cache_file, 'w') as f:
                json.dump(tool_results, f, indent=2)
            logger.info(f"Cached tool results")
        except Exception as e:
            logger.warning(f"Failed to cache tool results: {e}")
        
        logger.info(f"Loaded tool results for {len(tool_results)} contracts")
        return tool_results
    
    def _parse_tool_output(self, file_path: Path, tool_name: str) -> Dict:
        """Enhanced tool output parser with better line number extraction."""
        try:
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return self._parse_json_output(data, tool_name)
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                return self._parse_txt_output(content, tool_name)
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return {'issues': [], 'raw': None, 'error': str(e)}

    def _parse_json_output(self, data: Dict, tool_name: str) -> Dict:
        """Enhanced JSON output parser with better line number extraction."""
        issues = []
        
        if tool_name == 'slither':
            if isinstance(data, dict) and 'results' in data:
                results = data['results']
                if isinstance(results, dict) and 'detectors' in results:
                    for detector in results['detectors']:
                        issue = {
                            'type': detector.get('check', 'Unknown'),
                            'severity': detector.get('impact', 'Medium'),
                            'confidence': detector.get('confidence', 'Medium'),
                            'description': detector.get('description', ''),
                            'line': self._extract_line_from_slither(detector),
                            'vuln_category': self._map_issue_to_category(detector.get('check', ''))
                        }
                        issues.append(issue)
        
        elif tool_name == 'oyente':
            if 'vulnerabilities' in data:
                vulns = data['vulnerabilities']
                for vuln_type, locations in vulns.items():
                    if locations:
                        vuln_category = self._map_issue_to_category(vuln_type)
                        if isinstance(locations, list):
                            for location in locations:
                                line_num = self._extract_line_from_location(location)
                                issues.append({
                                    'type': vuln_type,
                                    'severity': 'High',
                                    'location': location,
                                    'line': line_num,
                                    'vuln_category': vuln_category
                                })
                        else:
                            issues.append({
                                'type': vuln_type,
                                'severity': 'High',
                                'line': -1,
                                'vuln_category': vuln_category
                            })
        
        elif tool_name == 'mythril':
            # Mythril JSON format
            if 'issues' in data:
                for issue in data['issues']:
                    line_num = self._extract_line_from_mythril(issue)
                    issues.append({
                        'type': issue.get('title', 'Unknown'),
                        'severity': issue.get('severity', 'Medium'),
                        'swc_id': issue.get('swc-id', ''),
                        'line': line_num,
                        'description': issue.get('description', ''),
                        'vuln_category': self._map_swc_to_category(issue.get('swc-id', ''))
                    })
        
        return {'issues': issues, 'raw': data}

    def _extract_line_from_slither(self, detector: Dict) -> int:
        """Extract line number from Slither detector output."""
        # Check elements field for source mappings
        if 'elements' in detector:
            for element in detector['elements']:
                if 'source_mapping' in element:
                    sm = element['source_mapping']
                    if 'lines' in sm and sm['lines']:
                        return sm['lines'][0] if isinstance(sm['lines'], list) else sm['lines']
                    if 'starting_column' in sm and 'ending_column' in sm:
                        # Try to extract from source mapping
                        if 'start' in sm:
                            return self._calculate_line_from_position(sm['start'])
        
        # Fallback: check description for line numbers
        description = detector.get('description', '')
        line_match = re.search(r'line[:\s]*(\d+)', description, re.IGNORECASE)
        if line_match:
            return int(line_match.group(1))
        
        return -1

    def _extract_line_from_mythril(self, issue: Dict) -> int:
        """Extract line number from Mythril issue."""
        # Check locations field
        if 'locations' in issue:
            for location in issue['locations']:
                if 'sourceMap' in location:
                    source_map = location['sourceMap']
                    if ':' in source_map:
                        try:
                            parts = source_map.split(':')
                            if len(parts) >= 3:
                                # Solidity source map format: start:length:file
                                start_pos = int(parts[0])
                                return self._calculate_line_from_position(start_pos)
                        except:
                            pass
        
        # Check source map in issue directly
        if 'sourceMap' in issue:
            try:
                parts = issue['sourceMap'].split(':')
                if len(parts) >= 3:
                    start_pos = int(parts[0])
                    return self._calculate_line_from_position(start_pos)
            except:
                pass
        
        # Fallback: parse from description
        description = issue.get('description', '')
        line_match = re.search(r'line[:\s]*(\d+)', description, re.IGNORECASE)
        if line_match:
            return int(line_match.group(1))
        
        return -1

    def _extract_line_from_location(self, location) -> int:
        """Extract line number from location string or object."""
        if isinstance(location, dict):
            if 'line' in location:
                return int(location['line'])
            if 'sourceMap' in location:
                return self._parse_source_map_line(location['sourceMap'])
        elif isinstance(location, str):
            # Try to extract line number from string
            line_match = re.search(r'line[:\s]*(\d+)', location, re.IGNORECASE)
            if line_match:
                return int(line_match.group(1))
            
            # Try to parse as source map (start:length:file format)
            if ':' in location:
                try:
                    parts = location.split(':')
                    if len(parts) >= 3 and parts[0].isdigit():
                        return self._calculate_line_from_position(int(parts[0]))
                except:
                    pass
        
        return -1

    def _calculate_line_from_position(self, position: int, source_code: str = None) -> int:
        """Calculate line number from character position in source code."""
        if source_code:
            # Count newlines up to position
            return source_code[:position].count('\n') + 1
        else:
            # Rough estimation if no source code available
            # Assume average 50 characters per line
            return max(1, position // 50)

    def _parse_txt_output(self, content: str, tool_name: str) -> Dict:
        """Enhanced text output parser with better line number extraction."""
        issues = []
        
        if tool_name == 'mythril':
            issues = self._parse_mythril_txt(content)
        elif tool_name == 'securify':
            issues = self._parse_securify_txt(content)
        elif tool_name == 'smartcheck':
            issues = self._parse_smartcheck_txt(content)
        elif tool_name == 'manticore':
            issues = self._parse_manticore_txt(content)
        
        return {'issues': issues, 'raw': content}

    def _parse_mythril_txt(self, content: str) -> List[Dict]:
        """Enhanced Mythril text parser with line number extraction."""
        issues = []
        
        # Split content into issue blocks
        issue_blocks = re.split(r'====.*?====', content)
        
        for i, block in enumerate(issue_blocks[1:]):  # Skip first empty block
            if not block.strip():
                continue
                
            # Extract title from delimiter
            title_match = re.search(r'====(.*?)====', content)
            title = title_match.group(1).strip() if title_match else f"Issue {i+1}"
            
            # Extract SWC ID
            swc_match = re.search(r'SWC ID:\s*(\d+)', block, re.IGNORECASE)
            swc_id = swc_match.group(1) if swc_match else ''
            
            # Extract severity
            severity_match = re.search(r'Severity:\s*(\w+)', block, re.IGNORECASE)
            severity = severity_match.group(1) if severity_match else 'Medium'
            
            # Extract line number - multiple strategies
            line_num = -1
            
            # Strategy 1: Look for explicit line mentions
            line_patterns = [
                r'line[:\s]*(\d+)',
                r'at line[:\s]*(\d+)',
                r'on line[:\s]*(\d+)',
                r'Line[:\s]*(\d+)',
                r'ln[:\s]*(\d+)'
            ]
            
            for pattern in line_patterns:
                line_match = re.search(pattern, block, re.IGNORECASE)
                if line_match:
                    line_num = int(line_match.group(1))
                    break
            
            # Strategy 2: Parse source mapping if present
            if line_num == -1:
                source_map_match = re.search(r'(\d+):(\d+):(\d+)', block)
                if source_map_match:
                    start_pos = int(source_map_match.group(1))
                    line_num = self._calculate_line_from_position(start_pos)
            
            issues.append({
                'type': title,
                'severity': severity,
                'swc_id': swc_id,
                'line': line_num,
                'description': block.strip(),
                
                'vuln_category': self._map_swc_to_category(swc_id)
            })
        
        return issues

    def _parse_securify_txt(self, content: str) -> List[Dict]:
        """Enhanced Securify text parser."""
        issues = []
        
        # Look for violation patterns with line numbers
        violation_patterns = [
            r'Violation for (.*?):\s*.*?line[:\s]*(\d+)',
            r'(DAO|TOD|Reentrancy|UnhandledException).*?line[:\s]*(\d+)',
            r'Pattern:\s*(.*?)\s+at line[:\s]*(\d+)'
        ]
        
        for pattern in violation_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                vuln_type = match.group(1).strip()
                line_num = int(match.group(2)) if len(match.groups()) > 1 else -1
                
                issues.append({
                    'type': vuln_type,
                    'severity': 'High',
                    'line': line_num,
                    'vuln_category': self._map_issue_to_category(vuln_type)
                })
        
        # Fallback: look for violations without line numbers
        simple_violations = re.findall(r'Violation for (.*?):', content)
        for vuln_type in simple_violations:
            if not any(issue['type'] == vuln_type for issue in issues):
                issues.append({
                    'type': vuln_type.strip(),
                    'severity': 'High',
                    'line': -1,
                    'vuln_category': self._map_issue_to_category(vuln_type)
                })
        
        return issues
    
    
    def _parse_smartcheck_txt(self, content: str) -> List[Dict]:
        """Parse SmartCheck text output."""
        issues = []
        
        # SmartCheck rule pattern
        rule_pattern = r'ruleId: "(.*?)"\s*.*?severity: "(.*?)"'
        matches = re.finditer(rule_pattern, content, re.IGNORECASE)
        
        for match in matches:
            rule_id, severity = match.groups()
            issues.append({
                'type': rule_id,
                'severity': severity,
                'vuln_category': self._map_issue_to_category(rule_id)
            })
        
        return issues
    
    def _parse_manticore_txt(self, content: str) -> List[Dict]:
        """Parse Manticore text output."""
        issues = []
        
        # Look for different types of issues in Manticore output
        if 'OVERFLOW' in content or 'UNDERFLOW' in content:
            issues.append({
                'type': 'Integer Overflow/Underflow',
                'severity': 'High',
                'vuln_category': 'integer_overflow'
            })
        
        if 'REVERT' in content or 'THROW' in content or 'ASSERT' in content:
            issues.append({
                'type': 'Unhandled Exception',
                'severity': 'Medium',
                'vuln_category': 'unhandled_exception'
            })
        
        return issues


    def extract_ground_truth_vulnerabilities(self) -> Dict:
        """Extract ground truth vulnerabilities from SolidiFI structure"""
        ground_truth = {}
        
        # Map SolidiFI directory structure to vulnerability types
        vuln_mapping = {
            'Re-entrancy': 'reentrancy',
            'Timestamp-Dependency': 'timestamp_dep', 
            'Unchecked-Send': 'unchecked_send',
            'Unhandled-Exceptions': 'unhandled_exp',
            'TOD': 'tod',
            'Overflow-Underflow': 'integer_flow',
            'tx.origin': 'tx_origin'
        }
        
        for contract in self.contracts:
            contract_id = contract['id']
            category = contract['category_name']
            
            # Each contract has exactly one injected vulnerability
            if category in vuln_mapping:
                ground_truth[contract_id] = {
                    'injected_vulnerability': vuln_mapping[category],
                    'line_number': self._extract_injection_line(contract),
                    'source_code': contract.get('source_code', '')
                }
        
        return ground_truth

    def analyze_line_number_accuracy(self, tool_results: Dict, ground_truth: Dict) -> Dict:
        """Analyze line number accuracy for detected vulnerabilities"""
        line_accuracy = {}
            
        for tool_name in self.tool_names:
            line_accuracy[tool_name] = {}
                
            for contract_id in ground_truth:
                if contract_id in tool_results and tool_name in tool_results[contract_id]:
                    gt_line = ground_truth[contract_id]['line_number']
                    detected_lines = self._extract_detected_lines(
                        tool_results[contract_id][tool_name]
                    )
                        
                    # Check if any detected line matches ground truth (±2 line tolerance)
                    line_match = any(abs(line - gt_line) <= 2 for line in detected_lines)
                        
                    line_accuracy[tool_name][contract_id] = {
                        'ground_truth_line': gt_line,
                        'detected_lines': detected_lines,
                        'line_accurate': line_match,
                        'has_detection': len(detected_lines) > 0
                    }
            
        return line_accuracy

    def _extract_detected_lines(self, tool_result: Dict) -> List[int]:
        """Enhanced line number extraction from tool outputs"""
        detected_lines = []
        
        for issue in tool_result.get('issues', []):
            line_num = self._extract_line_from_issue(issue)
            if line_num > 0:
                detected_lines.append(line_num)
        
        return detected_lines

    def _extract_injection_line(self, contract):
        """
        Extract the line number of the injected vulnerability from the ground truth data.
        """
        gt_path = Path("source_code_path")  # adjust based on your dataset dict structure
        if not gt_path.exists():
            return None
        
        with open(gt_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Adjust this parsing logic to match your GT format
                if 'line_number' in line.lower():
                    try:
                        return int(line.strip().split(':')[-1])
                    except ValueError:
                        continue
        return None

    def _extract_line_from_issue(self, issue: Dict) -> int:
        """Extract line number with improved parsing"""
        # Multiple extraction strategies for different tool formats
        
        # Strategy 1: Direct line field
        if 'line' in issue and issue['line'] > 0:
            return issue['line']
        
        # Strategy 2: Source mapping parsing  
        if 'src' in issue or 'sourceMap' in issue:
            return self._parse_source_mapping(issue)
        
        # Strategy 3: Text-based extraction from description
        if 'description' in issue or 'message' in issue:
            return self._extract_line_from_text(issue)
            
        return -1  # No line number found



    
    def _map_swc_to_category(self, swc_id: str) -> str:
        """Map SWC ID to vulnerability category."""
        swc_mapping = {
            '107': 'reentrancy', 'SWC-107': 'reentrancy',
            '101': 'integer_overflow', 'SWC-101': 'integer_overflow',
            '104': 'unchecked_return', 'SWC-104': 'unchecked_return',
            '116': 'timestamp_dependency', 'SWC-116': 'timestamp_dependency',
            '115': 'tx_origin', 'SWC-115': 'tx_origin',
            '110': 'unhandled_exception', 'SWC-110': 'unhandled_exception'
        }
        return swc_mapping.get(swc_id, 'unknown')
    
    def _map_issue_to_category(self, issue_type: str) -> str:
        """Map issue type to standardized vulnerability category."""
        issue_lower = issue_type.lower()
        
        if any(term in issue_lower for term in ['reentrancy', 'dao', 're-entrancy']):
            return 'reentrancy'
        elif any(term in issue_lower for term in ['overflow', 'underflow', 'integer']):
            return 'integer_overflow'
        elif any(term in issue_lower for term in ['unchecked', 'return', 'send']):
            return 'unchecked_return'
        elif any(term in issue_lower for term in ['timestamp', 'block.timestamp', 'now']):
            return 'timestamp_dependency'
        elif any(term in issue_lower for term in ['tx.origin', 'tx origin']):
            return 'tx_origin'
        elif any(term in issue_lower for term in ['exception', 'revert', 'throw']):
            return 'unhandled_exception'
        elif any(term in issue_lower for term in ['tod', 'transaction order']):
            return 'tod'
        else:
            return 'unknown'
    
    def _create_injected_bugs_mapping(self) -> Dict:
        """Create mapping of injected bugs based on contract categorization."""
        injected_bugs = {}
        
        for contract in self._contracts:
            contract_id = contract['id']
            vuln_type = contract['category_name']
            
            # Create synthetic bug entry for this vulnerability type
            bugs = [{
                'type': vuln_type,
                'line': 1,  # Default line number
                'detected_by': {}  # Will be filled based on tool analysis
            }]
            
            injected_bugs[contract_id] = bugs
        
        return injected_bugs
    
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate actual empirical performance metrics from SolidiFI tool results."""
        performance_metrics = {tool: {} for tool in self.tool_names}
        
        # Check cache
        cache_file = self.cache_dir / 'empirical_performance_cache.json'
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_metrics = json.load(f)
                logger.info("Loaded empirical performance metrics from cache")
                return cached_metrics
            except Exception as e:
                logger.warning(f"Failed to load performance metrics cache: {e}")
        
        # Initialize counters for each tool
        for tool in self.tool_names:
            performance_metrics[tool] = {
                'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0,
                'total_contracts': 0,
                'vulnerability_breakdown': {}
            }
        
        # Process each contract to build confusion matrix
        for contract in self._contracts:
            contract_id = contract['id']
            
            # Extract actual vulnerability type from SolidiFI structure
            actual_vuln_type = self._get_contract_ground_truth_vulnerability(contract)
            
            if not actual_vuln_type:
                continue
                
            for tool in self.tool_names:
                performance_metrics[tool]['total_contracts'] += 1
                
                # Get tool's detection results for this contract
                detected_vulns = self._get_tool_detections_for_contract(contract_id, tool)
                
                # Check if tool detected the actual vulnerability
                tool_detected_actual = actual_vuln_type in detected_vulns
                
                # Calculate confusion matrix elements
                if tool_detected_actual:
                    performance_metrics[tool]['TP'] += 1
                else:
                    performance_metrics[tool]['FN'] += 1
                    
                # Count false positives (detected vulns that aren't the actual one)
                false_positives = len(detected_vulns - {actual_vuln_type}) if detected_vulns else 0
                performance_metrics[tool]['FP'] += false_positives
                
                # True negatives: other vulnerability types not detected
                # (This is more complex in multi-class scenario, simplified here)
                other_vulns = set(['reentrancy', 'integer_overflow', 'unchecked_return', 
                                'timestamp_dependency', 'tx_origin', 'unhandled_exception', 'tod']) - {actual_vuln_type}
                tn_count = len(other_vulns - detected_vulns)
                performance_metrics[tool]['TN'] += tn_count
        
        # Calculate final performance metrics for each tool
        for tool in self.tool_names:
            TP = performance_metrics[tool]['TP']
            FP = performance_metrics[tool]['FP']
            FN = performance_metrics[tool]['FN']
            TN = performance_metrics[tool]['TN']
            
            epsilon = 1e-6
            
            # Calculate actual empirical metrics
            tpr = TP / (TP + FN + epsilon)
            fpr = FP / (FP + TN + epsilon)
            accuracy = (TP + TN) / (TP + FP + TN + FN + epsilon)
            precision = TP / (TP + FP + epsilon)
            recall = tpr  # Same as TPR
            
            performance_metrics[tool].update({
                'tpr': float(tpr),
                'fpr': float(fpr),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall)
            })
            
            logger.info(f"Tool {tool} empirical metrics: TPR={tpr:.3f}, FPR={fpr:.3f}, "
                    f"Acc={accuracy:.3f}, Prec={precision:.3f}, Rec={recall:.3f}")
        
        # Cache results
        try:
            with open(cache_file, 'w') as f:
                json.dump(performance_metrics, f, indent=2)
            logger.info("Cached empirical performance metrics")
        except Exception as e:
            logger.warning(f"Failed to cache performance metrics: {e}")
        
        return performance_metrics
    
    
    def _get_injected_bugs_for_contract(self, contract: Dict) -> set:
        """
        Step 1: Count injected bugs (from SolidiFI)
        
        In SolidiFI, each contract in a vulnerability category has that specific 
        vulnerability type injected. This follows the SolidiFI methodology.
        """
        injected_bugs = set()
        
        # Primary method: Use contract's vulnerability category
        # In SolidiFI, each contract in "Re-entrancy" folder has reentrancy injected
        vulnerability_category = contract.get('category_name', '')
        if vulnerability_category:
            normalized_vuln = self._normalize_vulnerability_type(vulnerability_category)
            if normalized_vuln:
                injected_bugs.add(normalized_vuln)
        
        # Secondary method: Check if contract has explicit bug information
        contract_id = contract['id']
        if contract_id in self._injected_bugs:
            for bug in self._injected_bugs[contract_id]:
                if isinstance(bug, dict) and 'type' in bug:
                    bug_type = self._normalize_vulnerability_type(bug['type'])
                    if bug_type:
                        injected_bugs.add(bug_type)
        
        # Fallback: Use vuln_type field if available
        if not injected_bugs and 'vuln_type' in contract:
            vuln_type = self._normalize_vulnerability_type(contract['vuln_type'])
            if vuln_type:
                injected_bugs.add(vuln_type) 
                
        # print("Injected bug stats:")
        # for contract_id in injected_bugs:
        #     print(f"{contract_id}: {injected_bugs[contract_id]}") 
        
        if injected_bugs:
            print(f"Contract {contract_id}: injected bugs = {list(injected_bugs)}")
        else:
            print(f"Contract {contract_id}: no injected bugs found")
        
        return injected_bugs

    def _get_detected_bugs_for_contract(self, contract_id: str, tool: str) -> set:
        """
        Step 2: Count detected bugs (from tool results)
        
        Parse the tool's analysis results for this specific contract
        and identify what vulnerability types were detected.
        """
        detected_bugs = set()
        
        # Check if we have tool results for this contract
        if contract_id not in self._tool_results:
            return detected_bugs
        
        if tool not in self._tool_results[contract_id]:
            return detected_bugs
        
        tool_result = self._tool_results[contract_id][tool]
        issues = tool_result.get('issues', [])
        
        # Parse each detected issue
        for issue in issues:
            if isinstance(issue, dict):
                issue_type = issue.get('type', '')
                vuln_category = issue.get('vuln_category', '')
            else:
                issue_type = str(issue)
                vuln_category = ''
            
            # Map detected issue to standard vulnerability type
            mapped_vuln = self._map_detected_issue_to_standard_type(issue_type, vuln_category)
            if mapped_vuln:
                detected_bugs.add(mapped_vuln)
        
        return detected_bugs

    def _calculate_contract_tool_confusion_matrix(self, injected_bugs: set, detected_bugs: set, contract: Dict) -> Tuple[int, int, int, int]:
        """
        Step 3: Calculate TP, FP, FN, TN for this specific contract-tool pair
        
        Following the exact definitions:
        - True Positives (TP): Correctly detected injected bugs
        - False Negatives (FN): Missed injected bugs  
        - False Positives (FP): Reported bugs not in injected set
        - True Negatives (TN): Correctly not reporting non-bugs
        """
        
        # True Positives: Correctly detected injected bugs
        tp = len(injected_bugs.intersection(detected_bugs))
        
        # False Negatives: Missed injected bugs
        fn = len(injected_bugs - detected_bugs)
        
        # False Positives: Reported bugs not in injected set
        fp = len(detected_bugs - injected_bugs)
        
        # True Negatives: Correctly not reporting non-bugs
        # For SolidiFI, this is vulnerability types that were neither injected nor detected
        all_possible_vulnerabilities = {
            'reentrancy', 'integer_overflow', 'unchecked_return',
            'timestamp_dependency', 'tx_origin', 'unhandled_exception', 'tod'
        }
        
        # Vulnerabilities that were not injected in this contract
        not_injected = all_possible_vulnerabilities - injected_bugs
        # Vulnerabilities that were not detected by the tool
        not_detected = all_possible_vulnerabilities - detected_bugs
        # True negatives: vulnerabilities that were correctly not flagged
        tn = len(not_injected.intersection(not_detected))
        
        return tp, fp, fn, tn 
    
    
    def _get_contract_ground_truth_vulnerability(self, contract: Dict) -> Optional[str]:
        """Extract the actual vulnerability type from contract metadata."""
        # Method 1: From contract category (SolidiFI structure)
        if 'category_name' in contract:
            return self._normalize_vulnerability_type(contract['category_name'])
        
        # Method 2: From contract ID path
        contract_id = contract.get('id', '')
        if '/' in contract_id:
            vuln_category = contract_id.split('/')[0]
            return self._normalize_vulnerability_type(vuln_category)
        
        return None

    def _get_tool_detections_for_contract(self, contract_id: str, tool: str) -> set:
        """Get vulnerability types detected by a tool for a specific contract."""
        detected_vulns = set()
        
        if contract_id not in self._tool_results:
            return detected_vulns
        
        if tool not in self._tool_results[contract_id]:
            return detected_vulns
        
        tool_result = self._tool_results[contract_id][tool]
        issues = tool_result.get('issues', [])
        
        for issue in issues:
            if isinstance(issue, dict):
                issue_type = issue.get('type', '')
                vuln_category = issue.get('vuln_category', '')
            else:
                issue_type = str(issue)
                vuln_category = ''
            
            # Map detected issue to standard vulnerability type
            mapped_vuln = self._map_detected_issue_to_standard_type(issue_type, vuln_category)
            if mapped_vuln:
                detected_vulns.add(mapped_vuln)
        
        return detected_vulns
    

    def _normalize_vulnerability_type(self, vuln_type: str) -> Optional[str]:
        """Normalize vulnerability type to standard format for SolidiFI evaluation."""
        if not vuln_type:
            return None
        
        vuln_lower = vuln_type.lower().strip()
        
        # SolidiFI-specific mappings (directory names -> standard names)
        solidifi_mappings = {
            # Direct SolidiFI directory names
            're-entrancy': 'reentrancy',
            'timestamp-dependency': 'timestamp_dependency',
            'unhandled-exceptions': 'unhandled_exception', 
            'unchecked-send': 'unchecked_return',
            'overflow-underflow': 'integer_overflow',
            'tx.origin': 'tx_origin',
            'tod': 'tod',
            
            # Alternative formats
            'reentrancy': 'reentrancy',
            'timestamp_dependency': 'timestamp_dependency',
            'unhandled_exception': 'unhandled_exception',
            'unchecked_return': 'unchecked_return',
            'integer_overflow': 'integer_overflow',
            'tx_origin': 'tx_origin',
            
            # Legacy formats
            'integer-overflow-underflow': 'integer_overflow',
            'authorization-through-tx-origin': 'tx_origin',
            'unchecked_return_value': 'unchecked_return',
        }
        
        # Direct mapping
        if vuln_lower in solidifi_mappings:
            return solidifi_mappings[vuln_lower]
        
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
        elif 'exception' in vuln_lower:
            return 'unhandled_exception'
        elif 'tod' in vuln_lower:
            return 'tod'
        
        logger.debug(f"Could not normalize vulnerability type: {vuln_type}")
        return None
    
    def _map_detected_issue_to_standard_type(self, issue_type: str, vuln_category: str = '') -> Optional[str]:
        """Map tool-detected issue to standard vulnerability type."""
        if not issue_type:
            return None
        
        issue_lower = issue_type.lower().strip()
        
        # Tool-specific detection patterns
        detection_mappings = {
            # Reentrancy detection patterns
            'reentrancy': 'reentrancy',
            're-entrancy': 'reentrancy',
            'dao': 'reentrancy', 
            'callstack': 'reentrancy',
            'money_concurrency': 'reentrancy',
            'reentrancy-eth': 'reentrancy',
            'reentrancy-no-eth': 'reentrancy',
            
            # Integer overflow detection patterns  
            'integer overflow': 'integer_overflow',
            'integer underflow': 'integer_overflow',
            'overflow': 'integer_overflow',
            'underflow': 'integer_overflow',
            
            # Unchecked send detection patterns
            'unchecked send': 'unchecked_return',
            'unchecked call': 'unchecked_return', 
            'unchecked return': 'unchecked_return',
            'unchecked-send': 'unchecked_return',
            'unchecked-lowlevel': 'unchecked_return',
            'external call to user-supplied address': 'unchecked_return',
            
            # Timestamp dependency detection patterns
            'timestamp dependency': 'timestamp_dependency',
            'timestamp': 'timestamp_dependency',
            'time_dependency': 'timestamp_dependency',
            'block timestamp': 'timestamp_dependency',
            'block-timestamp': 'timestamp_dependency',
            'dependence on predictable environment variable': 'timestamp_dependency',
            
            # tx.origin detection patterns
            'tx.origin': 'tx_origin',
            'tx origin': 'tx_origin', 
            'use of tx.origin': 'tx_origin',
            'tx-origin': 'tx_origin',
            'authorization through tx.origin': 'tx_origin',
            
            # Unhandled exception detection patterns
            'unhandled exception': 'unhandled_exception',
            'unhandled exceptions': 'unhandled_exception',
            'exception disorder': 'unhandled_exception',
            'assertion_failure': 'unhandled_exception',
            'assertion failure': 'unhandled_exception',
            
            # TOD detection patterns
            'tod': 'tod',
            'transaction order': 'tod',
            'todreceiver': 'tod',
            'todtransfer': 'tod',
            'transaction order dependency': 'tod',
        }
        
        # Direct mapping
        if issue_lower in detection_mappings:
            return detection_mappings[issue_lower]
        
        # Substring matching
        for pattern, vuln_type in detection_mappings.items():
            if pattern in issue_lower:
                return vuln_type
        
        # Use vulnerability category if provided
        if vuln_category:
            normalized_category = self._normalize_vulnerability_type(vuln_category)
            if normalized_category:
                return normalized_category
        
        return None


    # Add this method to improve the _create_injected_bugs_mapping method (around line 360)
    def _create_injected_bugs_mapping(self) -> Dict:
        """Create mapping of injected bugs based on SolidiFI structure and contract categorization."""
        injected_bugs = {}
        
        for contract in self._contracts:
            contract_id = contract['id']
            vuln_type = contract['category_name']
            
            # Create bug entry based on contract's vulnerability category
            # Each contract in SolidiFI has one primary vulnerability type injected
            normalized_vuln = self._normalize_vulnerability_type(vuln_type)
            
            bugs = [{
                'type': normalized_vuln if normalized_vuln else vuln_type,
                'original_type': vuln_type,
                'line': 1,  # Default line number (SolidiFI doesn't always provide exact lines)
                'category': contract.get('category_name', 'unknown'),
                'detected_by': {}  # Will be filled based on tool analysis
            }]
            
            injected_bugs[contract_id] = bugs
            
            logger.debug(f"Contract {contract_id}: Injected {normalized_vuln} (original: {vuln_type})")
        
        return injected_bugs
    
    
    def _map_detected_issue_to_bug_type(self, issue_type: str, contract_category: str) -> Optional[str]:
        """Map detected issue type to SolidiFI bug type."""
        issue_lower = issue_type.lower()
        
        # Direct mapping based on issue type
        if 'reentrancy' in issue_lower or 'dao' in issue_lower:
            return 'Re-entrancy'
        elif 'overflow' in issue_lower or 'underflow' in issue_lower:
            return 'Integer-Overflow-Underflow'
        elif 'unchecked' in issue_lower and ('send' in issue_lower or 'call' in issue_lower):
            return 'Unchecked-Send'
        elif 'timestamp' in issue_lower or 'now' in issue_lower:
            return 'Timestamp-Dependency'
        elif 'tx.origin' in issue_lower:
            return 'tx-origin'
        elif 'exception' in issue_lower or 'revert' in issue_lower:
            return 'Unhandled-Exceptions'
        elif 'tod' in issue_lower or 'transaction order' in issue_lower:
            return 'TOD'
        
        # If no direct mapping, try to infer from contract category
        if contract_category in SOLIDIFI_BUG_TYPES:
            return contract_category
        
        return None
    
    def _map_category_to_vuln_type(self, category_name: str) -> str:
        """Map SolidiFI category name to standardized vulnerability type."""
        category_lower = category_name.lower()
        
        direct_mappings = {
            're-entrancy': 'reentrancy',
            'overflow-underflow': 'integer_overflow',
            'integer-overflow-underflow': 'integer_overflow',
            'unchecked-send': 'unchecked_return',
            'timestamp-dependency': 'timestamp_dependency',
            'tx.origin': 'tx_origin',
            'tx-origin': 'tx_origin',
            'unhandled-exceptions': 'unhandled_exception',
            'tod': 'tod'
        }
        
        if category_lower in direct_mappings:
            return direct_mappings[category_lower]
        
        # Pattern-based mappings
        for vuln_type, patterns in self.vuln_type_mapping.items():
            for pattern in patterns:
                if pattern.lower() in category_lower:
                    return vuln_type
        
        return category_lower.replace('_', '-')
    
    def _parse_solidity_file(self, file_path: Path) -> str:
        """Parse Solidity file and extract source code."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                return ""
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return ""
    
    def _create_simple_ast(self, source_code: str) -> Dict:
        """Create a simple AST-like structure from source code."""
        lines = source_code.split('\n')
        functions = []
        statements = []
        expressions = []
        variables = []
        
        # Enhanced patterns for better node detection
        function_patterns = [
            r'function\s+(\w+)\s*\(',
            r'constructor\s*\(',
            r'modifier\s+(\w+)\s*\('
        ]
        
        statement_patterns = [
            r'require\s*\(',
            r'assert\s*\(',
            r'if\s*\(',
            r'for\s*\(',
            r'while\s*\(',
            r'return\s+',
            r'emit\s+',
            r'revert\s*\('
        ]
        
        expression_patterns = [
            r'\.call\s*\(',
            r'\.transfer\s*\(',
            r'\.send\s*\(',
            r'msg\.sender',
            r'tx\.origin',
            r'block\.timestamp',
            r'now',
            r'\+\+|\-\-',
            r'[=!<>]=?',
            r'&&|\|\|'
        ]
        
        variable_patterns = [
            r'(address|uint|int|bool|string|bytes)\s+(\w+)',
            r'(mapping)\s*\(',
            r'(struct)\s+(\w+)',
            r'(event)\s+(\w+)'
        ]
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('//'):
                continue
                
            # Check for functions
            for pattern in function_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    match = re.search(pattern, line, re.IGNORECASE)
                    func_name = match.group(1) if match and match.groups() else f"function_{len(functions)}"
                    functions.append({
                        'id': len(functions),
                        'nodeType': 'FunctionDefinition',
                        'name': func_name,
                        'src': f"{i}:{len(line)}:{i}",
                        'children': [],
                        'visibility': self._extract_visibility(line),
                        'stateMutability': self._extract_mutability(line)
                    })
                    break
            
            # Check for statements
            for pattern in statement_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    statements.append({
                        'id': len(statements) + 1000,
                        'nodeType': 'Statement',
                        'name': line[:100],
                        'src': f"{i}:{len(line)}:{i}",
                        'children': [],
                        'statementType': self._classify_statement(line)
                    })
                    break
            
            # Check for expressions
            for pattern in expression_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    expressions.append({
                        'id': len(expressions) + 2000,
                        'nodeType': 'Expression',
                        'name': line[:100],
                        'src': f"{i}:{len(line)}:{i}",
                        'children': [],
                        'expressionType': self._classify_expression(line)
                    })
                    break
            
            # Check for variables
            for pattern in variable_patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    if match.groups():
                        if len(match.groups()) > 1:
                            var_name = match.group(2)
                            type_name = match.group(1)
                        else:
                            var_name = match.group(1)
                            type_name = 'unknown'
                    else:
                        var_name = f"var_{len(variables)}"
                        type_name = 'unknown'
                        
                    variables.append({
                        'id': len(variables) + 3000,
                        'nodeType': 'VariableDeclaration',
                        'name': var_name,
                        'src': f"{i}:{len(line)}:{i}",
                        'children': [],
                        'typeName': type_name
                    })
        
        # Create hierarchical relationships
        all_nodes = functions + statements + expressions + variables
        
        # Create the complete AST structure
        ast = {
            'id': 0,
            'nodeType': 'SourceUnit',
            'src': f"0:{len(source_code)}:0",
            'children': all_nodes
        }
        
        return ast
    
    def _extract_visibility(self, line: str) -> str:
        """Extract visibility modifier from function declaration."""
        if re.search(r'\bpublic\b', line, re.IGNORECASE):
            return 'public'
        elif re.search(r'\bprivate\b', line, re.IGNORECASE):
            return 'private'
        elif re.search(r'\binternal\b', line, re.IGNORECASE):
            return 'internal'
        elif re.search(r'\bexternal\b', line, re.IGNORECASE):
            return 'external'
        return 'internal'
    
    def _extract_mutability(self, line: str) -> str:
        """Extract state mutability from function declaration."""
        if re.search(r'\bview\b', line, re.IGNORECASE):
            return 'view'
        elif re.search(r'\bpure\b', line, re.IGNORECASE):
            return 'pure'
        elif re.search(r'\bpayable\b', line, re.IGNORECASE):
            return 'payable'
        return 'nonpayable'
    
    def _classify_statement(self, line: str) -> str:
        """Classify statement type."""
        line_lower = line.lower()
        if 'require' in line_lower or 'assert' in line_lower:
            return 'require_statement'
        elif 'if' in line_lower:
            return 'if_statement'
        elif 'for' in line_lower or 'while' in line_lower:
            return 'loop_statement'
        elif 'return' in line_lower:
            return 'return_statement'
        elif 'emit' in line_lower:
            return 'emit_statement'
        return 'expression_statement'
    
    def _classify_expression(self, line: str) -> str:
        """Classify expression type."""
        line_lower = line.lower()
        if 'call' in line_lower or 'transfer' in line_lower or 'send' in line_lower:
            return 'function_call'
        elif 'msg.sender' in line_lower or 'tx.origin' in line_lower:
            return 'member_access'
        elif re.search(r'[=!<>]=?', line):
            return 'binary_operation'
        elif '++' in line or '--' in line:
            return 'unary_operation'
        return 'identifier'
    
    # Properties to access loaded data (avoiding circular dependencies)
    @property
    def contracts(self) -> List[Dict]:
        """Get loaded contracts."""
        return self._contracts if self._contracts is not None else []
    
    @property
    def tool_results(self) -> Dict:
        """Get loaded tool results."""
        return self._tool_results if self._tool_results is not None else {}
    
    @property
    def injected_bugs(self) -> Dict:
        """Get injected bugs mapping."""
        return self._injected_bugs if self._injected_bugs is not None else {}
    
    @property
    def performance_metrics(self) -> Dict:
        """Get performance metrics."""
        return self._performance_metrics if self._performance_metrics is not None else {}
    
    # Public interface methods
    def load_contracts(self) -> List[Dict]:
        """Load all contracts with their metadata."""
        return self.contracts
    
    def load_tool_results(self) -> Dict:
        """Load tool analysis results."""
        return self.tool_results
    
    def load_injected_bugs(self) -> Dict:
        """Load injected bugs mapping."""
        return self.injected_bugs
    
    def load_performance_metrics(self) -> Dict:
        """Load/calculate performance metrics."""
        return self.performance_metrics
    
    def get_dataset_split(self, contracts: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split contracts into train, validation, and test sets."""
        if not contracts:
            raise ValueError("No contracts to split!")
        
        train_split = self.config['data']['train_split']
        val_split = self.config['data']['val_split']
        test_split = self.config['data']['test_split']
        
        # Create category labels for stratified split
        categories = [contract['category_id'] for contract in contracts]
        
        # Try stratified split
        try:
            # First split: separate test set
            train_val_contracts, test_contracts, _, _ = train_test_split(
                contracts, categories,
                test_size=test_split,
                random_state=42,
                stratify=categories
            )
            
            # Second split: separate train and validation
            if len(train_val_contracts) > 1:
                train_val_categories = [contract['category_id'] for contract in train_val_contracts]
                val_size = val_split / (train_split + val_split)
                
                train_contracts, val_contracts, _, _ = train_test_split(
                    train_val_contracts, train_val_categories,
                    test_size=val_size,
                    random_state=42,
                    stratify=train_val_categories
                )
            else:
                train_contracts = train_val_contracts
                val_contracts = []
        except ValueError as e:
            logger.warning(f"Stratified split failed: {e}. Using random split.")
            # Shuffle first
            contracts_shuffled = contracts.copy()
            random.shuffle(contracts_shuffled)
            
            train_end = int(len(contracts_shuffled) * train_split)
            val_end = train_end + int(len(contracts_shuffled) * val_split)
            
            train_contracts = contracts_shuffled[:train_end]
            val_contracts = contracts_shuffled[train_end:val_end]
            test_contracts = contracts_shuffled[val_end:]
        
        logger.info(f"Dataset split - Train: {len(train_contracts)}, "
                   f"Val: {len(val_contracts)}, Test: {len(test_contracts)}")
        
        return train_contracts, val_contracts, test_contracts 
    
    
    def _get_contract_detected_bugs(self, contract_id: str, tool: str, tool_results: Dict) -> set:
        """Get detected vulnerabilities for a specific contract by a specific tool."""
        detected_bugs = set()
        
        if contract_id not in tool_results:
            return detected_bugs
        
        if tool not in tool_results[contract_id]:
            return detected_bugs
        
        tool_result = tool_results[contract_id][tool]
        issues = tool_result.get('issues', [])
        
        for issue in issues:
            if isinstance(issue, dict):
                issue_type = issue.get('type', '')
                vuln_category = issue.get('vuln_category', '')
            else:
                issue_type = str(issue)
                vuln_category = ''
            
            # Map detected issue to standard vulnerability type
            mapped_type = self._map_detected_issue_to_standard_type(issue_type, vuln_category)
            if mapped_type:
                detected_bugs.add(mapped_type)
        
        return detected_bugs
    

    def get_detailed_confusion_matrix_per_vulnerability(self) -> Dict:
        """Get detailed confusion matrix breakdown per tool and per vulnerability type."""
        detailed_analysis = {}
        
        vulnerability_types = [
            'reentrancy', 'integer_overflow', 'unchecked_return',
            'timestamp_dependency', 'tx_origin', 'unhandled_exception', 'tod'
        ]
        
        for tool in self.tool_names:
            detailed_analysis[tool] = {}
            
            for vuln_type in vulnerability_types:
                detailed_analysis[tool][vuln_type] = {
                    'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0,
                    'contracts_with_vuln': 0,
                    'contracts_detected': 0,
                    'false_positive_contracts': [],
                    'false_negative_contracts': []
                }
        
        # Analyze each contract
        for contract in self.contracts:
            contract_id = contract['id']
            actual_vuln_type = contract.get('category_name', '')
            normalized_vuln = self._normalize_vulnerability_type(actual_vuln_type)
            
            if not normalized_vuln:
                continue
                
            for tool in self.tool_names:
                # Get tool results for this contract
                detected_vulns = self._get_contract_detected_bugs(contract_id, tool, self.tool_results)
                
                for vuln_type in vulnerability_types:
                    # True condition: contract has this vulnerability
                    has_vulnerability = (vuln_type == normalized_vuln)
                    # Predicted condition: tool detected this vulnerability
                    tool_detected = vuln_type in detected_vulns
                    
                    # Update confusion matrix
                    if has_vulnerability and tool_detected:
                        detailed_analysis[tool][vuln_type]['TP'] += 1
                    elif has_vulnerability and not tool_detected:
                        detailed_analysis[tool][vuln_type]['FN'] += 1
                        detailed_analysis[tool][vuln_type]['false_negative_contracts'].append(contract_id)
                    elif not has_vulnerability and tool_detected:
                        detailed_analysis[tool][vuln_type]['FP'] += 1
                        detailed_analysis[tool][vuln_type]['false_positive_contracts'].append(contract_id)
                    elif not has_vulnerability and not tool_detected:
                        detailed_analysis[tool][vuln_type]['TN'] += 1
                    
                    if has_vulnerability:
                        detailed_analysis[tool][vuln_type]['contracts_with_vuln'] += 1
                    if tool_detected:
                        detailed_analysis[tool][vuln_type]['contracts_detected'] += 1
        
        return detailed_analysis
    
