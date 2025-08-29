#!/usr/bin/env python3
"""
Script to reorganize SolidiFI-benchmark tool analysis results
from the original structure to the expected structure for the revised loader.py

Original structure:
    results/
    ├── Oyente/
    │   ├── analyzed_buggy_contracts/
    │   │   ├── Re-entrancy/
    │   │   │   ├── results/
    │   │   └── ...
    │   └── analyzed_clean_contracts/
    └── ...

Target structure:
    results/
    ├── <contract_name>/
    │   ├── mythril_result.json
    │   ├── slither_result.json
    │   └── ...
"""

import os
import json
from pandas import read_json
import yaml
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional
import re
import argparse


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SOLIDIFI_BUG_TYPES = [
    'Re-entrancy',
    'Timestamp-Dependency', 
    'Unchecked-Send',
    'Unhandled-Exceptions',
    'TOD',
    'Integer-Overflow-Underflow',
    'tx-origin'
]

class SolidiFIResultsReorganizer:
    def __init__(self, solidifi_path: str, output_path: str):
        self.solidifi_path = Path(solidifi_path)
        self.output_path = Path(output_path)
        self.original_results_path = self.solidifi_path / "results"
        
        # Tool name mappings (original -> standardized)
        self.tool_mapping = {
            'Oyente': 'oyente',
            'Securify': 'securify',
            'Mythril': 'mythril',
            'Smartcheck': 'smartcheck',
            'Manticore': 'manticore',
            'Slither': 'slither'
        }
        
        # Vulnerability type mappings
        self.vuln_type_mapping = {
            'Re-entrancy': 'reentrancy',
            'Reentrancy': 'reentrancy',
            'Integer-Overflow-Underflow': 'integer_overflow',
            'Integer_Overflow': 'integer_overflow',
            'Unchecked-Send': 'unchecked_return',
            'Unchecked_Return_Value': 'unchecked_return',
            'Timestamp-Dependency': 'timestamp_dependency',
            'Timestamp_Dependency': 'timestamp_dependency',
            'Authorization-through-tx-origin': 'tx_origin',
            'Authorization_Through_tx_origin': 'tx_origin',
            'Unhandled-Exceptions': 'unhandled_exception',
            'Unhandled_Exception': 'unhandled_exception',
            'Business_Logic_Error': 'business_logic'
        }
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True) 
        
    def load_yaml(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def load_json(path):
        with open(path, 'r') as f:
            return json.load(f)

    def build_dataset(contracts_dir, tool_outputs_dir, output_path):
        dataset = []

        for contract_file in os.listdir(contracts_dir):
            if not contract_file.endswith('.yaml'):
                continue

            contract_path = os.path.join(contracts_dir, contract_file)
            contract_data = yaml.load_all(contract_path)
            contract_id = contract_data.get('id', contract_file.replace('.yaml', ''))

            injected_bugs = contract_data.get('injected_bugs', [])
            formatted_bugs = []

            for bug in injected_bugs:
                bug_type = bug.get('type')
                line = bug.get('line')

                if bug_type not in SOLIDIFI_BUG_TYPES:
                    continue

                bug_entry = {
                    'type': bug_type,
                    'line': line,
                    'detected_by': {}
                }

                # For each tool, check if this bug type + line is reported
                for tool_file in os.listdir(tool_outputs_dir):
                    if not tool_file.endswith('.json'):
                        continue

                    tool_name = tool_file.replace('.json', '')
                    tool_output = read_json(os.path.join(tool_outputs_dir, tool_file))
                    tool_contract_issues = tool_output.get(contract_id, {}).get('issues', [])

                    detected = False
                    for issue in tool_contract_issues:
                        issue_type = issue.get('type') if isinstance(issue, dict) else issue
                        issue_line = issue.get('line') if isinstance(issue, dict) else None

                        if issue_type == bug_type and (issue_line is None or issue_line == line):
                            detected = True
                            break

                    bug_entry['detected_by'][tool_name] = detected

                formatted_bugs.append(bug_entry)

            dataset.append({
                'id': contract_id,
                'injected_bugs': formatted_bugs,
                'source_code': contract_data.get('source_code', '')
            })

        # Save to unified YAML
        with open(output_path, 'w') as out:
            yaml.dump(dataset, out, sort_keys=False)

        print(f"Dataset saved to: {output_path}")
        
    def reorganize_results(self):
        """Main method to reorganize all tool results."""
        logger.info(f"Starting reorganization of results from {self.original_results_path}")
        logger.info(f"Output directory: {self.output_path}")
        
        if not self.original_results_path.exists():
            logger.error(f"Results directory not found: {self.original_results_path}")
            return
        
        # Process each tool's results
        for tool_dir in self.original_results_path.iterdir():
            if tool_dir.is_dir() and tool_dir.name in self.tool_mapping:
                logger.info(f"Processing {tool_dir.name} results...")
                self.process_tool_results(tool_dir, self.tool_mapping[tool_dir.name])
        
        logger.info("Reorganization completed!")
        
    def process_tool_results(self, tool_dir: Path, tool_name: str):
        """Process results for a specific tool."""
        processed_count = 0
        
        # Process buggy contracts
        buggy_dir = tool_dir / "analyzed_buggy_contracts"
        if buggy_dir.exists():
            processed_count += self.process_contract_category(buggy_dir, tool_name, is_buggy=True)
        
        # Process clean contracts
        clean_dir = tool_dir / "analyzed_clean_contracts"
        if clean_dir.exists():
            processed_count += self.process_contract_category(clean_dir, tool_name, is_buggy=False)
        
        logger.info(f"Processed {processed_count} {tool_name} results")
        
    def process_contract_category(self, category_dir: Path, tool_name: str, is_buggy: bool) -> int:
        """Process contracts in a category (buggy or clean)."""
        processed_count = 0
        category_type = "buggy" if is_buggy else "clean"
        
        # Iterate through vulnerability type directories
        for vuln_dir in category_dir.iterdir():
            if vuln_dir.is_dir():
                vuln_type = self.normalize_vuln_type(vuln_dir.name)
                logger.debug(f"Processing {category_type} {vuln_type} contracts for {tool_name}")
                
                # Look for results directory
                results_dir = vuln_dir / "results"
                if results_dir.exists():
                    processed_count += self.process_results_directory(
                        results_dir, tool_name, vuln_type, is_buggy
                    )
                else:
                    # Sometimes results might be directly in the vuln directory
                    processed_count += self.process_results_directory(
                        vuln_dir, tool_name, vuln_type, is_buggy
                    )
        
        return processed_count
    
    def process_results_directory(self, results_dir: Path, tool_name: str, 
                                vuln_type: str, is_buggy: bool) -> int:
        """Process individual result files in a directory."""
        processed_count = 0
        
        # Process each result file
        for result_file in results_dir.iterdir():
            if result_file.is_file() and self.is_result_file(result_file):
                contract_name = self.extract_contract_name(result_file.name)
                if contract_name:
                    # Create contract output directory
                    contract_output_dir = self.output_path / contract_name
                    contract_output_dir.mkdir(exist_ok=True)
                    
                    # Process and save the result
                    if self.process_result_file(result_file, contract_output_dir, 
                                              tool_name, vuln_type, is_buggy):
                        processed_count += 1
        
        return processed_count
    
    def is_result_file(self, file_path: Path) -> bool:
        """Check if a file is a tool result file."""
        # Common extensions for tool results
        valid_extensions = ['.json', '.txt', '.log', '.out', '.xml']
        return file_path.suffix.lower() in valid_extensions
    
    def extract_contract_name(self, filename: str) -> Optional[str]:
        """Extract contract name from result filename."""
        # Remove common suffixes and extensions
        contract_name = filename
        
        # Remove file extension
        for ext in ['.json', '.txt', '.log', '.out', '.xml']:
            if contract_name.endswith(ext):
                contract_name = contract_name[:-len(ext)]
                break
        
        # Remove tool-specific suffixes
        suffixes_to_remove = [
            '_result', '_results', '_output', '_report',
            '_analysis', '_analyzed', '_buggy', '_clean'
        ]
        
        for suffix in suffixes_to_remove:
            if contract_name.endswith(suffix):
                contract_name = contract_name[:-len(suffix)]
        
        # Remove timestamp patterns (e.g., _20230101_123456)
        contract_name = re.sub(r'_\d{8}_\d{6}$', '', contract_name)
        contract_name = re.sub(r'_\d{10,}$', '', contract_name)  # Unix timestamp
        
        # Clean up the name
        contract_name = contract_name.strip('_- ')
        
        return contract_name if contract_name else None
    
    def normalize_vuln_type(self, vuln_type: str) -> str:
        """Normalize vulnerability type name."""
        return self.vuln_type_mapping.get(vuln_type, vuln_type.lower().replace('-', '_'))
    
    def process_result_file(self, result_file: Path, output_dir: Path, 
                          tool_name: str, vuln_type: str, is_buggy: bool) -> bool:
        """Process and convert a single result file to the expected format."""
        try:
            # Read the original result
            result_data = self.read_result_file(result_file)
            
            # Convert to standardized format based on tool
            standardized_result = self.standardize_result(
                result_data, tool_name, vuln_type, is_buggy, result_file
            )
            
            # Save the standardized result
            output_file = output_dir / f"{tool_name}_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(standardized_result, f, indent=2)
            
            logger.debug(f"Saved {tool_name} result to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {result_file}: {e}")
            return False
    
    def read_result_file(self, file_path: Path) -> Dict:
        """Read a result file and return its content."""
        if file_path.suffix.lower() == '.json':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                # Try to read as text and parse
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return self.parse_text_result(content)
        else:
            # Read as text
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.parse_text_result(content)
    
    def parse_text_result(self, content: str) -> Dict:
        """Parse text-based tool output."""
        return {
            'raw_output': content,
            'format': 'text'
        }
    
    def standardize_result(self, result_data: Dict, tool_name: str, 
                         vuln_type: str, is_buggy: bool, file_path: Path) -> Dict:
        """Convert tool-specific result to standardized format."""
        
        if tool_name == 'oyente':
            return self.standardize_oyente_result(result_data, vuln_type, is_buggy)
        elif tool_name == 'mythril':
            return self.standardize_mythril_result(result_data, vuln_type, is_buggy)
        elif tool_name == 'slither':
            return self.standardize_slither_result(result_data, vuln_type, is_buggy)
        elif tool_name == 'securify':
            return self.standardize_securify_result(result_data, vuln_type, is_buggy)
        elif tool_name == 'smartcheck':
            return self.standardize_smartcheck_result(result_data, vuln_type, is_buggy)
        elif tool_name == 'manticore':
            return self.standardize_manticore_result(result_data, vuln_type, is_buggy)
        else:
            # Generic format
            return {
                'tool': tool_name,
                'vulnerability_type': vuln_type,
                'is_buggy': is_buggy,
                'data': result_data
            }
    
    def standardize_oyente_result(self, result_data: Dict, vuln_type: str, is_buggy: bool) -> Dict:
        """Standardize Oyente result format."""
        # Oyente typically has these fields
        standardized = {
            'tool': 'oyente',
            'vulnerability_type': vuln_type,
            'is_buggy': is_buggy,
            'callstack': result_data.get('callstack', False),
            'time_dependency': result_data.get('time_dependency', False),
            'reentrancy': result_data.get('reentrancy', False),
            'money_concurrency': result_data.get('money_concurrency', False),
            'assertion_failure': result_data.get('assertion_failure', False)
        }
        
        # If raw text output
        if 'raw_output' in result_data:
            standardized['raw_output'] = result_data['raw_output']
            # Try to parse vulnerabilities from text
            text = result_data['raw_output'].lower()
            standardized['callstack'] = 'callstack' in text and 'true' in text
            standardized['time_dependency'] = 'time dependency' in text or 'timestamp' in text
            standardized['reentrancy'] = 'reentrancy' in text or 're-entrancy' in text
        
        return standardized
    
    def standardize_mythril_result(self, result_data: Dict, vuln_type: str, is_buggy: bool) -> Dict:
        """Standardize Mythril result format."""
        standardized = {
            'tool': 'mythril',
            'vulnerability_type': vuln_type,
            'is_buggy': is_buggy,
            'success': True,
            'issues': []
        }
        
        # Check if it's already in Mythril's JSON format
        if 'issues' in result_data:
            standardized['issues'] = result_data['issues']
        elif 'results' in result_data:
            standardized['issues'] = result_data['results']
        elif 'raw_output' in result_data:
            # Parse from text output
            standardized['raw_output'] = result_data['raw_output']
            standardized['issues'] = self.parse_mythril_text_output(result_data['raw_output'])
        
        return standardized
    
    def parse_mythril_text_output(self, text: str) -> List[Dict]:
        """Parse Mythril text output to extract issues."""
        issues = []
        
        # Look for patterns like "==== Issue Title ===="
        issue_pattern = r'====\s*(.+?)\s*===='
        matches = re.finditer(issue_pattern, text)
        
        for match in matches:
            issue = {
                'title': match.group(1).strip(),
                'description': '',
                'severity': 'Medium',  # Default
                'swc-id': ''
            }
            
            # Extract severity if present
            if 'severity:' in text.lower():
                severity_match = re.search(r'severity:\s*(\w+)', text, re.IGNORECASE)
                if severity_match:
                    issue['severity'] = severity_match.group(1).capitalize()
            
            # Extract SWC ID if present
            swc_match = re.search(r'SWC-(\d+)', text)
            if swc_match:
                issue['swc-id'] = f'SWC-{swc_match.group(1)}'
            
            issues.append(issue)
        
        return issues
    
    def standardize_slither_result(self, result_data: Dict, vuln_type: str, is_buggy: bool) -> Dict:
        """Standardize Slither result format."""
        standardized = {
            'tool': 'slither',
            'vulnerability_type': vuln_type,
            'is_buggy': is_buggy,
            'success': True,
            'results': {
                'detectors': []
            }
        }
        
        # Check various possible formats
        if 'results' in result_data and 'detectors' in result_data['results']:
            standardized['results']['detectors'] = result_data['results']['detectors']
        elif 'detectors' in result_data:
            standardized['results']['detectors'] = result_data['detectors']
        elif isinstance(result_data, list):
            # Sometimes Slither outputs a list of issues
            standardized['results']['detectors'] = result_data
        elif 'raw_output' in result_data:
            standardized['raw_output'] = result_data['raw_output']
            # Parse text output if needed
            standardized['results']['detectors'] = self.parse_slither_text_output(result_data['raw_output'])
        
        return standardized
    
    def parse_slither_text_output(self, text: str) -> List[Dict]:
        """Parse Slither text output."""
        detectors = []
        
        # Look for patterns indicating issues
        lines = text.split('\n')
        current_issue = None
        
        for line in lines:
            # Slither often uses color codes and specific formatting
            if 'Reference:' in line or 'Check:' in line:
                if current_issue:
                    detectors.append(current_issue)
                
                current_issue = {
                    'check': line.split(':')[-1].strip() if ':' in line else 'unknown',
                    'impact': 'Medium',
                    'confidence': 'Medium',
                    'description': ''
                }
            elif current_issue and line.strip():
                current_issue['description'] += line.strip() + ' '
        
        if current_issue:
            detectors.append(current_issue)
        
        return detectors
    
    def standardize_securify_result(self, result_data: Dict, vuln_type: str, is_buggy: bool) -> Dict:
        """Standardize Securify result format."""
        standardized = {
            'tool': 'securify',
            'vulnerability_type': vuln_type,
            'is_buggy': is_buggy,
            'results': {}
        }
        
        # Securify outputs results per contract
        if 'results' in result_data:
            standardized['results'] = result_data['results']
        elif 'raw_output' in result_data:
            standardized['raw_output'] = result_data['raw_output']
            # Parse patterns from text
            standardized['results'] = self.parse_securify_text_output(result_data['raw_output'])
        else:
            # Assume the whole data is the results
            standardized['results'] = result_data
        
        return standardized
    
    def parse_securify_text_output(self, text: str) -> Dict:
        """Parse Securify text output."""
        results = {'Contract': {}}
        
        # Look for violation patterns
        patterns = ['DAO', 'TODReceiver', 'TODTransfer', 'UnhandledException', 
                   'Reentrancy', 'LockedEther', 'MissingInputValidation']
        
        for pattern in patterns:
            if pattern.lower() in text.lower():
                results['Contract'][pattern] = {
                    'violations': ['Unknown location'],
                    'warnings': []
                }
        
        return results
    
    def standardize_smartcheck_result(self, result_data: Dict, vuln_type: str, is_buggy: bool) -> Dict:
        """Standardize SmartCheck result format."""
        standardized = {
            'tool': 'smartcheck',
            'vulnerability_type': vuln_type,
            'is_buggy': is_buggy,
            'rules': []
        }
        
        if 'rules' in result_data:
            standardized['rules'] = result_data['rules']
        elif 'findings' in result_data:
            # Convert findings to rules format
            for finding in result_data['findings']:
                rule = {
                    'id': finding.get('rule', 'unknown'),
                    'severity': finding.get('severity', 'medium'),
                    'findings': [finding]
                }
                standardized['rules'].append(rule)
        elif 'raw_output' in result_data:
            standardized['raw_output'] = result_data['raw_output']
            standardized['rules'] = self.parse_smartcheck_text_output(result_data['raw_output'])
        
        return standardized
    
    def parse_smartcheck_text_output(self, text: str) -> List[Dict]:
        """Parse SmartCheck text output."""
        rules = []
        
        # Look for rule violations
        rule_pattern = r'Rule:\s*(\w+)'
        severity_pattern = r'Severity:\s*(\w+)'
        
        rule_matches = re.finditer(rule_pattern, text)
        for match in rule_matches:
            rule = {
                'id': match.group(1),
                'severity': 'medium',
                'findings': [{'line': 0, 'column': 0}]
            }
            
            # Try to find severity
            severity_match = re.search(severity_pattern, text[match.start():match.start()+200])
            if severity_match:
                rule['severity'] = severity_match.group(1).lower()
            
            rules.append(rule)
        
        return rules
    
    def standardize_manticore_result(self, result_data: Dict, vuln_type: str, is_buggy: bool) -> Dict:
        """Standardize Manticore result format."""
        standardized = {
            'tool': 'manticore',
            'vulnerability_type': vuln_type,
            'is_buggy': is_buggy,
            'issues': []
        }
        
        if 'issues' in result_data:
            standardized['issues'] = result_data['issues']
        elif 'findings' in result_data:
            standardized['issues'] = result_data['findings']
        elif 'raw_output' in result_data:
            standardized['raw_output'] = result_data['raw_output']
            standardized['issues'] = self.parse_manticore_text_output(result_data['raw_output'])
        
        return standardized
    
    def parse_manticore_text_output(self, text: str) -> List[Dict]:
        """Parse Manticore text output."""
        issues = []
        
        # Look for issue indicators
        if 'ASSERTION FAIL' in text or 'assertion failed' in text.lower():
            issues.append({
                'type': 'Assertion Failure',
                'severity': 'High',
                'description': 'Assertion failure detected'
            })
        
        if 'integer overflow' in text.lower():
            issues.append({
                'type': 'Integer Overflow',
                'severity': 'High',
                'description': 'Integer overflow detected'
            })
        
        if 'reentrancy' in text.lower() or 're-entrancy' in text.lower():
            issues.append({
                'type': 'Reentrancy',
                'severity': 'High',
                'description': 'Reentrancy vulnerability detected'
            })
        
        return issues
    
    def create_summary_report(self):
        """Create a summary report of the reorganization."""
        summary = {
            'total_contracts': 0,
            'tools_processed': {},
            'contracts_by_vulnerability': {}
        }
        
        # Count contracts and gather statistics
        for contract_dir in self.output_path.iterdir():
            if contract_dir.is_dir():
                summary['total_contracts'] += 1
                
                # Count tools per contract
                for tool_file in contract_dir.iterdir():
                    if tool_file.name.endswith('_result.json'):
                        tool_name = tool_file.name.replace('_result.json', '')
                        summary['tools_processed'][tool_name] = summary['tools_processed'].get(tool_name, 0) + 1
                        
                        # Read vulnerability type
                        try:
                            with open(tool_file, 'r') as f:
                                data = json.load(f)
                                vuln_type = data.get('vulnerability_type', 'unknown')
                                summary['contracts_by_vulnerability'][vuln_type] = \
                                    summary['contracts_by_vulnerability'].get(vuln_type, 0) + 1
                        except:
                            pass
        
        # Save summary
        summary_file = self.output_path / 'reorganization_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary report saved to {summary_file}")
        logger.info(f"Total contracts processed: {summary['total_contracts']}")
        logger.info(f"Tools processed: {summary['tools_processed']}")


def main():
    parser = argparse.ArgumentParser(
        description='Reorganize SolidiFI-benchmark tool results for the revised loader'
    )
    parser.add_argument(
        'solidifi_path',
        help='Path to the SolidiFI-benchmark directory'
    )
    parser.add_argument(
        '-o', '--output',
        default='results',
        help='Output directory for reorganized results (default: results)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create reorganizer and process results
    reorganizer = SolidiFIResultsReorganizer(args.solidifi_path, args.output)
    reorganizer.reorganize_results()
    reorganizer.create_summary_report() 
    
    # Save processed contract list to YAML
    # with open("results/processed_data.yaml", "w") as f:
    #     yaml.dump(all_contracts, f) 


if __name__ == '__main__':
    main()