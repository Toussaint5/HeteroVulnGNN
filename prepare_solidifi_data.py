"""
prepare_solidifi_data.py - Validate and organize existing SolidiFI benchmark data with tool results
"""

import os
import json
import csv
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Optional, Set
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SolidiFIDataValidator:
    def __init__(self, solidifi_path: str, results_path: str):
        self.solidifi_path = Path(solidifi_path)
        self.results_path = Path(results_path)
        
        self.tools = ['Oyente', 'Securify', 'Mythril', 'Smartcheck', 'Manticore', 'Slither']
        
        # Mapping between directory names and standardized names
        self.vuln_mapping = {
            'Reentrancy': 'Re-entrancy',
            'Re-entrancy': 'Re-entrancy',
            'Overflow-Underflow': 'Integer-Overflow',
            'Integer_Overflow': 'Integer-Overflow',
            'Integer_Underflow': 'Integer-Overflow',
            'Unchecked_Return_Value': 'Unchecked-Return-Value',
            'Timestamp_Dependency': 'Timestamp-Dependency',
            'Authorization_Through_tx_origin': 'Authorization-Through-tx-origin',
            'Unhandled_Exception': 'Unhandled-Exception',
            'Business_Logic_Error': 'Business-Logic-Error',
            'Unchecked-Send': 'Unchecked-Send',
            'TOD': 'TOD',  # Transaction Order Dependency
            'tx.origin': 'tx.origin'
        }
        
        # Tool output file patterns - updated based on actual results
        self.tool_patterns = {
            'Oyente': r'buggy_\d+\.sol.*\.json$',  # e.g., buggy_1.sol_HotDollarsToken.json
            'Mythril': [r'buggy_\d+\.sol$', r'buggy_\d+\.txt$', r'buggy_\d+.*\.txt$'],  # Multiple possible formats
            'Manticore': r'buggy_\d+\..*\.txt$',   # e.g., buggy_1.HotDollarsToken.txt
            'Securify': [r'buggy_\d+\.sol$', r'buggy_\d+\.txt$', r'buggy_\d+.*\.txt$'],  # Multiple possible formats
            'Slither': [r'buggy_\d+\.sol$', r'buggy_\d+\.json$', r'buggy_\d+.*\.json$'],  # Multiple possible formats
            'Smartcheck': [r'buggy_\d+\.sol\.txt$', r'buggy_\d+\.sol$', r'buggy_\d+\.txt$', r'buggy_\d+.*\.txt$']  # Multiple formats
        }
        
    def validate_and_report(self):
        """Validate the existing structure and report status."""
        logger.info(f"Validating SolidiFI data structure...")
        logger.info(f"SolidiFI path: {self.solidifi_path}")
        logger.info(f"Results path: {self.results_path}")
        
        # Step 1: Check if paths exist
        if not self.solidifi_path.exists():
            logger.error(f"SolidiFI path does not exist: {self.solidifi_path}")
            return False
            
        if not self.results_path.exists():
            logger.error(f"Results path does not exist: {self.results_path}")
            return False
        
        # Step 2: Validate tool directories
        validation_results = {}
        for tool in self.tools:
            tool_results = self._validate_tool_structure(tool)
            validation_results[tool] = tool_results
        
        # Step 3: Generate summary report
        self._generate_summary_report(validation_results)
        
        # Step 4: Create data statistics
        self._generate_data_statistics(validation_results)
        
        # Step 5: Explore missing results
        self._explore_missing_results(validation_results)
        
        return True
        
    def _validate_tool_structure(self, tool: str) -> Dict:
        """Validate structure for a specific tool."""
        logger.info(f"\nValidating {tool}...")
        
        tool_path = self.results_path / tool / 'analyzed_buggy_contracts'
        results = {
            'exists': tool_path.exists(),
            'vulnerabilities': {},
            'total_contracts': 0,
            'total_results': 0,
            'missing_results': [],
            'result_files_sample': []
        }
        
        if not tool_path.exists():
            logger.warning(f"Tool directory not found: {tool_path}")
            return results
        
        # Check each vulnerability type
        for vuln_dir in tool_path.iterdir():
            if not vuln_dir.is_dir():
                continue
                
            vuln_name = vuln_dir.name
            results_dir = vuln_dir / 'results'
            
            # Count contracts and results
            contracts = list(vuln_dir.glob('buggy_*.sol'))
            bug_logs = list(vuln_dir.glob('BugLog_*.csv'))
            
            result_files = []
            if results_dir.exists():
                # Get result files based on tool pattern(s)
                patterns = self.tool_patterns.get(tool, [r'.*'])
                if not isinstance(patterns, list):
                    patterns = [patterns]
                
                # Try each pattern
                for pattern in patterns:
                    matching_files = [f for f in results_dir.iterdir() 
                                    if f.is_file() and re.match(pattern, f.name)]
                    result_files.extend(matching_files)
                
                # Remove duplicates
                result_files = list(set(result_files))
                
                # If still no results, list all files for debugging
                if not result_files and not results['result_files_sample']:
                    all_files = list(results_dir.iterdir())
                    if all_files:
                        results['result_files_sample'] = [f.name for f in all_files[:5]]
                        logger.debug(f"  No matching files found. Sample files in {vuln_name}/results: {results['result_files_sample']}")
            
            # For Oyente, count unique contracts (not individual contract files)
            if tool == 'Oyente' and result_files:
                unique_contracts = set()
                for rf in result_files:
                    match = re.match(r'(buggy_\d+)\.sol', rf.name)
                    if match:
                        unique_contracts.add(match.group(1))
                actual_result_count = len(unique_contracts)
            else:
                actual_result_count = len(result_files)
            
            results['vulnerabilities'][vuln_name] = {
                'contracts': len(contracts),
                'bug_logs': len(bug_logs),
                'results': actual_result_count,
                'result_files': [f.name for f in result_files[:5]],  # Sample files
                'missing': max(0, len(contracts) - actual_result_count)
            }
            
            results['total_contracts'] += len(contracts)
            results['total_results'] += actual_result_count
            
            # Track missing results
            contract_names = {c.stem for c in contracts}
            result_contract_names = set()
            
            for rf in result_files:
                # Extract contract name from result file
                match = re.match(r'(buggy_\d+)', rf.stem)
                if match:
                    result_contract_names.add(match.group(1))
            
            missing = contract_names - result_contract_names
            for m in missing:
                results['missing_results'].append(f"{vuln_name}/{m}.sol")
        
        return results
    
    def _explore_missing_results(self, validation_results: Dict):
        """Explore why certain tools have no results."""
        logger.info("\n" + "="*80)
        logger.info("EXPLORING MISSING RESULTS")
        logger.info("="*80)
        
        for tool, results in validation_results.items():
            if results['exists'] and results['total_results'] == 0:
                logger.info(f"\n{tool} has no results. Exploring directory structure...")
                
                tool_path = self.results_path / tool / 'analyzed_buggy_contracts'
                
                # List all subdirectories
                subdirs = [d.name for d in tool_path.iterdir() if d.is_dir()]
                logger.info(f"  Vulnerability directories: {subdirs}")
                
                # Check each subdirectory
                for subdir in subdirs[:2]:  # Check first 2 subdirs
                    vuln_path = tool_path / subdir
                    logger.info(f"\n  Checking {subdir}:")
                    
                    # List contents
                    contents = list(vuln_path.iterdir())
                    logger.info(f"    Total items: {len(contents)}")
                    
                    # Check for results directory
                    results_dir = vuln_path / 'results'
                    if results_dir.exists():
                        result_files = list(results_dir.iterdir())
                        logger.info(f"    Results directory exists with {len(result_files)} files")
                        if result_files:
                            logger.info(f"    Sample files: {[f.name for f in result_files[:5]]}")
                    else:
                        logger.info(f"    No results directory found")
                        
                        # Check if results are in the main directory
                        possible_results = [f for f in vuln_path.iterdir() 
                                          if f.is_file() and 'buggy' in f.name.lower()]
                        if possible_results:
                            logger.info(f"    Found {len(possible_results)} possible result files in main directory")
                            logger.info(f"    Sample: {[f.name for f in possible_results[:3]]}")
        
    def _generate_summary_report(self, validation_results: Dict):
        """Generate a summary report of the validation."""
        logger.info("\n" + "="*80)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*80)
        
        total_contracts = 0
        total_results = 0
        tools_with_results = []
        
        for tool, results in validation_results.items():
            logger.info(f"\n{tool}:")
            logger.info(f"  Status: {'✓ Found' if results['exists'] else '✗ Not Found'}")
            
            if results['exists']:
                coverage = (results['total_results']/results['total_contracts']*100 
                           if results['total_contracts'] > 0 else 0)
                           
                logger.info(f"  Total contracts: {results['total_contracts']}")
                logger.info(f"  Total results: {results['total_results']}")
                logger.info(f"  Coverage: {coverage:.1f}%")
                
                if results['total_results'] > 0:
                    tools_with_results.append((tool, coverage))
                
                total_contracts += results['total_contracts']
                total_results += results['total_results']
                
                # Show vulnerability breakdown
                logger.info(f"  Vulnerabilities:")
                for vuln, stats in results['vulnerabilities'].items():
                    logger.info(f"    {vuln}: {stats['results']}/{stats['contracts']} results")
                    
                    # Show sample result files
                    if stats['result_files']:
                        logger.info(f"      Sample files: {', '.join(stats['result_files'][:3])}")
                    elif results['result_files_sample']:
                        logger.info(f"      No matching files. Directory contains: {results['result_files_sample'][:3]}")
        
        logger.info(f"\nOVERALL:")
        logger.info(f"  Total contracts across all tools: {total_contracts}")
        logger.info(f"  Total results across all tools: {total_results}")
        logger.info(f"  Overall coverage: {total_results/total_contracts*100:.1f}%" 
                  if total_contracts > 0 else "N/A")
        logger.info(f"  Tools with results: {', '.join([t[0] for t in tools_with_results])}")
        
        # Recommendations
        logger.info(f"\nRECOMMENDATIONS:")
        if len(tools_with_results) < 3:
            logger.warning("  ⚠️  Less than 3 tools have analysis results!")
            logger.warning("  The model needs results from multiple tools to learn effectively.")
        
        for tool, results in validation_results.items():
            if results['exists'] and results['total_results'] == 0:
                logger.warning(f"  ⚠️  {tool} has no analysis results in the expected format")
                if results['result_files_sample']:
                    logger.info(f"     Found files: {results['result_files_sample']}")
                    logger.info(f"     Expected pattern: {self.tool_patterns.get(tool, 'unknown')}")
        
    def _generate_data_statistics(self, validation_results: Dict):
        """Generate statistics about the data."""
        stats_file = self.results_path / 'data_statistics.json'
        
        stats = {
            'summary': {
                'total_tools': len(self.tools),
                'tools_found': sum(1 for r in validation_results.values() if r['exists']),
                'tools_with_results': sum(1 for r in validation_results.values() 
                                        if r['exists'] and r['total_results'] > 0),
                'total_contracts': sum(r['total_contracts'] for r in validation_results.values()),
                'total_results': sum(r['total_results'] for r in validation_results.values())
            },
            'per_tool': {},
            'per_vulnerability': {},
            'recommendations': []
        }
        
        # Per-tool statistics
        for tool, results in validation_results.items():
            coverage = (results['total_results'] / results['total_contracts'] * 100 
                       if results['total_contracts'] > 0 else 0)
            
            stats['per_tool'][tool] = {
                'found': results['exists'],
                'contracts': results['total_contracts'],
                'results': results['total_results'],
                'coverage': coverage,
                'has_results': results['total_results'] > 0
            }
        
        # Per-vulnerability statistics
        vuln_stats = {}
        for tool, results in validation_results.items():
            if results['exists']:
                for vuln, vuln_data in results['vulnerabilities'].items():
                    if vuln not in vuln_stats:
                        vuln_stats[vuln] = {
                            'total_contracts': 0,
                            'total_results': 0,
                            'tools': []
                        }
                    vuln_stats[vuln]['total_contracts'] += vuln_data['contracts']
                    vuln_stats[vuln]['total_results'] += vuln_data['results']
                    if vuln_data['results'] > 0:
                        vuln_stats[vuln]['tools'].append(tool)
        
        stats['per_vulnerability'] = vuln_stats
        
        # Add recommendations
        if stats['summary']['tools_with_results'] < 3:
            stats['recommendations'].append(
                "Need results from more tools. Currently only " + 
                f"{stats['summary']['tools_with_results']} tools have results."
            )
        
        # Save statistics
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"\nStatistics saved to: {stats_file}")
        
    def test_parsing(self):
        """Test parsing of sample files from each tool."""
        logger.info("\n" + "="*80)
        logger.info("TESTING TOOL OUTPUT PARSING")
        logger.info("="*80)
        
        for tool in self.tools:
            logger.info(f"\nTesting {tool} output parsing...")
            
            # Find a sample result file
            tool_path = self.results_path / tool / 'analyzed_buggy_contracts'
            if not tool_path.exists():
                logger.warning(f"  Tool directory not found")
                continue
                
            sample_found = False
            for vuln_dir in tool_path.iterdir():
                if not vuln_dir.is_dir():
                    continue
                    
                results_dir = vuln_dir / 'results'
                if not results_dir.exists():
                    continue
                
                # Get any file from results directory
                for result_file in results_dir.iterdir():
                    if result_file.is_file():
                        logger.info(f"  Sample file: {result_file.name}")
                        logger.info(f"  File type: {result_file.suffix}")
                        self._test_parse_file(result_file, tool)
                        sample_found = True
                        break
                        
                if sample_found:
                    break
                    
            if not sample_found:
                logger.warning(f"  No sample result files found")
                
    def _test_parse_file(self, file_path: Path, tool: str):
        """Test parsing a specific file."""
        try:
            # Check file size
            file_size = file_path.stat().st_size
            logger.info(f"  File size: {file_size} bytes")
            
            if file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"  ✓ Successfully parsed JSON")
                logger.info(f"  Found keys: {list(data.keys())[:10]}")  # Show first 10 keys
                
                # Oyente format
                if 'vulnerabilities' in data:
                    vulns = data['vulnerabilities']
                    for vuln_type, locations in vulns.items():
                        if locations:
                            logger.info(f"  Detected {vuln_type}: {len(locations)} instances")
                            
            elif file_path.suffix in ['.txt', '.sol']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                logger.info(f"  ✓ Successfully read file ({len(content)} chars)")
                
                # Detect format by content
                if tool in ['Mythril', 'Securify', 'Slither'] and file_path.suffix == '.sol':
                    # These tools output to .sol files with analysis results
                    if '====' in content and 'SWC ID:' in content:
                        logger.info(f"  Detected Mythril analysis format")
                    elif 'Severity:' in content:
                        logger.info(f"  Detected Slither/Securify format")
                    else:
                        # Show first few lines to understand format
                        lines = content.split('\n')[:5]
                        logger.info(f"  First few lines: {lines}")
                        
                elif tool == 'Smartcheck':
                    if 'ruleId:' in content:
                        logger.info(f"  Detected SmartCheck format")
                        rule_count = content.count('ruleId:')
                        logger.info(f"  Found {rule_count} rules/issues")
                        
                elif tool == 'Manticore':
                    if 'EVM Program counter:' in content:
                        logger.info(f"  Detected Manticore format")
                        issue_count = content.count('- ')
                        logger.info(f"  Found approximately {issue_count} issues")
                        
        except Exception as e:
            logger.error(f"  ✗ Failed to parse: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Validate SolidiFI benchmark data structure with tool results'
    )
    
    parser.add_argument('--solidifi', '-s', required=True, 
                       help='Path to SolidiFI-benchmark directory')
    parser.add_argument('--results', '-r', required=True,
                       help='Path to results directory')
    parser.add_argument('--test-parsing', '-t', action='store_true',
                       help='Test parsing of tool outputs')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    validator = SolidiFIDataValidator(args.solidifi, args.results)
    
    # Validate structure
    if validator.validate_and_report():
        logger.info("\n✓ Validation complete!")
        
        # Optionally test parsing
        if args.test_parsing:
            validator.test_parsing()
    else:
        logger.error("\n✗ Validation failed!")

if __name__ == '__main__':
    main()