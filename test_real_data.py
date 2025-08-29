"""
Test script to verify the real data loading is working correctly.
"""

import yaml
import logging
from pathlib import Path
import sys
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from data.loader import SolidiFIDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_real_data_loading():
    """Test loading real tool performance data."""
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize data loader
    data_loader = SolidiFIDataLoader(config)
    
    # Test 1: Load contracts
    logger.info("\n=== Testing Contract Loading ===")
    contracts = data_loader.load_contracts()
    logger.info(f"Loaded {len(contracts)} contracts")
    
    if contracts:
        sample_contract = contracts[0]
        logger.info(f"Sample contract: {sample_contract['id']}")
        logger.info(f"  Category: {sample_contract['category_name']}")
        logger.info(f"  Vulnerability type: {sample_contract['vuln_type']}")
        logger.info(f"  Is buggy: {sample_contract['is_buggy']}")
    
    # Test 2: Load tool results
    logger.info("\n=== Testing Tool Results Loading ===")
    tool_results = data_loader.load_tool_results()
    logger.info(f"Loaded results for {len(tool_results)} contracts")
    
    # Show sample results
    if tool_results:
        sample_contract_id = next(iter(tool_results.keys()))
        sample_results = tool_results[sample_contract_id]
        logger.info(f"\nSample results for contract: {sample_contract_id}")
        
        for tool_name, tool_result in sample_results.items():
            issues = tool_result.get('issues', [])
            logger.info(f"  {tool_name}: {len(issues)} issues detected")
            if issues:
                logger.info(f"    First issue: {issues[0].get('type', 'Unknown')}")
    
    # Test 3: Load performance metrics
    logger.info("\n=== Testing Performance Metrics Calculation ===")
    performance_metrics = data_loader.load_performance_metrics()
    
    # Show average metrics per tool
    logger.info("\nTool Performance Summary:")
    logger.info("-" * 70)
    logger.info(f"{'Tool':<12} {'TPR':<8} {'FPR':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<8}")
    logger.info("-" * 70)
    
    for tool_name in data_loader.tool_names:
        if tool_name in performance_metrics and performance_metrics[tool_name]:
            avg_metrics = data_loader._calculate_average_metrics(performance_metrics[tool_name])
            logger.info(f"{tool_name.title():<12} "
                       f"{avg_metrics['tpr']:<8.3f} "
                       f"{avg_metrics['fpr']:<8.3f} "
                       f"{avg_metrics['accuracy']:<10.3f} "
                       f"{avg_metrics['precision']:<10.3f} "
                       f"{avg_metrics['recall']:<8.3f}")
    
    # Calculate overall statistics
    logger.info("\n=== Overall Statistics ===")
    
    # Count contracts per vulnerability type
    vuln_counts = {}
    for contract in contracts:
        vuln_type = contract['vuln_type']
        vuln_counts[vuln_type] = vuln_counts.get(vuln_type, 0) + 1
    
    logger.info("\nContracts per vulnerability type:")
    for vuln_type, count in sorted(vuln_counts.items()):
        logger.info(f"  {vuln_type}: {count} contracts")
    
    # Count tool coverage
    logger.info("\nTool coverage (contracts with results):")
    tool_coverage = {tool: 0 for tool in data_loader.tool_names}
    
    for contract_id, results in tool_results.items():
        for tool in results:
            if tool in tool_coverage:
                tool_coverage[tool] += 1
    
    for tool, count in sorted(tool_coverage.items()):
        percentage = (count / len(contracts) * 100) if contracts else 0
        logger.info(f"  {tool}: {count} contracts ({percentage:.1f}%)")
    
    # Test 4: Dataset split
    logger.info("\n=== Testing Dataset Split ===")
    train_contracts, val_contracts, test_contracts = data_loader.get_dataset_split(contracts)
    logger.info(f"Train: {len(train_contracts)} contracts")
    logger.info(f"Validation: {len(val_contracts)} contracts")
    logger.info(f"Test: {len(test_contracts)} contracts")
    
    # Verify split maintains vulnerability distribution
    logger.info("\nVulnerability distribution in splits:")
    for split_name, split_contracts in [("Train", train_contracts), 
                                       ("Val", val_contracts), 
                                       ("Test", test_contracts)]:
        split_vulns = {}
        for contract in split_contracts:
            vuln = contract['vuln_type']
            split_vulns[vuln] = split_vulns.get(vuln, 0) + 1
        
        logger.info(f"\n{split_name}:")
        for vuln, count in sorted(split_vulns.items()):
            logger.info(f"  {vuln}: {count}")
    
    # Test 5: Clear cache if requested
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--clear-cache', action='store_true', 
                       help='Clear cached data before testing')
    args = parser.parse_args()
    
    if args.clear_cache:
        logger.info("\n=== Clearing Cache ===")
        cache_dir = Path(config['data']['cache_dir'])
        if cache_dir.exists():
            for cache_file in cache_dir.glob('*.json'):
                cache_file.unlink()
                logger.info(f"Deleted: {cache_file.name}")
    
    logger.info("\n All tests completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_real_data_loading()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()