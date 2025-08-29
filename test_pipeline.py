"""
Simple test script to verify the data loading pipeline works correctly.
"""

import os
import sys
import yaml
import logging
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_pipeline():
    """Test the complete data loading pipeline."""
    
    # Load configuration
    config_path = "config/config.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        logger.error("Please create config/config.yaml first")
        return False
    
    try:
        # Test 1: Import all modules
        logger.info("Testing module imports...")
        from data.loader import SolidiFIDataLoader
        from data.processor import SolidityCodeProcessor
        from data.graph_builder import HeterogeneousGraphBuilder
        from models.heterognn import HeteroToolGNN  # FIXED: Correct class name
        logger.info("✓ All modules imported successfully")
        
        # Test 2: Initialize data loader
        logger.info("Testing data loader initialization...")
        data_loader = SolidiFIDataLoader(config)
        logger.info("✓ Data loader initialized")
        
        # Test 3: Load contracts
        logger.info("Testing contract loading...")
        contracts = data_loader.load_contracts()
        if not contracts:
            logger.error("No contracts loaded!")
            logger.error("Please check that SolidiFI-benchmark directory exists and contains data")
            return False
        logger.info(f"✓ Loaded {len(contracts)} contracts")
        
        # Display sample contract info
        sample_contract = contracts[0]
        logger.info(f"Sample contract: {sample_contract['id']}")
        logger.info(f"  Category: {sample_contract['category_name']}")
        logger.info(f"  Vulnerability type: {sample_contract['vuln_type']}")
        
        # Test 4: Test data split
        logger.info("Testing data split...")
        train_contracts, val_contracts, test_contracts = data_loader.get_dataset_split(contracts)
        logger.info(f"✓ Split: {len(train_contracts)} train, {len(val_contracts)} val, {len(test_contracts)} test")
        
        if len(train_contracts) == 0:
            logger.error("No training contracts available!")
            return False
        
        # Test 5: Initialize processor and graph builder
        logger.info("Testing processor and graph builder...")
        processor = SolidityCodeProcessor(config)
        graph_builder = HeterogeneousGraphBuilder(config)
        logger.info("✓ Processor and graph builder initialized")
        
        # Test 6: Load additional data
        logger.info("Testing additional data loading...")
        tool_results = data_loader.load_tool_results()
        injected_bugs = data_loader.load_injected_bugs()
        performance_metrics = data_loader.load_performance_metrics()
        logger.info(f"✓ Additional data loaded")
        logger.info(f"  Tool results: {len(tool_results)} contracts")
        logger.info(f"  Injected bugs: {len(injected_bugs)} contracts")
        logger.info(f"  Performance metrics: {len(performance_metrics)} tools")
        
        # Test 7: Process a sample contract
        logger.info("Testing contract processing...")
        sample_contract = train_contracts[0]
        
        try:
            # Extract AST features
            ast_features = processor.extract_ast_features(sample_contract.get('ast', {}))
            logger.info(f"✓ Extracted {len(ast_features)} AST features")
            
            # Build graph
            hetero_data = graph_builder.build_heterogeneous_graph(
                sample_contract, ast_features, processor
            )
            logger.info(f"✓ Built heterogeneous graph with node types: {list(hetero_data.node_types)}")
            
            # Add labels
            contract_id = sample_contract['id']
            tool_labels = processor.create_tool_performance_labels(
                contract_id, injected_bugs, tool_results, performance_metrics
            )
            
            hetero_data = graph_builder.add_tool_performance_labels(hetero_data, tool_labels)
            logger.info("✓ Added labels to graph")
            logger.info(f"  Tool labels shape: {tool_labels.shape}")
            
        except Exception as e:
            logger.error(f"Error in contract processing: {e}")
            logger.error("This might indicate issues with data format or processing logic")
            return False
        
        # Test 8: Initialize model
        logger.info("Testing model initialization...")
        try:
            model = HeteroToolGNN(config)  # FIXED: Use correct class name
            num_params = sum(p.numel() for p in model.parameters())
            logger.info(f"✓ Model initialized with {num_params:,} parameters")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            return False
        
        # Test 9: Test model forward pass
        logger.info("Testing model forward pass...")
        try:
            # Import torch here to avoid early import issues
            import torch
            
            # Move model and data to same device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            hetero_data = hetero_data.to(device)
            
            with torch.no_grad():
                predictions = model(hetero_data)
            
            if isinstance(predictions, tuple):
                main_pred, tool_pred = predictions
                logger.info(f"✓ Model forward pass successful")
                logger.info(f"  Main predictions shape: {main_pred.shape}")
                logger.info(f"  Tool predictions keys: {list(tool_pred.keys()) if isinstance(tool_pred, dict) else 'Not dict'}")
            else:
                logger.info(f"✓ Model forward pass successful")
                logger.info(f"  Predictions shape: {predictions.shape}")
            
        except Exception as e:
            logger.error(f"Model forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test 10: Test data loading consistency
        logger.info("Testing data consistency...")
        
        # Check that all contracts have corresponding tool results
        contracts_with_results = 0
        for contract in contracts:
            if contract['id'] in tool_results:
                contracts_with_results += 1
        
        logger.info(f"  Contracts with tool results: {contracts_with_results}/{len(contracts)}")
        
        if contracts_with_results == 0:
            logger.warning("No contracts have tool results! Check data loading logic.")
        
        # Check tool coverage
        tool_coverage = {}
        for contract_id, results in tool_results.items():
            for tool_name in results:
                if tool_name not in tool_coverage:
                    tool_coverage[tool_name] = 0
                tool_coverage[tool_name] += 1
        
        logger.info("  Tool coverage:")
        for tool, count in sorted(tool_coverage.items()):
            logger.info(f"    {tool}: {count} contracts")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    logger.info("=== Data Pipeline Test ===")
    
    # Check if config file exists
    if not os.path.exists("config/config.yaml"):
        logger.error("Config file not found!")
        logger.error("Please create config/config.yaml with proper settings.")
        logger.error("You can use the fixed configuration from the analysis.")
        return
    
    # Check if SolidiFI benchmark exists
    try:
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        solidifi_path = config.get('data', {}).get('solidifi_path', './SolidiFI-benchmark')
        if not os.path.exists(solidifi_path):
            logger.error(f"SolidiFI benchmark not found at: {solidifi_path}")
            logger.error("Please ensure the SolidiFI-benchmark directory exists and contains data.")
            return
    except Exception as e:
        logger.error(f"Error reading config: {e}")
        return
    
    if test_data_pipeline():
        logger.info("All tests passed! Your pipeline should work for training.")
        logger.info("\nYou can now run training with:")
        logger.info("python experiments/train.py --config config/config.yaml")
        
        logger.info("\nNext steps:")
        logger.info("1. If you want to use a smaller batch size for debugging, edit config/config.yaml")
        logger.info("2. Consider starting with CPU training first: set mixed_precision: false in config")
        logger.info("3. Monitor the first few epochs to ensure loss is decreasing")
        
    else:
        logger.error("Some tests failed. Please fix the issues before training.")
        logger.error("\nCommon issues to check:")
        logger.error("1. SolidiFI-benchmark directory exists and contains results/")
        logger.error("2. Config file has correct paths")
        logger.error("3. All required Python packages are installed")
        sys.exit(1)

if __name__ == "__main__":
    main()