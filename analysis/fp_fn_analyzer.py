from typing import Dict
from data.loader import SolidiFIDataLoader


class FalsePositiveFalseNegativeAnalyzer:
    """Analyzer to generate tables matching reference format"""
    
    def __init__(self, data_loader: SolidiFIDataLoader):
        self.data_loader = data_loader
        self.tool_names = ['Oyente', 'Securify', 'Mythril', 'SmartCheck', 'Manticore', 'Slither']
        self.vulnerability_types = [
            'reentrancy', 'timestamp_dep', 'unchecked_send', 
            'unhandled_exp', 'tod', 'integer_flow', 'tx_origin'
        ]
    
    def generate_false_negative_table(self) -> Dict:
        """Generate Table 4: False negatives for each tool"""
        
        ground_truth = self.data_loader.extract_ground_truth_vulnerabilities()
        tool_results = self.data_loader.load_tool_results()
        line_accuracy = self.data_loader.analyze_line_number_accuracy(tool_results, ground_truth) 
        
        fn_table = {}
        
        for vuln_type in self.vulnerability_types:
            fn_table[vuln_type] = {}
            
            # Get all contracts with this vulnerability type
            vuln_contracts = [
                cid for cid, gt in ground_truth.items() 
                if gt['injected_vulnerability'] == vuln_type
            ]
            
            for tool in self.tool_names:
                total_injected = len(vuln_contracts)
                detected_count = 0
                incorrect_line_count = 0
                unreported_count = 0
                
                for contract_id in vuln_contracts:
                    tool_detected = self._tool_detected_vulnerability(
                        contract_id, tool, vuln_type, tool_results
                    )
                    
                    if tool_detected:
                        detected_count += 1
                        # Check line accuracy
                        if contract_id in line_accuracy[tool]:
                            if not line_accuracy[tool][contract_id]['line_accurate']:
                                incorrect_line_count += 1
                    else:
                        unreported_count += 1
                
                false_negatives = total_injected - detected_count
                
                fn_table[vuln_type][tool] = {
                    'total_injected': total_injected,
                    'detected': detected_count,
                    'false_negatives': false_negatives,
                    'incorrect_line': incorrect_line_count,
                    'unreported': unreported_count
                }
        
        return fn_table
    
    def generate_false_positive_table(self) -> Dict:
        """Generate Table 5: False positives reported by each tool"""
        
        ground_truth = self.data_loader.extract_ground_truth_vulnerabilities()
        tool_results = self.data_loader.load_tool_results()
        
        fp_table = {}
        
        for vuln_type in self.vulnerability_types:
            fp_table[vuln_type] = {}
            
            for tool in self.tool_names:
                reported_count = 0
                false_positive_count = 0
                false_line_count = 0
                
                # Analyze all contracts
                for contract_id in ground_truth:
                    actual_vuln = ground_truth[contract_id]['injected_vulnerability']
                    
                    detections = self._get_tool_detections_for_vuln(
                        contract_id, tool, vuln_type, tool_results
                    )
                    
                    reported_count += len(detections)
                    
                    if actual_vuln != vuln_type:
                        # This is a false positive (wrong vulnerability type)
                        false_positive_count += len(detections)
                    else:
                        # Check for line accuracy issues
                        for detection in detections:
                            if not self._is_line_accurate(detection, ground_truth[contract_id]):
                                false_line_count += 1
                
                fp_table[vuln_type][tool] = {
                    'reported': reported_count,
                    'false_line': false_line_count,
                    'false_positive': false_positive_count
                }
        
        return fp_table
    
    def format_tables_for_output(self, fn_table: Dict, fp_table: Dict) -> str:
        """Format tables to match reference format"""
        
        report = "# False Positive and False Negative Analysis\n\n"
        
        # Table 4: False Negatives
        report += "## Table 4: False negatives for each tool\n"
        report += "Numbers within parentheses are bugs with incorrect line numbers or unreported.\n\n"
        
        report += "| Security bug | Injected | Oyente | Securify | Mythril | SmartCheck | Manticore | Slither |\n"
        report += "|--------------|----------|--------|----------|---------|------------|-----------|----------|\n"
        
        for vuln_type in self.vulnerability_types:
            if vuln_type in fn_table:
                row = f"| {vuln_type.replace('_', ' ').title()} | "
                
                # Total injected (from any tool's data)
                total = next(iter(fn_table[vuln_type].values()))['total_injected']
                row += f"{total} | "
                
                for tool in self.tool_names:
                    if tool in fn_table[vuln_type]:
                        data = fn_table[vuln_type][tool]
                        fn_count = data['false_negatives']
                        incorrect_line = data['incorrect_line']
                        unreported = data['unreported']
                        
                        # Format: total_fn (incorrect_line + unreported)
                        row += f"{fn_count} ({incorrect_line}+{unreported}) | "
                    else:
                        row += "NA | "
                
                report += row + "\n"
        
        # Table 5: False Positives  
        report += "\n## Table 5: False positives reported by each tool\n\n"
        
        report += "| Bug Type | Threshold | "
        for tool in self.tool_names:
            report += f"{tool} Reported | FL | FP | "
        report += "\n"
        
        report += "|----------|-----------|"
        for _ in self.tool_names:
            report += "---------|----|----|"
        report += "\n"
        
        for vuln_type in self.vulnerability_types:
            if vuln_type in fp_table:
                row = f"| {vuln_type.replace('_', ' ').title()} | - | "
                
                for tool in self.tool_names:
                    if tool in fp_table[vuln_type]:
                        data = fp_table[vuln_type][tool]
                        reported = data['reported']
                        false_line = data['false_line'] 
                        false_positive = data['false_positive']
                        
                        row += f"{reported} | {false_line} | {false_positive} | "
                    else:
                        row += "- | - | - | "
                
                report += row + "\n"
        
        return report 
    
    
