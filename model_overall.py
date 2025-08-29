import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class ModelPerformanceAnalyzer:
    def __init__(self, yaml_file_path: str):
        """
        Initialize the analyzer with YAML data
        
        Args:
            yaml_file_path: Path to the YAML file containing model performance data
        """
        self.data = self.load_yaml_data("./training_plots/results_plot1.yaml")
        self.performance_summary = {}
        
    def load_yaml_data(self, file_path: str) -> Dict[str, Any]:
        """Load YAML data from file"""
        try:
            with open(file_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"File {file_path} not found. Using provided data structure.")
            # If file not found, return the data structure from the document
            return self._get_sample_data()
    
    def _get_sample_data(self) -> Dict[str, Any]:
        """Return the data structure from the provided YAML content"""
        # This would contain the actual data from your YAML file
        # For now, I'll use the structure you provided
        return {
            'metric_mae_data': {
                'tpr': [0.018314456567168236, 0.010372514836490154, 0.007814735174179077],  # truncated for brevity
                'fpr': [0.01170436106622219, 0.011036082170903683, 0.012410159222781658],  # truncated for brevity
                'accuracy': [0.02268458902835846, 0.018487995490431786, 0.01072391401976347],  # truncated for brevity
                'precision': [0.02286575175821781, 0.0190935879945755, 0.013390895910561085]  # truncated for brevity
            },
            'f1_data': {
                'train_f1_mae': [0.03846590220928192, 0.032459478825330734, 0.03287844732403755],  # truncated
                'val_f1_mae': [0.017495516687631607, 0.015711700543761253, 0.013895819894969463]  # truncated
            },
            'tool_mae_data': {
                'oyente': [0.01719435304403305, 0.013273322023451328, 0.007873867638409138],  # truncated
                'securify': [0.011028737761080265, 0.011096913367509842, 0.009210782125592232],  # truncated
                'mythril': [0.02117331326007843, 0.017072875052690506, 0.015765013173222542],  # truncated
                'smartcheck': [0.02094689942896366, 0.014280945993959904, 0.010117602534592152],  # truncated
                'manticore': [0.030733218416571617, 0.021450631320476532, 0.015317238867282867],  # truncated
                'slither': [0.015724975615739822, 0.01562000997364521, 0.014591176062822342]  # truncated
            },
            'current_performance': {
                'metric_names': ['TPR', 'FPR', 'Accuracy', 'Precision'],
                'metric_values': [0.000673765956889838, 6.747245788574219e-05, 0.0006760259275324643, 0.000522911548614502]
            },
            'tool_mape_data': {
                'oyente': [3.43887060880661, 2.6546644046902657, 1.5747735276818275],  # truncated
                'securify': [2.205747552216053, 2.2193826735019684, 1.8421564251184464],  # truncated
                'mythril': [4.234662652015686, 3.414575010538101, 3.1530026346445084],  # truncated
                'smartcheck': [4.189379885792732, 2.8561891987919807, 2.0235205069184303],  # truncated
                'manticore': [6.146643683314323, 4.290126264095306, 3.0634477734565735],  # truncated
                'slither': [3.1449951231479645, 3.124001994729042, 2.9182352125644684]  # truncated
            },
            'tool_rmse_data': {
                'oyente': [0.025992857292294502, 0.017522646114230156, 0.010697475634515285],  # truncated
                'securify': [0.012840324081480503, 0.01129421591758728, 0.010383017361164093],  # truncated
                'mythril': [0.02730989083647728, 0.023044627159833908, 0.021620525047183037],  # truncated
                'smartcheck': [0.029268903657794, 0.020639877766370773, 0.015786172822117805],  # truncated
                'manticore': [0.033626846969127655, 0.024657124653458595, 0.017877336591482162],  # truncated
                'slither': [0.016739612445235252, 0.017956774681806564, 0.017401529476046562]  # truncated
            },
            'final_summary': {
                'summary_names': ['MAPE', 'RMSE'],
                'summary_values': [0.09380479459650815, 0.0006167393294163048]
            }
        }
    
    def calculate_basic_metrics(self) -> Dict[str, float]:
        """Calculate basic performance metrics"""
        metrics = {}
        
        # Current performance metrics
        current_perf = self.data.get('current_performance', {})
        metric_names = current_perf.get('metric_names', [])
        metric_values = current_perf.get('metric_values', [])
        
        for name, value in zip(metric_names, metric_values):
            metrics[f"Current_{name}"] = value
        
        # MAE metrics
        mae_data = self.data.get('metric_mae_data', {})
        for metric_name, values in mae_data.items():
            if values:
                metrics[f"MAE_{metric_name}_mean"] = np.mean(values)
                metrics[f"MAE_{metric_name}_std"] = np.std(values)
                metrics[f"MAE_{metric_name}_min"] = np.min(values)
                metrics[f"MAE_{metric_name}_max"] = np.max(values)
                metrics[f"MAE_{metric_name}_final"] = values[-1]
        
        # F1 metrics
        f1_data = self.data.get('f1_data', {})
        for f1_type, values in f1_data.items():
            if values:
                metrics[f"{f1_type}_mean"] = np.mean(values)
                metrics[f"{f1_type}_std"] = np.std(values)
                metrics[f"{f1_type}_final"] = values[-1]
        
        # Final summary
        final_summary = self.data.get('final_summary', {})
        summary_names = final_summary.get('summary_names', [])
        summary_values = final_summary.get('summary_values', [])
        
        for name, value in zip(summary_names, summary_values):
            metrics[f"Final_{name}"] = value
        
        return metrics
    
    def calculate_tool_performance(self) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics for each tool"""
        tool_performance = {}
        
        # Get all tool names
        tools = set()
        for data_type in ['tool_mae_data', 'tool_mape_data', 'tool_rmse_data']:
            if data_type in self.data:
                tools.update(self.data[data_type].keys())
        
        for tool in tools:
            tool_performance[tool] = {}
            
            # MAE metrics
            mae_data = self.data.get('tool_mae_data', {}).get(tool, [])
            if mae_data:
                tool_performance[tool]['MAE_mean'] = np.mean(mae_data)
                tool_performance[tool]['MAE_std'] = np.std(mae_data)
                tool_performance[tool]['MAE_final'] = mae_data[-1]
                tool_performance[tool]['MAE_improvement'] = (mae_data[0] - mae_data[-1]) / mae_data[0] * 100 if mae_data[0] != 0 else 0
            
            # MAPE metrics
            mape_data = self.data.get('tool_mape_data', {}).get(tool, [])
            if mape_data:
                tool_performance[tool]['MAPE_mean'] = np.mean(mape_data)
                tool_performance[tool]['MAPE_std'] = np.std(mape_data)
                tool_performance[tool]['MAPE_final'] = mape_data[-1]
                tool_performance[tool]['MAPE_improvement'] = (mape_data[0] - mape_data[-1]) / mape_data[0] * 100 if mape_data[0] != 0 else 0
            
            # RMSE metrics
            rmse_data = self.data.get('tool_rmse_data', {}).get(tool, [])
            if rmse_data:
                tool_performance[tool]['RMSE_mean'] = np.mean(rmse_data)
                tool_performance[tool]['RMSE_std'] = np.std(rmse_data)
                tool_performance[tool]['RMSE_final'] = rmse_data[-1]
                tool_performance[tool]['RMSE_improvement'] = (rmse_data[0] - rmse_data[-1]) / rmse_data[0] * 100 if rmse_data[0] != 0 else 0
        
        return tool_performance
    
    def calculate_convergence_metrics(self) -> Dict[str, float]:
        """Calculate convergence and stability metrics"""
        convergence_metrics = {}
        
        # Analyze MAE convergence
        mae_data = self.data.get('metric_mae_data', {})
        for metric_name, values in mae_data.items():
            if len(values) >= 10:  # Need sufficient data points
                # Calculate rate of convergence (slope of last 20% of data)
                tail_length = max(10, len(values) // 5)
                tail_values = values[-tail_length:]
                x = np.arange(len(tail_values))
                slope = np.polyfit(x, tail_values, 1)[0]
                convergence_metrics[f"{metric_name}_convergence_rate"] = abs(slope)
                
                # Calculate stability (coefficient of variation in last 20%)
                cv = np.std(tail_values) / np.mean(tail_values) if np.mean(tail_values) != 0 else 0
                convergence_metrics[f"{metric_name}_stability"] = cv
        
        return convergence_metrics
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report"""
        basic_metrics = self.calculate_basic_metrics()
        tool_performance = self.calculate_tool_performance()
        convergence_metrics = self.calculate_convergence_metrics()
        
        report = []
        report.append("=" * 80)
        report.append("MODEL PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Current Performance Summary
        report.append("\n1. CURRENT PERFORMANCE SUMMARY")
        report.append("-" * 40)
        current_metrics = {k: v for k, v in basic_metrics.items() if k.startswith('Current_')}
        for metric, value in current_metrics.items():
            report.append(f"{metric.replace('Current_', '')}: {value:.6f}")
        
        # Final Summary Metrics
        report.append(f"\nFinal MAPE: {basic_metrics.get('Final_MAPE', 'N/A'):.6f}")
        report.append(f"Final RMSE: {basic_metrics.get('Final_RMSE', 'N/A'):.6f}")
        
        # MAE Performance Over Time
        report.append("\n2. MAE PERFORMANCE METRICS")
        report.append("-" * 40)
        mae_metrics = {k: v for k, v in basic_metrics.items() if 'MAE_' in k and not k.startswith('tool_')}
        for metric_type in ['tpr', 'fpr', 'accuracy', 'precision']:
            mean_key = f"MAE_{metric_type}_mean"
            final_key = f"MAE_{metric_type}_final"
            if mean_key in mae_metrics:
                improvement = ((mae_metrics.get(f"MAE_{metric_type}_max", 0) - mae_metrics[final_key]) / 
                              mae_metrics.get(f"MAE_{metric_type}_max", 1) * 100)
                report.append(f"{metric_type.upper()}:")
                report.append(f"  Mean MAE: {mae_metrics[mean_key]:.6f}")
                report.append(f"  Final MAE: {mae_metrics[final_key]:.6f}")
                report.append(f"  Improvement: {improvement:.2f}%")
        
        # F1 Score Analysis
        report.append("\n3. F1 SCORE ANALYSIS")
        report.append("-" * 40)
        train_f1_final = basic_metrics.get('train_f1_mae_final', 0)
        val_f1_final = basic_metrics.get('val_f1_mae_final', 0)
        report.append(f"Training F1 MAE (final): {train_f1_final:.6f}")
        report.append(f"Validation F1 MAE (final): {val_f1_final:.6f}")
        
        if train_f1_final > 0 and val_f1_final > 0:
            overfitting_ratio = val_f1_final / train_f1_final
            if overfitting_ratio < 0.8:
                report.append("Status: Possible underfitting")
            elif overfitting_ratio > 1.2:
                report.append("Status: Possible overfitting")
            else:
                report.append("Status: Good generalization")
        
        # Tool Performance Ranking
        report.append("\n4. TOOL PERFORMANCE RANKING")
        report.append("-" * 40)
        
        # Rank by final MAE
        tool_mae_ranking = []
        for tool, metrics in tool_performance.items():
            if 'MAE_final' in metrics:
                tool_mae_ranking.append((tool, metrics['MAE_final']))
        
        tool_mae_ranking.sort(key=lambda x: x[1])
        
        report.append("Ranking by Final MAE (lower is better):")
        for i, (tool, mae) in enumerate(tool_mae_ranking, 1):
            improvement = tool_performance[tool].get('MAE_improvement', 0)
            report.append(f"{i}. {tool}: {mae:.6f} (Improvement: {improvement:.2f}%)")
        
        # Best and Worst Performers
        if tool_mae_ranking:
            best_tool = tool_mae_ranking[0][0]
            worst_tool = tool_mae_ranking[-1][0]
            report.append(f"\nBest Performing Tool: {best_tool}")
            report.append(f"Worst Performing Tool: {worst_tool}")
        
        # Convergence Analysis
        report.append("\n5. CONVERGENCE ANALYSIS")
        report.append("-" * 40)
        for metric, value in convergence_metrics.items():
            report.append(f"{metric}: {value:.6f}")
        
        # Overall Assessment
        report.append("\n6. OVERALL ASSESSMENT")
        report.append("-" * 40)
        
        # Calculate overall score (lower is better for error metrics)
        final_mape = basic_metrics.get('Final_MAPE', float('inf'))
        final_rmse = basic_metrics.get('Final_RMSE', float('inf'))
        
        if final_mape < 0.1:
            mape_grade = "Excellent"
        elif final_mape < 0.2:
            mape_grade = "Good"
        elif final_mape < 0.5:
            mape_grade = "Fair"
        else:
            mape_grade = "Poor"
        
        if final_rmse < 0.001:
            rmse_grade = "Excellent"
        elif final_rmse < 0.01:
            rmse_grade = "Good"
        elif final_rmse < 0.1:
            rmse_grade = "Fair"
        else:
            rmse_grade = "Poor"
        
        report.append(f"MAPE Performance: {mape_grade} ({final_mape:.6f})")
        report.append(f"RMSE Performance: {rmse_grade} ({final_rmse:.6f})")
        
        # Recommendations
        report.append("\n7. RECOMMENDATIONS")
        report.append("-" * 40)
        
        if final_mape > 0.1:
            report.append("- Consider improving model architecture or hyperparameter tuning")
        
        if val_f1_final > train_f1_final * 1.2:
            report.append("- Address potential overfitting through regularization")
        
        if len(tool_mae_ranking) > 1:
            best_mae = tool_mae_ranking[0][1]
            worst_mae = tool_mae_ranking[-1][1]
            if worst_mae > best_mae * 2:
                report.append(f"- Focus on improving {worst_tool} tool performance")
        
        report.append("- Monitor convergence stability for early stopping")
        
        return "\n".join(report)
    
    def create_visualizations(self):
        """Create performance visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. MAE Trends
        mae_data = self.data.get('metric_mae_data', {})
        ax1 = axes[0, 0]
        for metric_name, values in mae_data.items():
            if values:
                ax1.plot(values, label=metric_name, marker='o', markersize=3)
        ax1.set_title('MAE Trends Over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MAE')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Tool Performance Comparison
        tool_performance = self.calculate_tool_performance()
        ax2 = axes[0, 1]
        tools = list(tool_performance.keys())
        final_maes = [tool_performance[tool].get('MAE_final', 0) for tool in tools]
        bars = ax2.bar(tools, final_maes, color='skyblue', alpha=0.7)
        ax2.set_title('Final MAE by Tool')
        ax2.set_ylabel('Final MAE')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, final_maes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(final_maes)*0.01, 
                    f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 3. F1 Score Comparison
        f1_data = self.data.get('f1_data', {})
        ax3 = axes[0, 2]
        if f1_data:
            train_f1 = f1_data.get('train_f1_mae', [])
            val_f1 = f1_data.get('val_f1_mae', [])
            if train_f1 and val_f1:
                epochs = range(len(train_f1))
                ax3.plot(epochs, train_f1, label='Training F1 MAE', color='blue', alpha=0.7)
                ax3.plot(epochs, val_f1, label='Validation F1 MAE', color='red', alpha=0.7)
                ax3.set_title('F1 Score MAE Trends')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('F1 MAE')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        
        # 4. Tool MAPE Comparison
        mape_data = self.data.get('tool_mape_data', {})
        ax4 = axes[1, 0]
        if mape_data:
            tools = list(mape_data.keys())
            final_mapes = [mape_data[tool][-1] if mape_data[tool] else 0 for tool in tools]
            bars = ax4.bar(tools, final_mapes, color='lightcoral', alpha=0.7)
            ax4.set_title('Final MAPE by Tool')
            ax4.set_ylabel('MAPE (%)')
            ax4.tick_params(axis='x', rotation=45)
        
        # 5. Performance Distribution
        ax5 = axes[1, 1]
        all_final_errors = []
        labels = []
        for tool, perf in tool_performance.items():
            if 'MAE_final' in perf:
                all_final_errors.append(perf['MAE_final'])
                labels.append(tool)
        
        if all_final_errors:
            ax5.boxplot([all_final_errors], labels=['All Tools'])
            ax5.scatter(range(1, len(all_final_errors) + 1), all_final_errors, alpha=0.6)
            ax5.set_title('MAE Distribution Across Tools')
            ax5.set_ylabel('Final MAE')
        
        # 6. Improvement Rates
        ax6 = axes[1, 2]
        improvements = []
        tool_names = []
        for tool, perf in tool_performance.items():
            if 'MAE_improvement' in perf:
                improvements.append(perf['MAE_improvement'])
                tool_names.append(tool)
        
        if improvements:
            colors = ['green' if imp > 0 else 'red' for imp in improvements]
            bars = ax6.bar(tool_names, improvements, color=colors, alpha=0.7)
            ax6.set_title('MAE Improvement Rate by Tool (%)')
            ax6.set_ylabel('Improvement (%)')
            ax6.tick_params(axis='x', rotation=45)
            ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def export_results(self, filename: str = 'performance_analysis.xlsx'):
        """Export detailed results to Excel"""
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Basic metrics
            basic_metrics = self.calculate_basic_metrics()
            basic_df = pd.DataFrame(list(basic_metrics.items()), columns=['Metric', 'Value'])
            basic_df.to_excel(writer, sheet_name='Basic_Metrics', index=False)
            
            # Tool performance
            tool_performance = self.calculate_tool_performance()
            tool_df = pd.DataFrame(tool_performance).T
            tool_df.to_excel(writer, sheet_name='Tool_Performance')
            
            # Raw data
            for data_type, data in self.data.items():
                if isinstance(data, dict) and data_type != 'current_performance':
                    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
                    sheet_name = data_type.replace('_', ' ').title()[:31]  # Excel sheet name limit
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"Results exported to {filename}")

# Usage example
def main():
    # Initialize analyzer
    analyzer = ModelPerformanceAnalyzer('results_plot1.yaml')
    
    # Generate comprehensive report
    report = analyzer.generate_performance_report()
    print(report)
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Export results
    analyzer.export_results()
    
    # Calculate and display summary statistics
    basic_metrics = analyzer.calculate_basic_metrics()
    tool_performance = analyzer.calculate_tool_performance()
    
    print("\n" + "="*50)
    print("QUICK SUMMARY")
    print("="*50)
    print(f"Overall MAPE: {basic_metrics.get('Final_MAPE', 'N/A')}")
    print(f"Overall RMSE: {basic_metrics.get('Final_RMSE', 'N/A')}")
    print(f"Best Tool: {min(tool_performance.items(), key=lambda x: x[1].get('MAE_final', float('inf')))[0] if tool_performance else 'N/A'}")
    print(f"Current TPR: {basic_metrics.get('Current_TPR', 'N/A')}")
    print(f"Current FPR: {basic_metrics.get('Current_FPR', 'N/A')}")

if __name__ == "__main__":
    main()