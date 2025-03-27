import pandas as pd
import json
from pathlib import Path
from typing import Dict, List

def extract_metrics(results: Dict) -> Dict:
    """Extract key metrics from evaluation results."""
    metrics = {
        'timestamp': results['timestamp'],
        'model_name': results['model_name'],
        'accuracy': results['metrics']['accuracy'],
        'precision': results['metrics']['precision'],
        'recall': results['metrics']['recall'],
        'f1': results['metrics']['f1'],
        'mean_confidence': results['metrics']['confidence']['mean_confidence'],
        'low_confidence_count': results['metrics']['confidence']['low_confidence_count'],
        'holdout_size': results['dataset_info']['holdout_size'],
        'num_categories': results['dataset_info']['num_categories']
    }
    
    # Add per-category metrics
    for category, cat_metrics in results['metrics']['per_category'].items():
        metrics[f'{category}_precision'] = cat_metrics['precision']
        metrics[f'{category}_recall'] = cat_metrics['recall']
        metrics[f'{category}_f1'] = cat_metrics['f1']
        metrics[f'{category}_support'] = cat_metrics['support']
    
    # Add confidence distribution
    for level, count in results['metrics']['confidence']['confidence_distribution'].items():
        metrics[f'confidence_{level}'] = count
    
    return metrics

def main():
    # Get all evaluation files
    history_path = Path("data/model_history")
    evaluation_files = list(history_path.glob("model_evaluation_*.json"))
    
    # Extract metrics from each file
    all_metrics = []
    for file in evaluation_files:
        with open(file, 'r') as f:
            results = json.load(f)
            metrics = extract_metrics(results)
            all_metrics.append(metrics)
    
    # Convert to DataFrame and sort by timestamp
    df = pd.DataFrame(all_metrics)
    df = df.sort_values('timestamp', ascending=False)
    
    # Save to CSV
    output_path = history_path / "model_metrics.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved metrics to {output_path}")

if __name__ == "__main__":
    main() 