import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
from ..base_model import BaseTransactionCategorizer

class ModelEvaluator:
    """Framework for evaluating transaction categorization models."""
    
    def __init__(self, holdout_path: str = "models/holdout/holdout_data.csv"):
        self.holdout_path = holdout_path
        self.history_path = Path("data/model_history")
        self.history_path.mkdir(parents=True, exist_ok=True)
        
    def create_holdout_set(self, data: pd.DataFrame, holdout_size: float = 0.1) -> None:
        """Create and save a holdout set for consistent evaluation."""
        train_data, holdout_data = train_test_split(
            data,
            test_size=holdout_size,
            random_state=42,
            stratify=data['Category']
        )
        # Create the holdout directory if it doesn't exist
        Path(self.holdout_path).parent.mkdir(parents=True, exist_ok=True)
        holdout_data.to_csv(self.holdout_path, index=False)
        
    def load_holdout_set(self) -> pd.DataFrame:
        """Load the holdout set."""
        return pd.read_csv(self.holdout_path)
    
    def evaluate_model(self, model: BaseTransactionCategorizer, model_name: str) -> Dict[str, Any]:
        """Evaluate a model on the holdout set and save results."""
        # Load holdout data
        holdout_data = self.load_holdout_set()
        
        # Prepare features and labels
        X = holdout_data[['Description', 'Amount', 'Extended Details']]
        y_true = holdout_data['Category']
        
        # Get predictions
        y_pred, confidence_scores = model.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Calculate per-category metrics
        categories = sorted(y_true.unique())
        per_category_metrics = {}
        for category in categories:
            cat_precision, cat_recall, cat_f1, _ = precision_recall_fscore_support(
                y_true == category, y_pred == category, average='binary'
            )
            per_category_metrics[category] = {
                'precision': float(cat_precision),
                'recall': float(cat_recall),
                'f1': float(cat_f1),
                'support': int(np.sum(y_true == category))
            }
        
        # Calculate confidence statistics
        confidence_stats = {
            'mean_confidence': float(np.mean(confidence_scores)),
            'std_confidence': float(np.std(confidence_scores)),
            'low_confidence_count': int(np.sum(confidence_scores < 0.7)),
            'confidence_distribution': {
                'very_low': int(np.sum(confidence_scores < 0.3)),
                'low': int(np.sum((confidence_scores >= 0.3) & (confidence_scores < 0.7))),
                'medium': int(np.sum((confidence_scores >= 0.7) & (confidence_scores < 0.9))),
                'high': int(np.sum(confidence_scores >= 0.9))
            }
        }
        
        # Prepare results
        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'model_info': model.get_model_info(),
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'per_category': per_category_metrics,
                'confidence': confidence_stats
            },
            'confusion_matrix': conf_matrix.tolist(),
            'dataset_info': {
                'holdout_size': len(holdout_data),
                'num_categories': len(categories),
                'category_distribution': holdout_data['Category'].value_counts().to_dict()
            }
        }
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to history."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_evaluation_{timestamp}.json"
        filepath = self.history_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    def get_model_history(self) -> pd.DataFrame:
        """Load and combine all historical model evaluations."""
        history_files = list(self.history_path.glob("model_evaluation_*.json"))
        history_data = []
        
        for file in history_files:
            with open(file, 'r') as f:
                results = json.load(f)
                history_data.append({
                    'timestamp': results['timestamp'],
                    'model_name': results['model_name'],
                    'accuracy': results['metrics']['accuracy'],
                    'precision': results['metrics']['precision'],
                    'recall': results['metrics']['recall'],
                    'f1': results['metrics']['f1'],
                    'mean_confidence': results['metrics']['confidence']['mean_confidence'],
                    'low_confidence_count': results['metrics']['confidence']['low_confidence_count'],
                    'holdout_size': results['dataset_info']['holdout_size']
                })
        
        return pd.DataFrame(history_data).sort_values('timestamp', ascending=False) 