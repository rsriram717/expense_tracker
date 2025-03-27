import pandas as pd
from pathlib import Path
from models.random_forest_model import RandomForestCategorizer
from models.evaluation.evaluator import ModelEvaluator

def load_categorized_data():
    """Load all categorized data from the data/categorized directory."""
    categorized_dir = Path("data/categorized")
    all_data = []
    
    for file in categorized_dir.glob("*.csv"):
        df = pd.read_csv(file)
        # Keep only relevant columns
        df = df[['Description', 'Amount', 'Extended Details', 'Category']]
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)

def main():
    # Load data
    print("Loading categorized data...")
    data = load_categorized_data()
    print(f"Loaded {len(data)} transactions with {data['Category'].nunique()} unique categories")
    print("\nCategory distribution:")
    print(data['Category'].value_counts())
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Create holdout set if it doesn't exist
    if not Path(evaluator.holdout_path).exists():
        print("\nCreating holdout set...")
        evaluator.create_holdout_set(data)
    
    # Prepare training data (excluding holdout)
    holdout_data = evaluator.load_holdout_set()
    train_data = data[~data.index.isin(holdout_data.index)]
    print(f"\nSplit data into {len(train_data)} training and {len(holdout_data)} holdout transactions")
    
    # Train and evaluate Random Forest model
    print("\nTraining Random Forest model...")
    rf_model = RandomForestCategorizer(n_estimators=100, max_depth=None)
    
    # Prepare features and labels
    X_train = train_data[['Description', 'Amount', 'Extended Details']]
    y_train = train_data['Category']
    
    # Train model
    rf_model.train(X_train, y_train)
    
    # Save model
    rf_model.save("models/random_forest_model.pkl")
    print("Model saved to models/random_forest_model.pkl")
    
    # Evaluate model
    print("\nEvaluating Random Forest model...")
    results = evaluator.evaluate_model(rf_model, "RandomForest_v1")
    
    # Print results
    print("\nModel Evaluation Results:")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"Precision: {results['metrics']['precision']:.4f}")
    print(f"Recall: {results['metrics']['recall']:.4f}")
    print(f"F1 Score: {results['metrics']['f1']:.4f}")
    print(f"Mean Confidence: {results['metrics']['confidence']['mean_confidence']:.4f}")
    print(f"Low Confidence Transactions: {results['metrics']['confidence']['low_confidence_count']}")
    
    print("\nPer-Category Performance:")
    for category, metrics in results['metrics']['per_category'].items():
        print(f"\n{category}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
    
    # Show model history
    print("\nModel History:")
    history = evaluator.get_model_history()
    print(history)

if __name__ == "__main__":
    main() 