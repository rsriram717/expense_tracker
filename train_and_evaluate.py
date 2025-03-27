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
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)

def main():
    # Load data
    print("Loading categorized data...")
    data = load_categorized_data()
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Create holdout set if it doesn't exist
    if not Path(evaluator.holdout_path).exists():
        print("Creating holdout set...")
        evaluator.create_holdout_set(data)
    
    # Prepare training data (excluding holdout)
    holdout_data = evaluator.load_holdout_set()
    train_data = data[~data.index.isin(holdout_data.index)]
    
    # Train and evaluate Random Forest model
    print("Training Random Forest model...")
    rf_model = RandomForestCategorizer(n_estimators=100, max_depth=None)
    
    # Prepare features and labels
    X_train = train_data[['description', 'amount']]
    y_train = train_data['category']
    
    # Train model
    rf_model.train(X_train, y_train)
    
    # Save model
    rf_model.save("models/random_forest_model.pkl")
    
    # Evaluate model
    print("Evaluating Random Forest model...")
    results = evaluator.evaluate_model(rf_model, "RandomForest_v1")
    
    # Print results
    print("\nModel Evaluation Results:")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"Precision: {results['metrics']['precision']:.4f}")
    print(f"Recall: {results['metrics']['recall']:.4f}")
    print(f"F1 Score: {results['metrics']['f1']:.4f}")
    print(f"Mean Confidence: {results['metrics']['confidence']['mean_confidence']:.4f}")
    print(f"Low Confidence Transactions: {results['metrics']['confidence']['low_confidence_count']}")
    
    # Show model history
    print("\nModel History:")
    history = evaluator.get_model_history()
    print(history)

if __name__ == "__main__":
    main() 