from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import pickle
import sys
from improved_categorizer import (
    test_llama_connection,
    get_llama_categories_batch,
    find_latest_model_file,
    PREDEFINED_CATEGORIES
)
from db_connector import get_engine

# Load environment variables from .env file
load_dotenv()

def load_data():
    """Load transaction data from the database"""
    try:
        engine = get_engine()
        data = pd.read_sql('SELECT * FROM transactions', engine)
        print(f"Loaded {len(data)} records from the database.")
        return data
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return None

def load_rf_model():
    """Load the latest Random Forest model"""
    try:
        model_filename = find_latest_model_file()
        if not model_filename:
            print("No model file found")
            return None, None
            
        print(f"Loading RF model from {model_filename}")
        
        with open(model_filename, 'rb') as f:
            model_data = pickle.load(f)
            
        model = model_data.get('model')
        feature_prep = model_data.get('feature_prep')
        categories = model_data.get('categories', [])
        
        if model and feature_prep:
            print(f"Model loaded successfully. Supports {len(categories)} categories.")
            return model, feature_prep
        else:
            print("Model loading failed: required components missing")
            return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def compare_models(data, num_transactions=200):
    """Compare Random Forest and Llama models on a subset of transactions"""
    # First make sure llama is available
    if not test_llama_connection():
        print("Llama connection failed. Cannot perform comparison.")
        return
    
    # Prepare data
    X_features = data[['description', 'amount']]
    y = data['category']
    
    # Create a consistent holdout set
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Take only the first num_transactions
    if len(X_test) > num_transactions:
        X_test = X_test.iloc[:num_transactions]
        y_test = y_test.iloc[:num_transactions]
    
    print(f"Using holdout set of {len(X_test)} transactions for comparison")
    
    # Prepare results dataframe
    results_df = pd.DataFrame({
        'description': X_test['description'].reset_index(drop=True),
        'amount': X_test['amount'].reset_index(drop=True),
        'true_category': y_test.reset_index(drop=True)
    })
    
    # Flag to track if RF prediction was successful
    rf_success = False
    
    # 1. Random Forest prediction
    print("\n--- Random Forest Prediction ---")
    rf_model, feature_prep = load_rf_model()
    if rf_model and feature_prep:
        try:
            start_time = time.time()
            
            # Transform features and predict
            test_data_rf = X_test.rename(columns={'description': 'Description', 'amount': 'Amount'})
            X_transformed = feature_prep.transform(test_data_rf)
            rf_predictions = rf_model.predict(X_transformed)
            
            rf_time = time.time() - start_time
            print(f"Random Forest prediction completed in {rf_time:.2f} seconds")
            
            # Add predictions to results
            results_df['rf_prediction'] = rf_predictions
            rf_success = True
        except Exception as e:
            print(f"Error during Random Forest prediction: {e}")
            # Add a dummy column with default values to avoid KeyError later
            results_df['rf_prediction'] = "Unknown"
    else:
        print("Random Forest prediction skipped due to model loading error")
        # Add a dummy column with default values to avoid KeyError later
        results_df['rf_prediction'] = "Unknown"
    
    # 2. Llama prediction
    print("\n--- Llama Model Prediction ---")
    start_time = time.time()
    
    # Prepare batches of (description, extended_details) tuples
    batch_size = 50
    llama_predictions = []
    
    for i in range(0, len(X_test), batch_size):
        batch_end = min(i + batch_size, len(X_test))
        print(f"Processing batch {i//batch_size + 1}/{(len(X_test) + batch_size - 1)//batch_size} ({batch_end - i} transactions)")
        
        batch_details = []
        for idx in range(i, batch_end):
            description = X_test.iloc[idx]['description']
            extended_details = X_test.iloc[idx].get('extended_details', None)
            batch_details.append((description, extended_details))
        
        # Get categories for the batch
        batch_categories = get_llama_categories_batch(batch_details, batch_size=len(batch_details))
        llama_predictions.extend(batch_categories)
        
        # Add a delay between batches
        if batch_end < len(X_test):
            print("  Waiting 1 second between batches...")
            time.sleep(1)
    
    llama_time = time.time() - start_time
    print(f"Llama prediction completed in {llama_time:.2f} seconds")
    
    # Add Llama predictions to results
    results_df['llama_prediction'] = llama_predictions
    
    # 3. Analyze results
    print("\n--- Results Analysis ---")
    
    # Add match columns
    results_df['llama_correct'] = results_df['llama_prediction'] == results_df['true_category']
    
    if rf_success:
        results_df['rf_correct'] = results_df['rf_prediction'] == results_df['true_category']
        results_df['models_agree'] = results_df['rf_prediction'] == results_df['llama_prediction']
        
        # Calculate combined metrics
        rf_accuracy = results_df['rf_correct'].mean()
        llama_accuracy = results_df['llama_correct'].mean()
        agreement_rate = results_df['models_agree'].mean()
        
        print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
        print(f"Llama Model Accuracy: {llama_accuracy:.4f}")
        print(f"Model Agreement Rate: {agreement_rate:.4f}")
        
        # Detailed comparison
        rf_only_correct = ((results_df['rf_correct']) & (~results_df['llama_correct'])).sum()
        llama_only_correct = ((~results_df['rf_correct']) & (results_df['llama_correct'])).sum()
        both_correct = ((results_df['rf_correct']) & (results_df['llama_correct'])).sum()
        both_incorrect = ((~results_df['rf_correct']) & (~results_df['llama_correct'])).sum()
        
        print(f"\nDetailed Comparison:")
        print(f"  Both models correct: {both_correct} ({both_correct/len(results_df):.1%})")
        print(f"  Only RF correct: {rf_only_correct} ({rf_only_correct/len(results_df):.1%})")
        print(f"  Only Llama correct: {llama_only_correct} ({llama_only_correct/len(results_df):.1%})")
        print(f"  Both models incorrect: {both_incorrect} ({both_incorrect/len(results_df):.1%})")
        
        # Analyze disagreements
        disagreements = results_df[~results_df['models_agree']]
        print(f"\nFound {len(disagreements)} disagreements between models")
        
        # Show sample disagreements
        if len(disagreements) > 0:
            print("\nSample disagreements (up to 10):")
            sample_size = min(10, len(disagreements))
            samples = disagreements.sample(sample_size) if len(disagreements) > sample_size else disagreements
            
            for i, row in samples.iterrows():
                correct_model = ""
                if row['rf_correct'] and not row['llama_correct']:
                    correct_model = "RF CORRECT"
                elif not row['rf_correct'] and row['llama_correct']:
                    correct_model = "LLAMA CORRECT"
                elif not row['rf_correct'] and not row['llama_correct']:
                    correct_model = "BOTH WRONG"
                
                print(f"\n{correct_model}:")
                print(f"  Description: {row['description']}")
                print(f"  True Category: {row['true_category']}")
                print(f"  RF Prediction: {row['rf_prediction']}")
                print(f"  Llama Prediction: {row['llama_prediction']}")
    else:
        # Only Llama metrics if RF failed
        llama_accuracy = results_df['llama_correct'].mean()
        print(f"Llama Model Accuracy: {llama_accuracy:.4f}")
        
        # Show some sample predictions
        print("\nSample Llama predictions (up to 10):")
        sample_size = min(10, len(results_df))
        samples = results_df.sample(sample_size)
        
        for i, row in samples.iterrows():
            correct = "CORRECT" if row['llama_correct'] else "INCORRECT"
            print(f"\n{correct}:")
            print(f"  Description: {row['description']}")
            print(f"  True Category: {row['true_category']}")
            print(f"  Llama Prediction: {row['llama_prediction']}")
    
    # Analyze category distribution to understand why Llama might be underperforming
    print("\n--- Category Distribution Analysis ---")
    true_dist = results_df['true_category'].value_counts()
    llama_dist = results_df['llama_prediction'].value_counts()
    
    print("True Category Distribution:")
    print(true_dist)
    
    print("\nLlama Predicted Category Distribution:")
    print(llama_dist)
    
    # Check if Llama is overusing any particular category
    common_categories = set(true_dist.index) & set(llama_dist.index)
    if common_categories:
        print("\nCategory Usage Comparison:")
        for cat in sorted(common_categories):
            true_count = true_dist.get(cat, 0)
            llama_count = llama_dist.get(cat, 0)
            diff = llama_count - true_count
            print(f"  {cat}: True={true_count}, Llama={llama_count}, Diff={diff:+d}")
            
    # Analyze Llama errors by category
    print("\nLlama Accuracy by True Category:")
    for cat in sorted(true_dist.index):
        cat_subset = results_df[results_df['true_category'] == cat]
        cat_accuracy = cat_subset['llama_correct'].mean()
        count = len(cat_subset)
        print(f"  {cat}: {cat_accuracy:.2f} ({(cat_accuracy * count):.0f}/{count})")
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"model_comparison_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to {output_file}")
    
    return results_df

if __name__ == "__main__":
    print("Starting model comparison...")
    data = load_data()
    
    if data is not None:
        results = compare_models(data, num_transactions=200)
    else:
        print("Cannot continue without data.") 