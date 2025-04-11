"""
Improved Transaction Categorizer Module

This module provides enhanced transaction categorization functionality with
multiple model support (RandomForest, LLM) and better data handling.
"""

import os
import pandas as pd
import numpy as np
import joblib
import glob
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import pickle

from transaction_categorizer import TransactionCategorizer
from model_training import train_model as train_rf_model
from model_training import find_latest_model_file
from llm_service import client, get_llama_category, get_llama_categories_batch, test_llama_connection
from config import (
    MODELS_DIR, 
    TO_CATEGORIZE_DIR, 
    OUTPUT_DIR,
    PREDEFINED_CATEGORIES,
    INFERENCE_API_KEY_ENV_VAR
)

# Export client for other modules
__all__ = ['train_model', 'categorize_transactions', 'find_latest_model_file', 
           'evaluate_model_on_holdout', 'client', 'llm_available']

# Global LLM client check
llm_available = False
try:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get(INFERENCE_API_KEY_ENV_VAR)
    if api_key and client is not None:
        llm_available = test_llama_connection(max_retries=1)
        if llm_available:
            print("LLM API connection successful - Llama model available.")
        else:
            print("LLM API connection failed - Llama model unavailable.")
    else:
        print(f"Warning: No {INFERENCE_API_KEY_ENV_VAR} environment variable found or client not initialized for LLM API.")
except Exception as e:
    print(f"Error initializing LLM client: {e}")

def train_model(test_size=0.2, random_state=42):
    """Train a RandomForest model and return it with metrics.
    
    Args:
        test_size: Percentage of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (model, feature_prep, model_version, model_filename, metrics)
    """
    try:
        model, feature_prep, model_version, model_filename = train_rf_model(test_size, random_state)
        
        if model is None:
            return None, None, None, None, {"error": "Model training failed. See logs for details."}
            
        # Calculate additional metrics if needed
        metrics = {
            "accuracy": 0.0,  # Placeholder - would be calculated during training
            "f1_macro": 0.0,  # Placeholder - would be calculated during training
            "training_time": datetime.now().isoformat()
        }
        
        return model, feature_prep, model_version, model_filename, metrics
    except Exception as e:
        print(f"Error in train_model: {e}")
        return None, None, None, None, {"error": str(e)}

def categorize_transactions(input_dir=TO_CATEGORIZE_DIR, output_dir=OUTPUT_DIR, model_type="local"):
    """Categorize all CSV files in the input directory.
    
    Args:
        input_dir: Directory containing CSV files to categorize
        output_dir: Directory to save categorized CSV files
        model_type: Type of model to use ('local', 'llama', or 'hybrid')
        
    Returns:
        tuple: (dict of dataframes with categorized transactions, model version, model filename)
    """
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist")
        return None, None, None
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # Initialize the appropriate categorizer based on model_type
    categorizer = None
    model_version = None
    model_filename = None
    
    if model_type == "local":
        # Initialize RandomForest categorizer
        categorizer = TransactionCategorizer()
        model_version = "local_v1"  # Placeholder - would come from the model
        model_filename = "random_forest_model"  # Placeholder - would come from the model
    elif model_type == "llama":
        # Use LLM API for categorization
        if not llm_available:
            print(f"Cannot use LLM model: API client not initialized or connection failed")
            return None, None, None
        model_version = "llama_3.1"
        model_filename = "llama_3.1_api"
    elif model_type == "hybrid":
        # Use both models
        categorizer = TransactionCategorizer()
        if not llm_available:
            print(f"Cannot use hybrid model: LLM API client not initialized or connection failed")
            return None, None, None
        model_version = "hybrid_v1"
        model_filename = "rf_llama_hybrid"
    else:
        print(f"Unsupported model type: {model_type}")
        return None, None, None
        
    # Get all CSV files in the input directory
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return None, None, None
        
    # Process each file
    results = {}
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...")
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Standardize column names to lowercase
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            
            # Ensure required columns exist
            required_cols = ['description', 'amount']
            if not all(col in df.columns for col in required_cols):
                print(f"File {filename} missing required columns: {required_cols}")
                continue
                
            # Add a source_file column
            df['source_file'] = filename
            
            # Categorize transactions based on model_type
            if model_type == "local":
                # Use RandomForest model
                categorized_df = df.copy()
                categorized_df['category'] = None
                categorized_df['confidence'] = None
                
                for idx, row in df.iterrows():
                    result = categorizer.categorize(row['description'], row['amount'])
                    categorized_df.loc[idx, 'category'] = result['category']
                    categorized_df.loc[idx, 'confidence'] = result['confidence']
                    
            elif model_type == "llama":
                # Use LLM API
                categorized_df = df.copy()
                categorized_df['category'] = None
                categorized_df['confidence'] = None
                
                # Prepare transactions for batch processing
                transactions = [(row['description'], 
                                row.get('extended_details') if 'extended_details' in row else None) 
                               for _, row in df.iterrows()]
                
                # Process in batches using the LLM service
                results_list = get_llama_categories_batch(transactions)
                
                # Update the DataFrame with results
                for idx, (category, confidence) in enumerate(results_list):
                    if idx < len(categorized_df):
                        categorized_df.loc[idx, 'category'] = category
                        categorized_df.loc[idx, 'confidence'] = confidence
                
            elif model_type == "hybrid":
                # Use a hybrid approach - RF first, then LLM for low confidence items
                categorized_df = df.copy()
                categorized_df['category'] = None
                categorized_df['confidence'] = None
                categorized_df['model_used'] = None
                
                # First use RandomForest for all
                transactions_for_llm = []
                rf_indices = []
                
                for idx, row in df.iterrows():
                    result = categorizer.categorize(row['description'], row['amount'])
                    categorized_df.loc[idx, 'category'] = result['category'] 
                    categorized_df.loc[idx, 'confidence'] = result['confidence']
                    categorized_df.loc[idx, 'model_used'] = 'rf'
                    
                    # If confidence is below threshold, add to LLM batch
                    if result['confidence'] < 0.7:  # Threshold for LLM backup
                        transactions_for_llm.append((row['description'], 
                                                   row.get('extended_details') if 'extended_details' in row else None))
                        rf_indices.append(idx)
                
                # Process low-confidence items with LLM
                if transactions_for_llm:
                    llm_results = get_llama_categories_batch(transactions_for_llm)
                    
                    # Update only those rows where LLM was needed
                    for batch_idx, (category, confidence) in enumerate(llm_results):
                        df_idx = rf_indices[batch_idx]
                        # Only use LLM result if confidence is higher than RF
                        if confidence > categorized_df.loc[df_idx, 'confidence']:
                            categorized_df.loc[df_idx, 'category'] = category
                            categorized_df.loc[df_idx, 'confidence'] = confidence
                            categorized_df.loc[df_idx, 'model_used'] = 'llama'
                
            # Store the categorized DataFrame in results
            results[filename] = categorized_df
            
            # Save the categorized DataFrame to the output directory
            output_path = os.path.join(output_dir, f"categorized_{filename}")
            categorized_df.to_csv(output_path, index=False)
            print(f"Saved categorized file to {output_path}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
            
    return results, model_version, model_filename

def evaluate_model_on_holdout(model_type="local"):
    """Evaluate a model on the holdout dataset.
    
    Args:
        model_type: Type of model to evaluate ('local' or 'llama')
        
    Returns:
        dict: Evaluation metrics
    """
    try:
        holdout_path = os.path.join('data', 'holdout_set.csv')
        if not os.path.exists(holdout_path):
            return {"error": "Holdout set does not exist"}
            
        # Load holdout data
        holdout_df = pd.read_csv(holdout_path)
        
        # Ensure required columns exist
        required_cols = ['description', 'amount', 'category']
        if not all(col in holdout_df.columns for col in required_cols):
            return {"error": f"Holdout set missing required columns: {required_cols}"}
            
        # Initialize metrics
        metrics = {
            "accuracy": 0.0,
            "model_type": model_type,
            "evaluation_time": datetime.now().isoformat()
        }
        
        # Evaluate based on model type
        if model_type == "local":
            # Evaluate RandomForest model
            categorizer = TransactionCategorizer()
            correct = 0
            total = 0
            
            for idx, row in holdout_df.iterrows():
                result = categorizer.categorize(row['description'], row['amount'])
                if result['category'] == row['category']:
                    correct += 1
                total += 1
            
            if total > 0:
                metrics["accuracy"] = correct / total
            
        elif model_type == "llama":
            # Evaluate LLM model
            if not llm_available:
                return {"error": "LLM API not available for evaluation"}
                
            # Prepare transactions for batch processing
            transactions = [(row['description'], 
                            row.get('extended_details') if 'extended_details' in row else None) 
                           for _, row in holdout_df.iterrows()]
            
            # Get predictions
            llm_results = get_llama_categories_batch(transactions)
            
            # Calculate accuracy
            correct = 0
            for idx, (category, _) in enumerate(llm_results):
                if idx < len(holdout_df) and category == holdout_df.iloc[idx]['category']:
                    correct += 1
            
            if len(holdout_df) > 0:
                metrics["accuracy"] = correct / len(holdout_df)
            
        return metrics
    except Exception as e:
        print(f"Error in evaluate_model_on_holdout: {e}")
        return {"error": str(e)} 