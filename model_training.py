"""
ML Model Training Module

This module handles training, evaluation, and management of ML models
for transaction categorization.
"""

import os
import pandas as pd
import numpy as np
import re
import joblib
import glob
from datetime import datetime, timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, Any

from config import MODEL_VERSION_FILE, MODELS_DIR
from db_connector import get_engine, model_versions_table, model_scores_table

# --- Model Feature Processing Classes ---

# Function to preprocess text
def preprocess_text(text):
    if pd.isna(text):
        return ""
    # Convert to lowercase and remove special characters
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text

# Custom transformer for text features
class TextFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            preprocessor=preprocess_text,
            ngram_range=(1, 2),
            max_features=500
        )
        
    def fit(self, X, y=None):
        # Ensure Description exists and handle NaNs
        if 'Description' not in X.columns:
            raise ValueError("Input DataFrame must contain a 'Description' column.")
        self.vectorizer.fit(X['Description'].fillna(''))
        return self
    
    def transform(self, X):
        if 'Description' not in X.columns:
            raise ValueError("Input DataFrame must contain a 'Description' column.")
        return self.vectorizer.transform(X['Description'].fillna(''))

# Custom transformer for numeric features
class NumericFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        # Ensure Amount exists
        if 'Amount' not in X.columns:
            raise ValueError("Input DataFrame must contain an 'Amount' column.")
        amounts = self._extract_amounts(X)
        self.scaler.fit(amounts.reshape(-1, 1))
        return self
    
    def transform(self, X):
        if 'Amount' not in X.columns:
            raise ValueError("Input DataFrame must contain an 'Amount' column.")
        amounts = self._extract_amounts(X)
        return self.scaler.transform(amounts.reshape(-1, 1))
    
    def _extract_amounts(self, X):
        # Convert to numeric and handle any errors
        return np.array(pd.to_numeric(X['Amount'].fillna(0), errors='coerce').fillna(0))

# Class to prepare all features in one step
class FeaturePreparation:
    def __init__(self):
        self.pipeline = Pipeline([
            ('features', FeatureUnion([
                ('text', TextFeatures()),
                ('numeric', NumericFeatures())
            ]))
        ])
        
    def fit_transform(self, X, y=None):
        return self.pipeline.fit_transform(X, y)
    
    def transform(self, X):
        return self.pipeline.transform(X)

def get_next_model_version():
    """Reads the last model version, increments it, and saves it back."""
    version = 0
    try:
        if os.path.exists(MODEL_VERSION_FILE):
            with open(MODEL_VERSION_FILE, 'r') as f:
                version = int(f.read().strip())
    except (ValueError, FileNotFoundError):
        version = 0 # Start from 0 if file invalid or not found

    next_version = version + 1

    try:
        os.makedirs(os.path.dirname(MODEL_VERSION_FILE), exist_ok=True)
        with open(MODEL_VERSION_FILE, 'w') as f:
            f.write(str(next_version))
    except Exception as e:
        print(f"Warning: Could not write new model version to {MODEL_VERSION_FILE}: {e}")

    return next_version

def find_latest_model_file(models_dir=MODELS_DIR, base_name="random_forest_v"):
    """Finds the model file with the highest version number."""
    search_pattern = os.path.join(models_dir, f"{base_name}*.joblib")
    model_files = glob.glob(search_pattern)
    if not model_files:
        return None
    
    # Extract version numbers and find the max
    latest_file = max(model_files, key=lambda f: int(re.search(r'_v(\d+)\.joblib$', f).group(1)))
    return latest_file

def train_model(test_size=0.2, random_state=42):
    """Train a RandomForest model on transaction data.
    
    Args:
        test_size: Percentage of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (model, feature_prep, model_version, model_filename)
    """
    # --- Load Data From Database --- 
    try:
        engine = get_engine()
        # Read all finalized transactions from the database
        data = pd.read_sql('SELECT * FROM transactions', engine)
        print(f"Loaded {len(data)} records from the database.")
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return None, None, None, None

    if data.empty:
        print("No training data found in the database. Please submit some categorized transactions first.")
        return None, None, None, None

    # --- Deduplication --- 
    # Define columns that identify a unique transaction
    identifying_columns = ['transaction_date', 'description', 'amount', 'source_file']
    
    # Check if all identifying columns exist
    missing_id_cols = [col for col in identifying_columns if col not in data.columns]
    if missing_id_cols:
        print(f"Warning: Database table is missing columns needed for deduplication: {missing_id_cols}. Skipping deduplication.")
    elif 'timestamp' not in data.columns:
         print(f"Warning: Database table is missing 'timestamp' column needed for deduplication. Skipping deduplication.")
    else:
        print(f"Deduplicating based on: {identifying_columns} keeping the latest entry.")
        # Ensure timestamp is datetime for sorting
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        # Sort by identifying columns and then by timestamp descending
        data = data.sort_values(by=identifying_columns + ['timestamp'], ascending=[True]*len(identifying_columns) + [False])
        # Keep the first (most recent) entry for each unique transaction
        initial_count = len(data)
        data = data.drop_duplicates(subset=identifying_columns, keep='first')
        deduplicated_count = len(data)
        print(f"Deduplication complete. Kept {deduplicated_count} unique transactions (removed {initial_count - deduplicated_count}).")

    # --- Data Preparation --- 
    # Check required columns after loading and deduplication
    required_cols = ['description', 'amount', 'category']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        print(f"Error: Training data from database is missing required columns: {missing_cols}")
        print(f"Available columns: {list(data.columns)}")
        return None, None, None, None

    # Basic stats about the dataset
    print(f"Using {len(data)} unique transactions for training.")
    if 'category' in data.columns:
        category_counts = data['category'].value_counts()
        print(f"Category distribution ({len(category_counts)} unique categories):")
        for category, count in category_counts.items():
            print(f"  {category}: {count}")
    else:
        print("Warning: 'category' column not found for distribution stats.")
        return None, None, None, None # Cannot train without category

    # Prepare features
    try:
        feature_prep = FeaturePreparation()
        # Pass standardized columns expected by transformers
        # Ensure the columns exist before renaming
        X_features = data[['description', 'amount']].rename(columns={'description': 'Description', 'amount': 'Amount'})
        X = feature_prep.fit_transform(X_features)
        y = data['category']
    except ValueError as e:
        print(f"Error during feature preparation: {e}")
        return None, None, None, None
    except Exception as e:
        print(f"Unexpected error during feature preparation: {e}")
        return None, None, None, None

    # Split data for validation
    try:
        # Ensure there's enough data to split
        if len(data) < 2 or len(data['category'].unique()) < 2:
             print("Warning: Not enough data or categories to perform train/test split. Training on full dataset.")
             X_train, y_train = X, y
             X_test, y_test = X, y # Evaluate on training data
        else:
             X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
             )
    except ValueError as e:
        # Common issue if a category has only one sample
        print(f"Error during train/test split (potentially due to low sample count per category): {e}")
        print("Attempting training on full dataset...")
        X_train, y_train = X, y # Train on full data if split fails
        X_test, y_test = X, y # Evaluate on training data (not ideal, but better than crashing)

    # --- Train Model --- 
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)

    # --- Evaluate --- 
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")

    # Only print classification report if test set is valid (different from train set)
    if X_test.shape[0] > 0 and (not hasattr(X_train, 'indices') or not np.array_equal(X_train.indices, X_test.indices) or not np.array_equal(y_train.values, y_test.values)):
         print("Classification report:")
         # Ensure target_names matches the unique classes in y_test for the report
         report_classes = sorted(y_test.unique().tolist())
         print(classification_report(y_test, y_pred, labels=report_classes, zero_division=0))
    else:
         print("Classification report skipped (evaluated on training data or test set empty).")

    # --- Model Saving with Versioning ---
    base_name = "random_forest"
    v_num = get_next_model_version()
    model_version = f"{base_name}_v{v_num}"
    model_filename = os.path.join(MODELS_DIR, f"{model_version}.joblib")

    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save model and feature preparation pipeline together
    model_data = {
        'model': model,
        'feature_prep': feature_prep,
        'categories': list(model.classes_),
        'model_version': model_version # Store version inside the file too
    }

    # Save using joblib
    try:
        joblib.dump(model_data, model_filename)
        print(f"Model saved to {model_filename}")
    except Exception as e:
        print(f"Error saving model file {model_filename}: {e}")
        return None, None, None, None # Failed to save

    # Return model, feature_prep, version string, and filename
    return model, feature_prep, model_version, model_filename

def load_model(model_filename=None):
    """Load a trained model from file.
    
    Args:
        model_filename: Path to the model file. If None, loads the latest model.
        
    Returns:
        dict: Dictionary containing model, feature_prep, and categories
    """
    try:
        if model_filename is None:
            model_filename = find_latest_model_file()
            if not model_filename:
                print("No model file found")
                return None
        
        print(f"Loading model from {model_filename}")
        
        # Load using joblib
        model_data = joblib.load(model_filename)
            
        # Validate model data
        required_keys = ['model', 'feature_prep', 'categories']
        missing_keys = [key for key in required_keys if key not in model_data]
        
        if missing_keys:
            print(f"Model file is missing required components: {missing_keys}")
            return None
            
        print(f"Model loaded successfully. Supports {len(model_data['categories'])} categories.")
        return model_data
    except Exception as e:
        print(f"Error loading model: {repr(e)}") # Reverted to simpler exception printing
        return None

def store_evaluation_results(metrics: Dict[str, Any], model_info: Dict[str, Any], holdout_size: int):
    """Stores model evaluation results (from any model type) in the database."""
    model_version = metrics.get('model_name', 'unknown_model')
    model_filename = model_info.get('model_filename', 'N/A') # Get filename from model_info if available
    print(f"Storing evaluation results for model: {model_version}")

    # Print key metrics to console for visibility
    print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
    print(f"  F1 (Weighted): {metrics.get('f1_weighted', 'N/A'):.4f}")
    print(f"  Avg Confidence: {metrics.get('avg_confidence', 'N/A'):.4f}")

    engine = get_engine()
    if not model_version or not metrics:
        print("Error: Missing model version or metrics for DB storage.")
        return False

    # Prepare version data
    version_data = {
        'model_version': model_version,
        'model_filename': model_filename,
        # Use evaluation timestamp as the primary timestamp for this record
        'training_timestamp': datetime.fromisoformat(metrics['evaluation_timestamp']) if 'evaluation_timestamp' in metrics else datetime.now(timezone.utc),
        'training_dataset_size': None, # We don't know training size from evaluation context
        'holdout_set_size': holdout_size
    }

    # Prepare scores data
    scores_data = []
    eval_timestamp = datetime.fromisoformat(metrics['evaluation_timestamp'])

    # Metrics to store in the database (match keys from evaluate_model metrics dict)
    score_metrics_keys = [
        'accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted',
        'avg_confidence', 'std_confidence'
    ]

    for metric_name in score_metrics_keys:
        if metric_name in metrics and pd.notna(metrics[metric_name]):
            scores_data.append({
                # 'model_version_id' will be added after inserting the version
                'evaluation_timestamp': eval_timestamp,
                'metric_name': metric_name,
                'metric_value': metrics[metric_name]
            })

    # Add processing time as a metric
    if 'processing_time_seconds' in metrics and pd.notna(metrics['processing_time_seconds']):
         scores_data.append({
                'evaluation_timestamp': eval_timestamp,
                'metric_name': 'processing_time_seconds',
                'metric_value': metrics['processing_time_seconds']
            })

    if not scores_data:
        print("Warning: No valid metrics found to store in model_scores.")
        # Decide if you still want to store the version record
        # return False # Option: Abort if no scores

    # Use a transaction to insert version and scores
    with engine.begin() as connection:
        try:
            # Check if model_version already exists
            select_stmt = model_versions_table.select().where(model_versions_table.c.model_version == model_version)
            existing_version = connection.execute(select_stmt).fetchone()

            if existing_version:
                model_version_id = existing_version.id
                print(f"Model version '{model_version}' already exists with ID: {model_version_id}. Updating scores.")
                # Optional: Update timestamp or other fields if desired
                # update_stmt = model_versions_table.update().where(model_versions_table.c.id == model_version_id).values(last_evaluated=eval_timestamp)
                # connection.execute(update_stmt)

                # Delete old scores for this model version ID before inserting new ones
                delete_scores_stmt = model_scores_table.delete().where(model_scores_table.c.model_version_id == model_version_id)
                connection.execute(delete_scores_stmt)
                print(f"Deleted old scores for model version ID {model_version_id}.")

            else:
                # Insert new model version record
                insert_stmt = model_versions_table.insert().values(version_data)
                result = connection.execute(insert_stmt)
                model_version_id = result.inserted_primary_key[0]
                print(f"Stored new model version: {model_version} with ID: {model_version_id}")

            # Insert new model scores records
            if scores_data and model_version_id:
                for score in scores_data:
                    score['model_version_id'] = model_version_id
                connection.execute(model_scores_table.insert(), scores_data)
                print(f"Stored {len(scores_data)} evaluation metrics for model ID {model_version_id}.")
            else:
                 print(f"No new scores were stored for model ID {model_version_id}.")

            return True # Indicate success

        except Exception as e:
            print(f"Database Error storing evaluation results: {e}")
            print("Evaluation metrics NOT stored in database.")
            # Transaction automatically rolls back on exception with engine.begin()
            return False # Indicate failure

if __name__ == "__main__":
    print("ML Model Training")
    print("=================")
    model, feature_prep, model_version, model_filename = train_model()
    
    if model:
        print(f"Successfully trained model: {model_version}")
    else:
        print("Model training failed.") 