"""
Transaction Categorizer Module

This module provides a unified interface for categorizing financial transactions
using various methods (ML, LLM, hybrid approach, merchant rules).
"""

import os
import pandas as pd
import numpy as np
import time
import pickle
from typing import Dict, List, Tuple, Optional, Any, Union

from config import (
    MODELS_DIR, 
    TO_CATEGORIZE_DIR, 
    OUTPUT_DIR,
    RF_CONFIDENCE_THRESHOLD,
    PREDEFINED_CATEGORIES
)
import model_training
from llm_service import (
    test_llama_connection,
    process_file_with_llama,
    get_llama_category,
    client
)
from merchant_postprocessor import MerchantPostProcessor

class TransactionCategorizer:
    """A unified transaction categorizer that supports multiple categorization methods"""
    
    def __init__(self, rf_confidence_threshold=RF_CONFIDENCE_THRESHOLD):
        """Initialize the categorizer
        
        Args:
            rf_confidence_threshold: Threshold for using RF prediction (0.0-1.0)
        """
        self.rf_model = None
        self.feature_prep = None
        self.rf_categories = []
        self.rf_confidence_threshold = rf_confidence_threshold
        self.merchant_category_map = {}  # Cache for merchant->category mappings
        
        # Load the ML model
        self._load_model()
        
        # Check if LLM is available
        self.llm_available = test_llama_connection()
        
    def _load_model(self):
        """Load the latest model file"""
        try:
            # model_data = load_model() # Old call
            model_data = model_training.load_model() # New explicit call
        except Exception as e:
            print(f"Error during initial model load: {e}")
            model_data = None # Ensure model_data is None on error

        if model_data:
            self.rf_model = model_data.get('model')
            self.feature_prep = model_data.get('feature_prep')
            self.rf_categories = model_data.get('categories', [])
            return True
        return False
            
    def _get_rf_prediction(self, description: str, amount: float) -> Tuple[Optional[str], float]:
        """Get prediction from Random Forest model with confidence
        
        Args:
            description: Transaction description
            amount: Transaction amount
            
        Returns:
            tuple: (predicted_category, confidence)
        """
        if not self.rf_model or not self.feature_prep:
            return None, 0.0
            
        try:
            # Prepare data for RF
            data = pd.DataFrame({
                'Description': [description],
                'Amount': [amount]
            })
            
            # Transform features
            X_transformed = self.feature_prep.transform(data)
            
            # Get prediction and probabilities
            prediction = self.rf_model.predict(X_transformed)[0]
            probabilities = self.rf_model.predict_proba(X_transformed)[0]
            
            # Get confidence (probability of predicted class)
            predicted_idx = list(self.rf_model.classes_).index(prediction)
            confidence = probabilities[predicted_idx]
            
            return prediction, confidence
        except Exception as e:
            print(f"Error in RF prediction: {e}")
            return None, 0.0
            
    def categorize(self, description: str, amount: float, extended_details: Optional[str] = None) -> Dict[str, Any]:
        """Categorize a transaction using the hybrid approach
        
        Args:
            description: Transaction description
            amount: Transaction amount
            extended_details: Optional additional details
            
        Returns:
            dict: {
                'category': predicted category,
                'source': 'rf', 'llm', 'cache' or 'fallback',
                'confidence': confidence score (0.0-1.0),
                'rf_prediction': RF model prediction,
                'rf_confidence': RF confidence score,
                'llm_prediction': LLM model prediction
            }
        """
        result = {
            'category': None,
            'source': None,
            'confidence': 0.0,
            'rf_prediction': None,
            'rf_confidence': 0.0,
            'llm_prediction': None
        }
        
        # Check merchant cache first
        clean_description = description.strip().upper()
        if clean_description in self.merchant_category_map:
            cached_category = self.merchant_category_map[clean_description]
            result['category'] = cached_category
            result['source'] = 'cache'
            result['confidence'] = 1.0
            return result
        
        # Try Random Forest first
        rf_prediction, rf_confidence = self._get_rf_prediction(description, amount)
        result['rf_prediction'] = rf_prediction
        result['rf_confidence'] = rf_confidence
        
        # If RF confidence is high enough, use RF prediction
        if rf_prediction and rf_confidence >= self.rf_confidence_threshold:
            result['category'] = rf_prediction
            result['source'] = 'rf'
            result['confidence'] = rf_confidence
            return result
        
        # Otherwise, try LLM if available
        if self.llm_available:
            llm_prediction = get_llama_category(description, extended_details)
            result['llm_prediction'] = llm_prediction
            
            if llm_prediction:
                result['category'] = llm_prediction
                result['source'] = 'llm'
                result['confidence'] = 0.95  # High confidence for LLM-based categories
                return result
        
        # If all else fails, use RF prediction even with low confidence
        if rf_prediction:
            result['category'] = rf_prediction
            result['source'] = 'rf_fallback'
            result['confidence'] = rf_confidence
        else:
            # Last resort
            result['category'] = 'Misc'
            result['source'] = 'fallback'
            result['confidence'] = 0.0
            
        return result
    
    def update_merchant_cache(self, description: str, category: str):
        """Update the merchant->category mapping cache
        
        Args:
            description: Transaction description
            category: Category to associate with this merchant
        """
        clean_description = description.strip().upper()
        self.merchant_category_map[clean_description] = category
        
    def save_merchant_cache(self, filename='merchant_cache.csv'):
        """Save the merchant->category mapping cache to a file"""
        if not self.merchant_category_map:
            print("No merchant mappings to save")
            return False
            
        df = pd.DataFrame({
            'merchant': list(self.merchant_category_map.keys()),
            'category': list(self.merchant_category_map.values())
        })
        
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} merchant mappings to {filename}")
        return True
        
    def load_merchant_cache(self, filename='merchant_cache.csv'):
        """Load the merchant->category mapping cache from a file"""
        if not os.path.exists(filename):
            print(f"Merchant cache file {filename} not found")
            return False
            
        df = pd.read_csv(filename)
        
        for _, row in df.iterrows():
            self.merchant_category_map[row['merchant']] = row['category']
            
        print(f"Loaded {len(df)} merchant mappings from {filename}")
        return True

def categorize_transactions(input_dir=TO_CATEGORIZE_DIR, output_dir=OUTPUT_DIR, use_postprocessor=True):
    """Categorize all transaction files in the input directory
    
    Args:
        input_dir: Directory containing transaction CSV files
        output_dir: Directory where categorized files will be saved
        use_postprocessor: Whether to use merchant post-processing
        
    Returns:
        bool: True if at least one file was processed successfully
    """
    # Create categorizer
    categorizer = TransactionCategorizer()
    
    # Initialize merchant post-processor if requested
    processor = None
    if use_postprocessor:
        try:
            processor = MerchantPostProcessor()
            print("Merchant post-processor initialized.")
        except Exception as e:
            print(f"Warning: Could not initialize merchant post-processor: {e}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of files to process
    files_to_process = []
    try:
        for filename in os.listdir(input_dir):
            if filename.lower().endswith('.csv'):
                files_to_process.append(os.path.join(input_dir, filename))
    except Exception as e:
        print(f"Error reading input directory: {e}")
        return False
    
    if not files_to_process:
        print(f"No CSV files found in {input_dir}")
        return False
    
    print(f"Found {len(files_to_process)} files to process")
    
    # Process each file
    successful_count = 0
    
    # Determine if we should use LLM instead of local model
    use_llm = categorizer.llm_available
    
    for file_path in files_to_process:
        base_name = os.path.basename(file_path)
        output_name = f"categorized_{base_name}"
        output_path = os.path.join(output_dir, output_name)
        
        if use_llm:
            # Use LLM API with batch processing
            success = process_file_with_llama(file_path, output_path, processor)
        else:
            # Use local model
            success = process_file_with_local_model(file_path, output_path, categorizer, processor)
            
        if success:
            successful_count += 1
    
    print(f"Successfully processed {successful_count} of {len(files_to_process)} files")
    return successful_count > 0

def process_file_with_local_model(file_path: str, output_path: str, categorizer: TransactionCategorizer, processor=None) -> bool:
    """Process a single file with the local ML model
    
    Args:
        file_path: Path to input CSV file
        output_path: Path to save output CSV file
        categorizer: Initialized TransactionCategorizer instance
        processor: Optional merchant post-processor
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Processing file {file_path} with hybrid model...")
    
    try:
        # Read the CSV file
        data = pd.read_csv(file_path)
        
        # Basic preprocessing
        data.columns = [c.lower() for c in data.columns]
        if 'category' not in data.columns:
            data['category'] = ''
        if 'confidence' not in data.columns:
            data['confidence'] = 0.0
        if 'source' not in data.columns:
            data['source'] = ''
            
        # Count how many transactions we'll process
        mask = data['category'].isna() | (data['category'] == '')
        uncategorized_count = mask.sum()
        already_categorized_count = len(data) - uncategorized_count
        
        if uncategorized_count == 0:
            print(f"  All {len(data)} transactions already have categories. Nothing to do.")
            return True
            
        print(f"  Found {uncategorized_count} transactions without categories (out of {len(data)} total)")
        if already_categorized_count > 0:
            print(f"  Preserving {already_categorized_count} existing categorizations")
            
        # Process each uncategorized transaction
        start_time = time.time()
        
        for idx in data[mask].index:
            description = data.loc[idx, 'description']
            amount = data.loc[idx, 'amount']
            extended_details = data.loc[idx, 'extended_details'] if 'extended_details' in data.columns else None
            
            # Get categorization
            result = categorizer.categorize(description, amount, extended_details)
            
            # Update data
            data.loc[idx, 'category'] = result['category']
            data.loc[idx, 'confidence'] = result['confidence']
            data.loc[idx, 'source'] = result['source']
            
        # Apply post-processing if requested
        if processor:
            print("  Applying merchant post-processing...")
            data = processor.process_transactions(data)
        
        # Save categorized data
        data.to_csv(output_path, index=False)
        
        elapsed_time = time.time() - start_time
        transactions_per_second = uncategorized_count / elapsed_time if elapsed_time > 0 else 0
        print(f"  Completed processing {uncategorized_count} transactions in {elapsed_time:.2f} seconds")
        print(f"  Average speed: {transactions_per_second:.2f} transactions per second")
        print(f"  Output saved to {output_path}")
        
        return True
    except Exception as e:
        print(f"  Error processing file: {e}")
        return False

if __name__ == "__main__":
    print("Transaction Categorizer")
    print("======================")
    categorize_transactions() 