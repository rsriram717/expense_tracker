from dotenv import load_dotenv
import os
import pandas as pd
import pickle
import time
from datetime import datetime
import random
from improved_categorizer import (
    test_llama_connection,
    find_latest_model_file,
    client
)
from db_connector import get_engine
# Import directly from llm_prompts module instead of from improved_llama_prompt
from llm_prompts import (
    SYSTEM_PROMPT,
    generate_improved_prompt,
    parse_batch_response
)

# Load environment variables from .env file
load_dotenv()

class HybridCategorizer:
    """A hybrid categorizer that combines RF and Llama models"""
    
    def __init__(self, rf_confidence_threshold=0.7):
        """Initialize the hybrid categorizer
        
        Args:
            rf_confidence_threshold: Threshold for using RF prediction (0.0-1.0)
        """
        self.rf_model = None
        self.feature_prep = None
        self.rf_confidence_threshold = rf_confidence_threshold
        self.merchant_category_map = {}  # Cache for merchant->category mappings
        
        # Load the RF model
        self._load_rf_model()
        
        # Check if Llama is available
        self.llama_available = test_llama_connection()
        
    def _load_rf_model(self):
        """Load the latest Random Forest model"""
        try:
            model_filename = find_latest_model_file()
            if not model_filename:
                print("No model file found")
                return
                
            print(f"Loading RF model from {model_filename}")
            
            with open(model_filename, 'rb') as f:
                model_data = pickle.load(f)
                
            self.rf_model = model_data.get('model')
            self.feature_prep = model_data.get('feature_prep')
            self.categories = model_data.get('categories', [])
            
            if self.rf_model and self.feature_prep:
                print(f"RF model loaded successfully. Supports {len(self.categories)} categories.")
                return True
            else:
                print("Model loading failed: required components missing")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
            
    def _get_rf_prediction(self, description, amount):
        """Get prediction from Random Forest model with confidence
        
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
            
    def _get_llama_prediction(self, description, extended_details=None):
        """Get prediction from Llama model
        
        Returns:
            str: predicted category
        """
        if not self.llama_available or not client:
            return None
            
        try:
            # Generate prompt for a single transaction
            transaction = [(description, extended_details)]
            prompt = generate_improved_prompt(transaction)
            
            # Call the API
            response = client.chat.completions.create(
                model="meta-llama/llama-3.1-8b-instruct/fp-8",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.0
            )
            
            raw_response = response.choices[0].message.content.strip()
            
            # Parse the response
            predictions = parse_batch_response(raw_response, 1)
            if predictions:
                return predictions[0]
            return None
        except Exception as e:
            print(f"Error in Llama prediction: {e}")
            return None
    
    def categorize(self, description, amount, extended_details=None):
        """Categorize a transaction using the hybrid approach
        
        Returns:
            dict: {
                'category': predicted category,
                'source': 'rf' or 'llama',
                'confidence': confidence score (0.0-1.0),
                'rf_prediction': RF model prediction,
                'rf_confidence': RF confidence score,
                'llama_prediction': Llama model prediction
            }
        """
        result = {
            'category': None,
            'source': None,
            'confidence': 0.0,
            'rf_prediction': None,
            'rf_confidence': 0.0,
            'llama_prediction': None
        }
        
        # Check merchant cache first
        clean_description = description.strip().upper()
        if clean_description in self.merchant_category_map:
            cached_category = self.merchant_category_map[clean_description]
            result['category'] = cached_category
            result['source'] = 'cache'
            result['confidence'] = 1.0
            print(f"Using cached category for '{clean_description}': {cached_category}")
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
        
        # Otherwise, try Llama
        if self.llama_available:
            llama_prediction = self._get_llama_prediction(description, extended_details)
            result['llama_prediction'] = llama_prediction
            
            if llama_prediction:
                result['category'] = llama_prediction
                result['source'] = 'llama'
                result['confidence'] = 0.5  # Default confidence for Llama
                return result
        
        # If all else fails, use RF prediction even with low confidence
        if rf_prediction:
            result['category'] = rf_prediction
            result['source'] = 'rf_fallback'
            result['confidence'] = rf_confidence
            
        return result
    
    def update_merchant_cache(self, description, category):
        """Update the merchant->category mapping cache"""
        clean_description = description.strip().upper()
        self.merchant_category_map[clean_description] = category
        
    def save_merchant_cache(self, filename='merchant_cache.csv'):
        """Save the merchant->category mapping cache to a file"""
        if not self.merchant_category_map:
            print("No merchant mappings to save")
            return
            
        df = pd.DataFrame({
            'merchant': list(self.merchant_category_map.keys()),
            'category': list(self.merchant_category_map.values())
        })
        
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} merchant mappings to {filename}")
        
    def load_merchant_cache(self, filename='merchant_cache.csv'):
        """Load the merchant->category mapping cache from a file"""
        if not os.path.exists(filename):
            print(f"Merchant cache file {filename} not found")
            return
            
        df = pd.read_csv(filename)
        
        for _, row in df.iterrows():
            self.merchant_category_map[row['merchant']] = row['category']
            
        print(f"Loaded {len(df)} merchant mappings from {filename}")


def test_hybrid_categorizer():
    """Test the hybrid categorizer on sample transactions"""
    # Create the hybrid categorizer
    categorizer = HybridCategorizer(rf_confidence_threshold=0.7)
    
    # Load test data
    comparison_files = [f for f in os.listdir() if f.startswith('model_comparison_') and f.endswith('.csv')]
    if not comparison_files:
        print("No comparison results found. Cannot test hybrid categorizer.")
        return
        
    # Get the most recent file
    latest_file = max(comparison_files)
    print(f"Loading test transactions from {latest_file}")
    
    # Load the data
    results = pd.read_csv(latest_file)
    
    # Sample some transactions
    sample_size = min(20, len(results))
    sample = results.sample(sample_size)
    
    # Test the hybrid categorizer
    print("\n===== TESTING HYBRID CATEGORIZER =====")
    hybrid_results = []
    
    for _, row in sample.iterrows():
        description = row['description']
        amount = row['amount']
        true_category = row['true_category']
        
        print(f"\nProcessing: '{description}'")
        start_time = time.time()
        
        # Get hybrid prediction
        result = categorizer.categorize(description, amount)
        
        # Record results
        result['description'] = description
        result['amount'] = amount
        result['true_category'] = true_category
        result['correct'] = result['category'] == true_category
        result['processing_time'] = time.time() - start_time
        hybrid_results.append(result)
        
        # Print result
        print(f"  True Category: {true_category}")
        print(f"  Hybrid Category: {result['category']} (source: {result['source']}, confidence: {result['confidence']:.2f})")
        if result['rf_prediction']:
            print(f"  RF Prediction: {result['rf_prediction']} (confidence: {result['rf_confidence']:.2f})")
        if result['llama_prediction']:
            print(f"  Llama Prediction: {result['llama_prediction']}")
        print(f"  Correct: {result['correct']}")
        print(f"  Processing Time: {result['processing_time']:.2f} seconds")
        
        # Update merchant cache if correct
        if result['correct']:
            categorizer.update_merchant_cache(description, true_category)
    
    # Calculate overall results
    correct_count = sum(r['correct'] for r in hybrid_results)
    accuracy = correct_count / len(hybrid_results)
    
    print("\n===== HYBRID CATEGORIZER RESULTS =====")
    print(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(hybrid_results)})")
    
    # Breakdown by source
    sources = {}
    for r in hybrid_results:
        source = r['source']
        if source not in sources:
            sources[source] = {'total': 0, 'correct': 0}
        sources[source]['total'] += 1
        if r['correct']:
            sources[source]['correct'] += 1
    
    print("\nBreakdown by source:")
    for source, stats in sources.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {source}: {accuracy:.2%} ({stats['correct']}/{stats['total']})")
    
    # Compare with original Llama results
    llama_correct = sample['llama_correct'].sum()
    llama_accuracy = llama_correct / len(sample)
    
    print(f"\nOriginal Llama accuracy: {llama_accuracy:.2%} ({llama_correct}/{len(sample)})")
    
    if accuracy > llama_accuracy:
        improvement = (accuracy - llama_accuracy) / llama_accuracy * 100 if llama_accuracy > 0 else float('inf')
        print(f"Improvement: +{improvement:.1f}%")
    
    # Save merchant cache
    categorizer.save_merchant_cache()
    
    return hybrid_results


if __name__ == "__main__":
    print("Testing hybrid categorizer...")
    test_hybrid_categorizer() 