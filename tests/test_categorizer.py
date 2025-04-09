"""
Tests for the transaction categorizer functionality
"""

import os
import pandas as pd
import unittest
from unittest.mock import patch, MagicMock
import sys

# Add parent directory to path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transaction_categorizer import TransactionCategorizer, categorize_transactions
from model_training import find_latest_model_file
from llm_service import test_llama_connection

class TestTransactionCategorizer(unittest.TestCase):
    """Tests for the TransactionCategorizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock the model loading and LLM connection
        with patch('transaction_categorizer.load_model') as mock_load:
            with patch('transaction_categorizer.test_llama_connection') as mock_test:
                mock_load.return_value = self._create_mock_model_data()
                mock_test.return_value = False  # No LLM by default
                self.categorizer = TransactionCategorizer()
    
    def _create_mock_model_data(self):
        """Create a mock model data dictionary"""
        mock_model = MagicMock()
        mock_model.predict.return_value = ['Groceries']
        mock_model.predict_proba.return_value = [[0.8, 0.1, 0.1]]
        mock_model.classes_ = ['Groceries', 'Restaurants', 'Transportation']
        
        mock_feature_prep = MagicMock()
        mock_feature_prep.transform.return_value = 'transformed_features'
        
        return {
            'model': mock_model,
            'feature_prep': mock_feature_prep,
            'categories': ['Groceries', 'Restaurants', 'Transportation']
        }
    
    def test_init(self):
        """Test initialization"""
        self.assertIsNotNone(self.categorizer.rf_model)
        self.assertIsNotNone(self.categorizer.feature_prep)
        self.assertEqual(len(self.categorizer.rf_categories), 3)
        self.assertEqual(self.categorizer.llm_available, False)
    
    def test_get_rf_prediction(self):
        """Test getting prediction from RF model"""
        category, confidence = self.categorizer._get_rf_prediction('GROCERY STORE', 50.0)
        
        self.assertEqual(category, 'Groceries')
        self.assertEqual(confidence, 0.8)
        
        # Verify the model was called correctly
        self.categorizer.rf_model.predict.assert_called_once()
        self.categorizer.rf_model.predict_proba.assert_called_once()
    
    def test_categorize_rf_high_confidence(self):
        """Test categorization with high RF confidence"""
        result = self.categorizer.categorize('GROCERY STORE', 50.0)
        
        self.assertEqual(result['category'], 'Groceries')
        self.assertEqual(result['source'], 'rf')
        self.assertEqual(result['confidence'], 0.8)
    
    def test_merchant_cache(self):
        """Test merchant cache functionality"""
        # Add a merchant to the cache
        self.categorizer.update_merchant_cache('TARGET', 'Shopping')
        
        # Test that it's used when categorizing
        result = self.categorizer.categorize('TARGET', 100.0)
        
        self.assertEqual(result['category'], 'Shopping')
        self.assertEqual(result['source'], 'cache')
        self.assertEqual(result['confidence'], 1.0)
    
    @patch('transaction_categorizer.process_file_with_llama')
    @patch('transaction_categorizer.process_file_with_local_model')
    def test_categorize_transactions_calls_correct_processor(self, mock_local, mock_llama):
        """Test that categorize_transactions calls the correct processor"""
        # Mock file existence and listing
        with patch('os.path.exists', return_value=True):
            with patch('os.listdir', return_value=['file1.csv', 'file2.csv']):
                with patch('os.makedirs'):
                    with patch('transaction_categorizer.TransactionCategorizer') as mock_cat:
                        # Set up the mock categorizer to have LLM available
                        instance = mock_cat.return_value
                        instance.llm_available = True
                        
                        # Both mocks return success
                        mock_local.return_value = True
                        mock_llama.return_value = True
                        
                        # Call the function
                        result = categorize_transactions(
                            input_dir='test_input',
                            output_dir='test_output'
                        )
                        
                        # Verify it used the LLM processor
                        self.assertEqual(mock_llama.call_count, 2)
                        self.assertEqual(mock_local.call_count, 0)
                        self.assertTrue(result)
    
    def test_save_load_merchant_cache(self):
        """Test saving and loading the merchant cache"""
        # Add some merchants
        self.categorizer.update_merchant_cache('TARGET', 'Shopping')
        self.categorizer.update_merchant_cache('WALMART', 'Shopping')
        self.categorizer.update_merchant_cache('STARBUCKS', 'Food & Drink')
        
        # Mock the file operations
        mock_open = unittest.mock.mock_open()
        with patch('builtins.open', mock_open):
            with patch('pandas.DataFrame.to_csv') as mock_to_csv:
                # Test saving
                result = self.categorizer.save_merchant_cache('test_cache.csv')
                self.assertTrue(result)
                mock_to_csv.assert_called_once()

        # Test loading (with a mock file)
        test_data = pd.DataFrame({
            'merchant': ['KROGER', 'CHEVRON'],
            'category': ['Groceries', 'Transportation']
        })
        
        with patch('os.path.exists', return_value=True):
            with patch('pandas.read_csv', return_value=test_data):
                result = self.categorizer.load_merchant_cache('test_cache.csv')
                self.assertTrue(result)
                
                # Verify the cache was updated
                self.assertEqual(self.categorizer.merchant_category_map['KROGER'], 'Groceries')
                self.assertEqual(self.categorizer.merchant_category_map['CHEVRON'], 'Transportation')

if __name__ == '__main__':
    unittest.main() 