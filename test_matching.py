#!/usr/bin/env python
"""
Test script for the improved merchant pattern matching
"""

import pandas as pd
from merchant_postprocessor import MerchantPostProcessor

def test_pattern_matching():
    """Test the merchant pattern matching with some examples"""
    # Create a processor
    processor = MerchantPostProcessor()
    
    # Test patterns
    test_descriptions = [
        # Positive matches
        "UBER TRIP HELP.UBER.COM",
        "STARBUCKS NEW YORK NY",
        "AMAZON.COM PAYMENT",
        "TRADER JOE'S #123 CHICAGO IL",
        "NETFLIX SUBSCRIPTION",
        
        # Negative matches (should no longer match due to word boundary checks)
        "SMARTPHONE REPAIR SHOP",
        "PHONETICS LAB PAYMENT",
        "AUTOZONE AUTO PARTS",
        "WINESTORE DOWNTOWN",
        "SHELL OF YOUR FORMER SELF THERAPY",
        "CARNIVAL CRUISE LINE",
        "MCARDLE FITNESS CENTER",
        "APPLEBEE'S RESTAURANT",
        
        # Common issues (edge cases)
        "T-MOBILE BILL PAYMENT",
        "THE WINE SHOP",
        "PELOTON MONTHLY MEMBERSHIP",
        "OPENAI CHATGPT PLUS",
        "MOBILE PHONE STORE",
        "BMW SERVICE CENTER"
    ]
    
    # Create test dataframe
    test_df = pd.DataFrame({
        'Description': test_descriptions,
        'Category': ['Unknown'] * len(test_descriptions),
        'Confidence': [0.5] * len(test_descriptions)
    })
    
    # Process the test data
    print("Testing merchant pattern matching with improved algorithm...\n")
    result_df = processor.process_transactions(test_df)
    
    # Show result comparison
    print("\nFinal Results:")
    print("-" * 100)
    print(f"{'Description':<40} {'Original Category':<20} {'New Category':<20} {'Matched?':<10}")
    print("-" * 100)
    
    for i in range(len(test_df)):
        original = test_df.iloc[i]
        result = result_df.iloc[i]
        matched = original['Category'] != result['Category']
        
        print(f"{original['Description']:<40} {original['Category']:<20} {result['Category']:<20} {'✓' if matched else '✗'}")

if __name__ == "__main__":
    test_pattern_matching() 