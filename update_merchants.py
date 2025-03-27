#!/usr/bin/env python
"""
Script to update merchant categories based on model predictions
"""

import os
import pandas as pd
import numpy as np
import argparse
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from merchant_postprocessor import MerchantPostProcessor

def extract_merchant_suggestions(categorized_data_dir='data/categorized', 
                                 merchant_file='data/merchants/merchant_categories.csv',
                                 confidence_threshold=0.9,
                                 min_occurrences=3):
    """
    Extract potential merchant patterns from categorized data
    """
    # Load categorized data
    csv_files = [f for f in os.listdir(categorized_data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {categorized_data_dir}")
        return None
    
    all_data = []
    for file in csv_files:
        try:
            df = pd.read_csv(os.path.join(categorized_data_dir, file))
            if 'Description' in df.columns and 'Category' in df.columns:
                all_data.append(df[['Description', 'Category']])
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not all_data:
        print("No valid data found.")
        return None
        
    # Combine all data
    data = pd.concat(all_data, ignore_index=True)
    
    # Extract potential merchant names
    merchants = extract_merchant_names(data)
    
    # Load existing merchant patterns
    try:
        processor = MerchantPostProcessor(merchant_file=merchant_file)
        existing_merchants = processor.merchants
    except Exception:
        existing_merchants = pd.DataFrame(columns=['merchant_pattern', 'category', 'confidence'])
    
    # Find new merchant suggestions
    suggestions = []
    
    for merchant, stats in merchants.items():
        # Filter by minimum occurrences
        if stats['count'] < min_occurrences:
            continue
            
        # Calculate confidence based on category consistency
        total = stats['count']
        max_category_count = max(stats['categories'].values())
        consistency = max_category_count / total
        
        # Only include if confidence is high enough
        if consistency >= confidence_threshold:
            # Get the most common category
            most_common_category = max(stats['categories'].items(), key=lambda x: x[1])[0]
            
            # Check if this merchant pattern already exists
            if merchant in existing_merchants['merchant_pattern'].values:
                existing_row = existing_merchants[existing_merchants['merchant_pattern'] == merchant].iloc[0]
                existing_category = existing_row['category']
                
                # If the category is different, add as a suggestion for update
                if existing_category != most_common_category:
                    suggestions.append({
                        'merchant_pattern': merchant,
                        'current_category': existing_category,
                        'suggested_category': most_common_category,
                        'confidence': consistency,
                        'occurrences': total,
                        'status': 'update'
                    })
            else:
                # New merchant pattern
                suggestions.append({
                    'merchant_pattern': merchant,
                    'current_category': None,
                    'suggested_category': most_common_category,
                    'confidence': consistency,
                    'occurrences': total,
                    'status': 'new'
                })
    
    # Convert to DataFrame and sort
    if suggestions:
        suggestions_df = pd.DataFrame(suggestions)
        suggestions_df = suggestions_df.sort_values(by=['status', 'confidence', 'occurrences'], 
                                                   ascending=[True, False, False])
        return suggestions_df
    else:
        print("No new merchant suggestions found.")
        return None

def extract_merchant_names(data):
    """
    Extract potential merchant names from transaction descriptions
    """
    # Preprocess descriptions
    descriptions = data['Description'].fillna('').str.upper()
    
    # Use TF-IDF to find common tokens
    vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),  # Include 1-3 word phrases
        min_df=3,            # Appear in at least 3 transactions
        max_df=0.7,          # Appear in at most 70% of transactions
        stop_words='english'
    )
    
    # Fit the vectorizer
    try:
        vectorizer.fit(descriptions)
    except Exception as e:
        print(f"Error in vectorizer: {e}")
        return {}
    
    # Get feature names (potential merchant names)
    feature_names = vectorizer.get_feature_names_out()
    
    # Track merchant statistics
    merchants = {}
    
    # Group transactions by common tokens
    for i, row in data.iterrows():
        desc = row['Description']
        if pd.isna(desc):
            continue
            
        desc_upper = desc.upper()
        category = row['Category']
        
        # Try to find merchant names in description
        for name in feature_names:
            if ' ' in name:  # Focus on multi-word patterns first
                if name in desc_upper:
                    update_merchant_stats(merchants, name, category)
        
        # For short descriptions, consider the whole description as a merchant name
        words = desc_upper.split()
        if 1 <= len(words) <= 3:
            name = desc_upper
            update_merchant_stats(merchants, name, category)
    
    return merchants

def update_merchant_stats(merchants, name, category):
    """
    Update merchant statistics
    """
    if name not in merchants:
        merchants[name] = {
            'count': 0,
            'categories': defaultdict(int)
        }
    
    merchants[name]['count'] += 1
    merchants[name]['categories'][category] += 1

def update_merchant_categories(suggestions, merchant_file='data/merchants/merchant_categories.csv'):
    """
    Update merchant categories based on approved suggestions
    """
    if not suggestions or len(suggestions) == 0:
        print("No suggestions to update.")
        return
        
    # Load existing merchant patterns
    try:
        processor = MerchantPostProcessor(merchant_file=merchant_file)
        existing_merchants = processor.merchants
    except Exception:
        existing_merchants = pd.DataFrame(columns=['merchant_pattern', 'category', 'confidence'])
    
    updates = 0
    
    for _, suggestion in suggestions.iterrows():
        merchant = suggestion['merchant_pattern']
        category = suggestion['suggested_category']
        confidence = suggestion['confidence']
        
        if suggestion['status'] == 'update':
            # Update existing merchant
            idx = existing_merchants.index[existing_merchants['merchant_pattern'] == merchant].tolist()
            if idx:
                existing_merchants.at[idx[0], 'category'] = category
                existing_merchants.at[idx[0], 'confidence'] = confidence
                updates += 1
        elif suggestion['status'] == 'new':
            # Add new merchant
            new_row = pd.DataFrame([{
                'merchant_pattern': merchant,
                'category': category,
                'confidence': confidence
            }])
            existing_merchants = pd.concat([existing_merchants, new_row], ignore_index=True)
            updates += 1
    
    # Save the updated merchants
    try:
        existing_merchants.to_csv(merchant_file, index=False)
        print(f"Updated {updates} merchant patterns in {merchant_file}")
    except Exception as e:
        print(f"Error saving merchant file: {e}")

def interactive_update(categorized_data_dir='data/categorized', 
                       merchant_file='data/merchants/merchant_categories.csv',
                       confidence_threshold=0.9,
                       min_occurrences=3,
                       batch_size=10):
    """
    Interactive merchant category update process
    """
    # Get suggestions
    suggestions = extract_merchant_suggestions(
        categorized_data_dir=categorized_data_dir,
        merchant_file=merchant_file,
        confidence_threshold=confidence_threshold,
        min_occurrences=min_occurrences
    )
    
    if suggestions is None or len(suggestions) == 0:
        print("No merchant pattern suggestions found.")
        return
    
    print(f"\nFound {len(suggestions)} potential merchant pattern updates:")
    print(f" - {len(suggestions[suggestions['status'] == 'new'])} new patterns")
    print(f" - {len(suggestions[suggestions['status'] == 'update'])} updates to existing patterns")
    
    # Ask user if they want to review
    response = input("\nWould you like to review these suggestions? (y/n): ").strip().lower()
    if response != 'y':
        print("Update cancelled.")
        return
    
    # Initialize tracking
    approved = []
    
    # Process in batches
    for i in range(0, len(suggestions), batch_size):
        batch = suggestions.iloc[i:i+batch_size]
        
        print(f"\nReviewing suggestions {i+1}-{min(i+batch_size, len(suggestions))} of {len(suggestions)}:")
        print("-" * 100)
        
        for j, (_, suggestion) in enumerate(batch.iterrows()):
            status = suggestion['status']
            merchant = suggestion['merchant_pattern']
            current = suggestion['current_category'] if status == 'update' else "None"
            suggested = suggestion['suggested_category']
            confidence = suggestion['confidence']
            occurrences = suggestion['occurrences']
            
            print(f"{i+j+1}. {'Update' if status == 'update' else 'New'}: {merchant}")
            print(f"   Current category: {current}")
            print(f"   Suggested category: {suggested} (confidence: {confidence:.2f}, occurrences: {occurrences})")
            
            response = input(f"   Approve this {'update' if status == 'update' else 'addition'}? (y/n/q): ").strip().lower()
            
            if response == 'q':
                print("Review cancelled.")
                # Apply approved changes before exiting
                update_merchant_categories(pd.DataFrame(approved), merchant_file)
                return
            elif response == 'y':
                approved.append(suggestion)
            
            print("-" * 100)
        
        # Ask to continue after each batch
        if i + batch_size < len(suggestions):
            response = input("\nContinue to next batch? (y/n): ").strip().lower()
            if response != 'y':
                break
    
    # Apply approved changes
    if approved:
        approved_df = pd.DataFrame(approved)
        update_merchant_categories(approved_df, merchant_file)
        print(f"Applied {len(approved)} merchant pattern updates.")
    else:
        print("No changes were approved.")

def main():
    parser = argparse.ArgumentParser(description='Update merchant categories based on model predictions')
    
    parser.add_argument('--data-dir', type=str, default='data/categorized',
                        help='Directory containing categorized transaction data')
    
    parser.add_argument('--merchant-file', type=str, default='data/merchants/merchant_categories.csv',
                        help='Merchant categories file')
    
    parser.add_argument('--confidence', type=float, default=0.9,
                        help='Minimum confidence threshold for suggestions (0.0-1.0)')
    
    parser.add_argument('--min-occurrences', type=int, default=3,
                        help='Minimum number of occurrences for a merchant pattern')
    
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Number of suggestions to review at one time')
    
    parser.add_argument('--non-interactive', action='store_true',
                        help='Generate a report file without interactive review')
    
    parser.add_argument('--output', type=str, 
                        help='Output file for non-interactive mode')
    
    args = parser.parse_args()
    
    if args.non_interactive:
        # Just generate suggestions and save to file
        suggestions = extract_merchant_suggestions(
            categorized_data_dir=args.data_dir,
            merchant_file=args.merchant_file,
            confidence_threshold=args.confidence,
            min_occurrences=args.min_occurrences
        )
        
        if suggestions is not None and len(suggestions) > 0:
            output_file = args.output or 'merchant_suggestions.csv'
            suggestions.to_csv(output_file, index=False)
            print(f"Saved {len(suggestions)} merchant suggestions to {output_file}")
    else:
        # Interactive mode
        interactive_update(
            categorized_data_dir=args.data_dir,
            merchant_file=args.merchant_file,
            confidence_threshold=args.confidence,
            min_occurrences=args.min_occurrences,
            batch_size=args.batch_size
        )

if __name__ == '__main__':
    main() 