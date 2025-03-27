import pandas as pd
import os
import re

class MerchantPostProcessor:
    """
    A post-processor that uses a list of known merchants to improve categorization.
    This can be used after ML model prediction to correct common merchants.
    """
    
    def __init__(self, merchant_file='data/merchants/merchant_categories.csv', min_confidence_threshold=0.85):
        """Initialize the post-processor with a merchant categories file."""
        self.merchant_file = merchant_file
        self.min_confidence_threshold = min_confidence_threshold
        self.merchants = self._load_merchants()
        
    def _load_merchants(self):
        """Load merchant categories from CSV file."""
        if not os.path.exists(self.merchant_file):
            print(f"Warning: Merchant file {self.merchant_file} not found.")
            return pd.DataFrame(columns=['merchant_pattern', 'category', 'confidence'])
        
        try:
            merchants = pd.read_csv(self.merchant_file)
            print(f"Loaded {len(merchants)} merchant patterns for post-processing.")
            return merchants
        except Exception as e:
            print(f"Error loading merchant file: {e}")
            return pd.DataFrame(columns=['merchant_pattern', 'category', 'confidence'])
    
    def reload_merchants(self):
        """Reload merchant data from file in case it was updated."""
        self.merchants = self._load_merchants()
        
    def process_transactions(self, transactions_df):
        """
        Apply merchant-based post-processing to transaction categories.
        
        Args:
            transactions_df: DataFrame with transactions including 'Description', 
                            'Category', and 'Confidence' columns
        
        Returns:
            DataFrame with updated categories and confidence values
        """
        if len(self.merchants) == 0:
            print("No merchant patterns available for post-processing.")
            return transactions_df
        
        # Make a copy to avoid modifying the original
        df = transactions_df.copy()
        
        # Track changes for reporting
        changes = []
        
        # Check each transaction
        for idx, row in df.iterrows():
            description = row['Description']
            if pd.isna(description):
                continue
                
            description = description.upper()
            appears_as = row.get('Appears On Your Statement As', '')
            if pd.isna(appears_as):
                appears_as = ''
            appears_as = appears_as.upper()
            
            # Combined text to search for merchant patterns
            full_text = f"{description} {appears_as}"
            
            # Check each merchant pattern
            for _, merchant_row in self.merchants.iterrows():
                pattern = merchant_row['merchant_pattern'].upper()
                category = merchant_row['category']
                confidence = float(merchant_row['confidence'])
                
                # Only use merchant patterns with high confidence
                if confidence < self.min_confidence_threshold:
                    continue
                
                # More strict pattern matching - match word boundaries 
                # or as standalone words to avoid partial matches
                pattern_re = r'\b' + re.escape(pattern) + r'\b'
                if re.search(pattern_re, full_text) or pattern == full_text.strip():
                    current_confidence = row.get('Confidence', 0)
                    
                    # Only override if ML prediction has lower confidence
                    # and the confidence improvement is significant
                    confidence_improvement = confidence - current_confidence
                    
                    if confidence_improvement > 0.15:
                        old_category = row['Category']
                        old_confidence = current_confidence
                        
                        # Update category and confidence
                        df.at[idx, 'Category'] = category
                        df.at[idx, 'Confidence'] = confidence
                        
                        # Record the change
                        changes.append({
                            'Description': description,
                            'Old Category': old_category,
                            'New Category': category,
                            'Old Confidence': old_confidence,
                            'New Confidence': confidence,
                            'Matched Pattern': pattern,
                            'Confidence Improvement': confidence_improvement
                        })
                        break  # Stop after finding the first match
        
        # Report changes
        if changes:
            print(f"Post-processing updated {len(changes)} transactions:")
            for change in changes[:10]:  # Show first 10 changes only to avoid overwhelming output
                print(f"  {change['Description']}: {change['Old Category']} ({change['Old Confidence']:.2f}) -> {change['New Category']} ({change['New Confidence']:.2f}) [+{change['Confidence Improvement']:.2f}]")
            
            if len(changes) > 10:
                print(f"  ... and {len(changes) - 10} more changes.")
        else:
            print("No transactions were updated by post-processing.")
        
        return df
    
    def add_merchant_pattern(self, pattern, category, confidence=0.9):
        """
        Add a new merchant pattern to the CSV file.
        
        Args:
            pattern: The merchant pattern to match (case-insensitive)
            category: The category to assign
            confidence: Confidence level (0.0-1.0)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add to the DataFrame
            new_row = pd.DataFrame([{
                'merchant_pattern': pattern,
                'category': category,
                'confidence': confidence
            }])
            
            if len(self.merchants) == 0:
                # Create new DataFrame if empty
                self.merchants = new_row
            else:
                # Append to existing
                self.merchants = pd.concat([self.merchants, new_row], ignore_index=True)
            
            # Save to file
            self.merchants.to_csv(self.merchant_file, index=False)
            print(f"Added merchant pattern: {pattern} -> {category} (confidence: {confidence})")
            return True
        except Exception as e:
            print(f"Error adding merchant pattern: {e}")
            return False
    
    def remove_merchant_pattern(self, pattern):
        """
        Remove a merchant pattern from the CSV file.
        
        Args:
            pattern: The pattern to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if pattern exists
            if pattern not in self.merchants['merchant_pattern'].values:
                print(f"Pattern '{pattern}' not found.")
                return False
            
            # Filter out the pattern
            self.merchants = self.merchants[self.merchants['merchant_pattern'] != pattern]
            
            # Save to file
            self.merchants.to_csv(self.merchant_file, index=False)
            print(f"Removed merchant pattern: {pattern}")
            return True
        except Exception as e:
            print(f"Error removing merchant pattern: {e}")
            return False


# Usage example:
if __name__ == "__main__":
    processor = MerchantPostProcessor()
    
    # Example: Add a new merchant pattern
    # processor.add_merchant_pattern("CHIPOTLE", "Food & Drink", 0.95)
    
    # Example: Process a transactions file
    transactions_file = "data/output/improved_categorized_transactions_example.csv"
    if os.path.exists(transactions_file):
        print(f"Processing transactions from {transactions_file}")
        transactions = pd.read_csv(transactions_file)
        
        # Apply post-processing
        updated_transactions = processor.process_transactions(transactions)
        
        # Save updated transactions
        output_file = "data/output/postprocessed_transactions_example.csv"
        updated_transactions.to_csv(output_file, index=False)
        print(f"Saved updated transactions to {output_file}")
    else:
        print(f"Transactions file {transactions_file} not found.") 