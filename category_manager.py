import pandas as pd
import numpy as np
from tabulate import tabulate
import os
import json
import glob
from categorize_transactions import TransactionCategorizer

class CategoryManager:
    def __init__(self, categorizer=None):
        """Initialize the category manager."""
        self.categorizer = categorizer or TransactionCategorizer()
        self.feedback_data = pd.DataFrame()
        self.feedback_file = 'category_feedback.csv'
        self.load_feedback()
        
    def load_feedback(self):
        """Load previous feedback if available."""
        if os.path.exists(self.feedback_file):
            self.feedback_data = pd.read_csv(self.feedback_file)
            print(f"Loaded {len(self.feedback_data)} feedback entries")
    
    def save_feedback(self):
        """Save feedback for future use."""
        self.feedback_data.to_csv(self.feedback_file, index=False)
        print(f"Saved {len(self.feedback_data)} feedback entries to {self.feedback_file}")
    
    def display_transactions(self, df, start=0, count=10):
        """Display a subset of transactions for review."""
        end = min(start + count, len(df))
        subset = df.iloc[start:end].copy()
        
        # Format for display
        display_cols = ['Date', 'Description', 'Amount', 'Category', 'Confidence']
        formatted = subset[display_cols].copy()
        
        # Format confidence as percentage
        if 'Confidence' in formatted.columns:
            formatted['Confidence'] = formatted['Confidence'].apply(lambda x: f"{x*100:.1f}%")
        
        print(tabulate(formatted, headers='keys', tablefmt='grid', showindex=True))
        
        return subset
    
    def interactive_review(self, transactions_path, categorized_output_path=None, batch_size=10):
        """Allow interactive review and correction of categorized transactions."""
        if self.categorizer.model is None:
            print("No model loaded. Please train or load a model first.")
            return
        
        # Categorize transactions
        categorized = self.categorizer.categorize(transactions_path)
        total_transactions = len(categorized)
        
        # Sort by confidence (ascending) to review the least confident predictions first
        if 'Confidence' in categorized.columns:
            categorized = categorized.sort_values(by='Confidence')
        
        start_idx = 0
        while start_idx < total_transactions:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"\nReviewing transactions {start_idx+1}-{min(start_idx+batch_size, total_transactions)} of {total_transactions}")
            print(f"File: {os.path.basename(transactions_path)}")
            print("Low confidence transactions are shown first.\n")
            
            subset = self.display_transactions(categorized, start_idx, batch_size)
            
            print("\nOptions:")
            print("[n]ext batch, [p]revious batch, [s]ave and exit")
            print("[#] to edit a transaction (e.g., '2' to edit the 3rd transaction shown)")
            print("[c] to view categories, [a] to add a new category")
            
            choice = input("\nEnter choice: ").strip().lower()
            
            if choice == 'n':
                start_idx += batch_size
            elif choice == 'p':
                start_idx = max(0, start_idx - batch_size)
            elif choice == 's':
                break
            elif choice == 'c':
                self.display_categories()
            elif choice == 'a':
                self.add_new_category()
            elif choice.isdigit() and 0 <= int(choice) < len(subset):
                self.edit_transaction(categorized, subset, int(choice))
            else:
                print("Invalid choice. Please try again.")
        
        # Save final results
        if categorized_output_path:
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(categorized_output_path)), exist_ok=True)
            categorized.to_csv(categorized_output_path, index=False)
            print(f"Saved categorized transactions to {categorized_output_path}")
        
        # Save feedback
        self.save_feedback()
        
        return categorized
    
    def review_directory(self, transactions_directory, output_directory, batch_size=10):
        """Review and correct all CSV files in a directory."""
        # Find all CSV files in the directory
        csv_files = glob.glob(os.path.join(transactions_directory, "*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {transactions_directory}")
            return
        
        print(f"Found {len(csv_files)} CSV files for review")
        
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            output_path = os.path.join(output_directory, f"reviewed_{filename}")
            
            print(f"\nReviewing {filename}...")
            self.interactive_review(file_path, output_path, batch_size)
    
    def edit_transaction(self, full_df, subset, idx):
        """Edit the category for a transaction."""
        # Get actual index in the full dataframe
        actual_idx = subset.index[idx]
        
        transaction = full_df.loc[actual_idx]
        print(f"\nEditing transaction: {transaction['Description']} (${transaction['Amount']})")
        print(f"Current category: {transaction['Category']}")
        
        # Show available categories
        categories = sorted(self.categorizer.categories)
        for i, cat in enumerate(categories):
            print(f"{i+1}. {cat}")
        
        print("Enter category number, or type a new category name:")
        choice = input("> ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(categories):
            new_category = categories[int(choice) - 1]
        else:
            new_category = choice
        
        # Update the category
        full_df.at[actual_idx, 'Category'] = new_category
        full_df.at[actual_idx, 'Confidence'] = 1.0  # Set to 100% since it's manually verified
        
        # Record feedback for model improvement
        feedback_row = transaction.to_dict()
        feedback_row['Category'] = new_category
        self.feedback_data = pd.concat([self.feedback_data, pd.DataFrame([feedback_row])], ignore_index=True)
        
        print(f"Updated category to: {new_category}")
        input("Press Enter to continue...")
    
    def display_categories(self):
        """Display all available categories."""
        if not self.categorizer.categories:
            print("No categories available.")
            return
        
        print("\nAvailable Categories:")
        categories = sorted(self.categorizer.categories)
        for i, cat in enumerate(categories):
            print(f"{i+1}. {cat}")
        
        input("\nPress Enter to continue...")
    
    def add_new_category(self):
        """Add a new category to the system."""
        print("\nEnter new category name:")
        new_category = input("> ").strip()
        
        if not new_category:
            print("Category name cannot be empty.")
            return
        
        if new_category in self.categorizer.categories:
            print(f"Category '{new_category}' already exists.")
            return
        
        self.categorizer.categories.append(new_category)
        print(f"Added new category: {new_category}")
        
        # Update the saved model to include the new category
        self.categorizer.save_model()
        
        input("Press Enter to continue...")
    
    def retrain_with_feedback(self, original_data_path):
        """Retrain the model with feedback data."""
        if len(self.feedback_data) == 0:
            print("No feedback data available for retraining.")
            return
        
        print(f"Retraining model with {len(self.feedback_data)} feedback entries...")
        self.categorizer.retrain_with_feedback(original_data_path, self.feedback_data)
        print("Model retrained successfully.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Manage transaction categories.')
    parser.add_argument('--review', action='store_true', help='Review and modify categorized transactions')
    parser.add_argument('--transactions', default='data/to_categorize', 
                        help='Path to transactions file or directory for review')
    parser.add_argument('--output', default='data/output', 
                        help='Path for output file or directory')
    parser.add_argument('--batch-size', type=int, default=10, 
                        help='Number of transactions to display at once')
    parser.add_argument('--retrain', action='store_true', 
                        help='Retrain model with feedback data')
    parser.add_argument('--training-data', default='data/categorized', 
                        help='Original training data for retraining (file or directory)')
    parser.add_argument('--file', help='Specific file to review if transactions is a directory')
    
    args = parser.parse_args()
    
    # Initialize categorizer and manager
    categorizer = TransactionCategorizer()
    if categorizer.model is None:
        print("Training model first...")
        categorizer.train(args.training_data)
    
    manager = CategoryManager(categorizer)
    
    if args.review:
        if os.path.isdir(args.transactions):
            if args.file:
                # Review specific file
                file_path = os.path.join(args.transactions, args.file)
                if not os.path.exists(file_path):
                    print(f"File {args.file} not found in {args.transactions}")
                    return
                
                output_path = os.path.join(args.output, f"reviewed_{args.file}")
                manager.interactive_review(file_path, output_path, args.batch_size)
            else:
                # List files and ask user which to review
                csv_files = glob.glob(os.path.join(args.transactions, "*.csv"))
                if not csv_files:
                    print(f"No CSV files found in {args.transactions}")
                    return
                
                print("\nAvailable files for review:")
                for i, file_path in enumerate(csv_files):
                    filename = os.path.basename(file_path)
                    print(f"{i+1}. {filename}")
                
                choice = input("\nEnter file number to review (or 'a' for all): ")
                
                if choice == 'a':
                    # Review all files
                    for file_path in csv_files:
                        filename = os.path.basename(file_path)
                        output_path = os.path.join(args.output, f"reviewed_{filename}")
                        print(f"\nReviewing {filename}...")
                        manager.interactive_review(file_path, output_path, args.batch_size)
                elif choice.isdigit() and 1 <= int(choice) <= len(csv_files):
                    # Review specific file
                    file_path = csv_files[int(choice) - 1]
                    filename = os.path.basename(file_path)
                    output_path = os.path.join(args.output, f"reviewed_{filename}")
                    manager.interactive_review(file_path, output_path, args.batch_size)
                else:
                    print("Invalid choice")
        else:
            output_path = args.output
            if os.path.isdir(output_path):
                basename = os.path.basename(args.transactions)
                output_path = os.path.join(output_path, f"reviewed_{basename}")
            
            manager.interactive_review(args.transactions, output_path, args.batch_size)
    
    if args.retrain:
        manager.retrain_with_feedback(args.training_data)


if __name__ == "__main__":
    main() 