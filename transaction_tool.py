#!/usr/bin/env python
import argparse
import os
import glob
from categorize_transactions import TransactionCategorizer
from category_manager import CategoryManager

def print_banner():
    """Print a banner for the tool."""
    print("\n" + "="*60)
    print("TRANSACTION CATEGORIZATION TOOL")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Credit Card Transaction Categorization Tool')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train model command
    train_parser = subparsers.add_parser('train', help='Train a new categorization model')
    train_parser.add_argument('--data', default='data/categorized', 
                              help='Path to categorized data for training (file or directory)')
    train_parser.add_argument('--model', default='transaction_model.pkl',
                             help='Path to save the trained model')
    
    # Categorize command
    categorize_parser = subparsers.add_parser('categorize', help='Categorize new transactions')
    categorize_parser.add_argument('--transactions', default='data/to_categorize',
                                 help='Path to new transactions for categorization (file or directory)')
    categorize_parser.add_argument('--output', default='data/output',
                                 help='Path for output categorized file or directory')
    categorize_parser.add_argument('--model', default='transaction_model.pkl',
                                 help='Path to the trained model')
    
    # Review command
    review_parser = subparsers.add_parser('review', help='Review and modify categorized transactions')
    review_parser.add_argument('--transactions', default='data/to_categorize',
                             help='Path to transactions file or directory for review')
    review_parser.add_argument('--output', default='data/output',
                             help='Path for output file or directory')
    review_parser.add_argument('--batch-size', type=int, default=10,
                             help='Number of transactions to display at once')
    review_parser.add_argument('--model', default='transaction_model.pkl',
                             help='Path to the trained model')
    review_parser.add_argument('--file', help='Specific file to review if directory contains multiple files')
    
    # Retrain command
    retrain_parser = subparsers.add_parser('retrain', help='Retrain model with feedback data')
    retrain_parser.add_argument('--training-data', default='data/categorized',
                              help='Original training data for retraining (file or directory)')
    retrain_parser.add_argument('--feedback', default='category_feedback.csv',
                              help='Feedback data file')
    retrain_parser.add_argument('--model', default='transaction_model.pkl',
                              help='Path to the trained model')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate spending report from categorized data')
    report_parser.add_argument('--data', default='data/output',
                             help='Path to categorized data file or directory')
    report_parser.add_argument('--output', default='spending_report.csv',
                             help='Path for output report file')
    report_parser.add_argument('--file', help='Specific file to generate report for if directory contains multiple files')
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.command == 'train':
        print(f"Training a new model using data from {args.data}...")
        categorizer = TransactionCategorizer(model_path=args.model)
        categorizer.train(args.data)
        print(f"Model trained and saved to {args.model}")
    
    elif args.command == 'categorize':
        print(f"Categorizing transactions from {args.transactions}...")
        categorizer = TransactionCategorizer(model_path=args.model)
        if categorizer.model is None:
            print(f"No model found at {args.model}. Training first...")
            categorizer.train('data/categorized')
        
        if os.path.isdir(args.transactions):
            results = categorizer.categorize_directory(args.transactions, args.output)
            
            # Generate and display summary statistics
            total_transactions = 0
            for filename, df in results.items():
                total_transactions += len(df)
            
            print(f"\nCategorized {total_transactions} transactions from {len(results)} files")
            print(f"Results saved to {args.output} directory")
        else:
            output_path = args.output
            if os.path.isdir(output_path):
                basename = os.path.basename(args.transactions)
                output_path = os.path.join(output_path, f"categorized_{basename}")
            
            categorized_df = categorizer.categorize(args.transactions, output_path)
            stats = categorizer.generate_category_stats(categorized_df)
            
            print("\nCategory statistics:")
            print(stats[['count', 'sum']].sort_values('sum', ascending=False))
            print(f"\nCategorized {len(categorized_df)} transactions")
            print(f"Output saved to {output_path}")
    
    elif args.command == 'review':
        print(f"Reviewing and modifying transaction categories...")
        categorizer = TransactionCategorizer(model_path=args.model)
        if categorizer.model is None:
            print(f"No model found at {args.model}. Training first...")
            categorizer.train('data/categorized')
        
        manager = CategoryManager(categorizer)
        
        if os.path.isdir(args.transactions):
            if args.file:
                # Review specific file in directory
                transaction_file = os.path.join(args.transactions, args.file)
                if not os.path.exists(transaction_file):
                    print(f"File {args.file} not found in {args.transactions}")
                    return
                
                output_file = os.path.join(args.output, f"reviewed_{args.file}")
                manager.interactive_review(transaction_file, output_file, args.batch_size)
            else:
                # List files and ask user which one to review
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
                        output_file = os.path.join(args.output, f"reviewed_{filename}")
                        print(f"\nReviewing {filename}...")
                        manager.interactive_review(file_path, output_file, args.batch_size)
                elif choice.isdigit() and 1 <= int(choice) <= len(csv_files):
                    # Review specific file
                    file_path = csv_files[int(choice) - 1]
                    filename = os.path.basename(file_path)
                    output_file = os.path.join(args.output, f"reviewed_{filename}")
                    manager.interactive_review(file_path, output_file, args.batch_size)
                else:
                    print("Invalid choice")
        else:
            # Single file review
            output_path = args.output
            if os.path.isdir(output_path):
                basename = os.path.basename(args.transactions)
                output_path = os.path.join(output_path, f"reviewed_{basename}")
            
            manager.interactive_review(args.transactions, output_path, args.batch_size)
    
    elif args.command == 'retrain':
        print(f"Retraining model with feedback data...")
        categorizer = TransactionCategorizer(model_path=args.model)
        if categorizer.model is None:
            print(f"No model found at {args.model}. Training first...")
            categorizer.train(args.training_data)
        
        manager = CategoryManager(categorizer)
        manager.retrain_with_feedback(args.training_data)
    
    elif args.command == 'report':
        print(f"Generating spending report...")
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from datetime import datetime
        
        if os.path.isdir(args.data):
            if args.file:
                # Process specific file
                data_file = os.path.join(args.data, args.file)
                if not os.path.exists(data_file):
                    print(f"File {args.file} not found in {args.data}")
                    return
                
                dfs = [pd.read_csv(data_file)]
                report_name = f"report_{os.path.splitext(args.file)[0]}"
            else:
                # Combine all files
                csv_files = glob.glob(os.path.join(args.data, "*.csv"))
                if not csv_files:
                    print(f"No CSV files found in {args.data}")
                    return
                
                dfs = [pd.read_csv(file) for file in csv_files]
                report_name = f"report_combined_{datetime.now().strftime('%Y%m%d')}"
            
            # Combine dataframes
            df = pd.concat(dfs, ignore_index=True)
        else:
            # Single file
            if not os.path.exists(args.data):
                print(f"Error: Categorized data file {args.data} not found.")
                return
            
            df = pd.read_csv(args.data)
            report_name = f"report_{os.path.splitext(os.path.basename(args.data))[0]}"
        
        # Calculate spending by category
        category_summary = df.groupby('Category')['Amount'].agg(['sum', 'count', 'mean'])
        category_summary = category_summary.sort_values('sum', ascending=False)
        
        # Filter out payment categories
        spending_categories = category_summary[~category_summary.index.str.contains('PAYMENT', case=False)]
        
        # Save the report
        output_path = f"{report_name}.csv" if args.output == 'spending_report.csv' else args.output
        spending_categories.to_csv(output_path)
        print(f"Spending report saved to {output_path}")
        
        # Generate pie chart if matplotlib is available
        try:
            plt.figure(figsize=(12, 8))
            
            # Top 10 categories pie chart
            top_categories = spending_categories.head(10)
            plt.pie(top_categories['sum'], labels=top_categories.index, autopct='%1.1f%%')
            plt.title('Top 10 Spending Categories')
            plt.axis('equal')
            
            plt.tight_layout()
            chart_file = f"{os.path.splitext(output_path)[0]}_chart.png"
            plt.savefig(chart_file)
            print(f"Spending chart saved to {chart_file}")
            
        except ImportError:
            print("Matplotlib not available - skipping chart generation")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 