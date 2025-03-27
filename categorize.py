#!/usr/bin/env python
"""
Convenience script to run transaction categorization with both ML and merchant rules
"""

import os
import sys
import argparse
from improved_categorizer import train_model, categorize_transactions

def parse_args():
    parser = argparse.ArgumentParser(description='Categorize financial transactions using ML and merchant rules')
    
    parser.add_argument('--train', action='store_true',
                        help='Retrain the model before categorizing')
    
    parser.add_argument('--update-merchants', action='store_true',
                        help='Prompt to update merchant patterns during training (implies --train)')
    
    parser.add_argument('--input-dir', type=str, default='data/to_categorize',
                        help='Directory containing transaction CSV files to categorize')
    
    parser.add_argument('--output-dir', type=str, default='data/output',
                        help='Directory where categorized files will be saved')
    
    parser.add_argument('--model-file', type=str, default='improved_model.pkl',
                        help='Path to the model file')
    
    parser.add_argument('--skip-post', action='store_true',
                        help='Skip merchant post-processing')
    
    parser.add_argument('--categorize-only', action='store_true',
                        help='Only categorize, without training (even if --train or --update-merchants are specified)')
    
    parser.add_argument('--update-merchants-only', action='store_true',
                        help='Only update merchant patterns without training or categorizing')
                        
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Financial Transaction Categorizer")
    print("=================================")
    
    # Handle --update-merchants-only flag
    if args.update_merchants_only:
        try:
            import update_merchants
            print("\nRunning merchant pattern update process...")
            update_merchants.interactive_update()
            return 0
        except ImportError:
            print("Error: update_merchants module not available.")
            return 1
        except Exception as e:
            print(f"Error during merchant update process: {e}")
            return 1
    
    # Check directories exist
    if not args.categorize_only and not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return 1
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Update the --train flag if --update-merchants is specified
    if args.update_merchants:
        args.train = True
    
    # Train or check model
    model = None
    if args.train and not args.categorize_only:
        print("\nTraining model...")
        model, _ = train_model(update_merchants=args.update_merchants)
        if model is None:
            print("Model training failed. Exiting.")
            return 1
    elif not args.categorize_only and not os.path.exists(args.model_file):
        print(f"\nModel file '{args.model_file}' not found. Use --train to train a new model.")
        return 1
    
    # Categorize transactions
    if not args.update_merchants_only:
        print("\nCategorizing transactions...")
        categorize_transactions(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_file=args.model_file,
            use_postprocessor=(not args.skip_post)
        )
    
    print("\nProcess complete!")
    return 0

if __name__ == '__main__':
    sys.exit(main()) 