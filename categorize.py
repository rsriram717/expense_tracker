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
    
    parser.add_argument('--input-dir', type=str, default='data/to_categorize',
                        help='Directory containing transaction CSV files to categorize')
    
    parser.add_argument('--output-dir', type=str, default='data/output',
                        help='Directory where categorized files will be saved')
    
    parser.add_argument('--model-file', type=str, default='improved_model.pkl',
                        help='Path to the model file')
    
    parser.add_argument('--skip-post', action='store_true',
                        help='Skip merchant post-processing')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Financial Transaction Categorizer")
    print("=================================")
    
    # Check directories exist
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return 1
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train or check model
    model = None
    if args.train:
        print("\nTraining model...")
        model, _ = train_model()
        if model is None:
            print("Model training failed. Exiting.")
            return 1
    elif not os.path.exists(args.model_file):
        print(f"\nModel file '{args.model_file}' not found. Use --train to train a new model.")
        return 1
    
    # Categorize transactions
    print("\nCategorizing transactions...")
    categorize_transactions(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_file=args.model_file,
        use_postprocessor=(not args.skip_post)
    )
    
    print("\nCategorization complete!")
    return 0

if __name__ == '__main__':
    sys.exit(main()) 