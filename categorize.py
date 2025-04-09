#!/usr/bin/env python
"""
Command-line interface for categorizing financial transactions
"""

import os
import sys
import argparse
from model_training import train_model
from transaction_categorizer import categorize_transactions
from config import TO_CATEGORIZE_DIR, OUTPUT_DIR, MODELS_DIR

def parse_args():
    parser = argparse.ArgumentParser(description='Categorize financial transactions using ML and LLM')
    
    parser.add_argument('--train', action='store_true',
                        help='Retrain the model before categorizing')
    
    parser.add_argument('--input-dir', type=str, default=TO_CATEGORIZE_DIR,
                        help='Directory containing transaction CSV files to categorize')
    
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help='Directory where categorized files will be saved')
    
    parser.add_argument('--skip-post', action='store_true',
                        help='Skip merchant post-processing')

    parser.add_argument('--force-local', action='store_true',
                        help='Force using local model even if LLM is available')
    
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
    
    # Train model if requested
    if args.train:
        print("\nTraining model...")
        model, _, model_version, _ = train_model()
        if model is None:
            print("Model training failed. Exiting.")
            return 1
        else:
            print(f"Model training successful. New model version: {model_version}")
    
    # Categorize transactions
    print("\nCategorizing transactions...")
    result = categorize_transactions(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        use_postprocessor=(not args.skip_post)
    )
    
    if result:
        print("\nCategorization complete!")
        return 0
    else:
        print("\nCategorization failed.")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 