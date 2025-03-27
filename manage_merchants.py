#!/usr/bin/env python
"""
Script to manage merchant patterns for transaction categorization
"""

import argparse
import pandas as pd
import os
from merchant_postprocessor import MerchantPostProcessor

def list_merchants(processor, args):
    """List all merchant patterns"""
    if len(processor.merchants) == 0:
        print("No merchant patterns found.")
        return
    
    # Convert to DataFrame for nice display
    df = processor.merchants.copy()
    
    # Sort by category then merchant
    df = df.sort_values(by=['category', 'merchant_pattern'])
    
    # Filter by category if specified
    if args.category:
        df = df[df['category'].str.lower() == args.category.lower()]
        if len(df) == 0:
            print(f"No merchant patterns found for category '{args.category}'.")
            return
    
    # Print header
    print(f"\nFound {len(df)} merchant patterns:")
    print("-" * 80)
    print(f"{'Merchant Pattern':<40} {'Category':<20} {'Confidence':<10}")
    print("-" * 80)
    
    # Print each merchant pattern
    for _, row in df.iterrows():
        print(f"{row['merchant_pattern']:<40} {row['category']:<20} {row['confidence']:<10.2f}")

def add_merchant(processor, args):
    """Add a new merchant pattern"""
    # Check required arguments
    if not args.pattern or not args.category:
        print("Error: Both --pattern and --category are required for adding a merchant pattern.")
        return
    
    # Set confidence
    confidence = args.confidence if args.confidence is not None else 0.9
    
    # Validate confidence range
    if confidence < 0 or confidence > 1:
        print("Error: Confidence must be between 0 and 1.")
        return
    
    # Add the pattern
    success = processor.add_merchant_pattern(args.pattern, args.category, confidence)
    
    if success:
        print(f"Successfully added merchant pattern: {args.pattern} -> {args.category} (confidence: {confidence})")
    else:
        print("Failed to add merchant pattern.")

def remove_merchant(processor, args):
    """Remove a merchant pattern"""
    # Check required arguments
    if not args.pattern:
        print("Error: --pattern is required for removing a merchant pattern.")
        return
    
    # Remove the pattern
    success = processor.remove_merchant_pattern(args.pattern)
    
    if success:
        print(f"Successfully removed merchant pattern: {args.pattern}")
    else:
        print(f"Failed to remove merchant pattern: {args.pattern}")

def export_merchants(processor, args):
    """Export merchant patterns to a CSV file"""
    # Check required arguments
    if not args.output:
        print("Error: --output is required for exporting merchant patterns.")
        return
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Export to CSV
    processor.merchants.to_csv(args.output, index=False)
    print(f"Successfully exported {len(processor.merchants)} merchant patterns to {args.output}")

def import_merchants(processor, args):
    """Import merchant patterns from a CSV file"""
    # Check required arguments
    if not args.input:
        print("Error: --input is required for importing merchant patterns.")
        return
    
    # Check if file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        return
    
    try:
        # Read CSV file
        df = pd.read_csv(args.input)
        
        # Check required columns
        required_cols = ['merchant_pattern', 'category', 'confidence']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns in import file: {missing_cols}")
            return
        
        # Save to merchant file
        df.to_csv(processor.merchant_file, index=False)
        processor.reload_merchants()
        
        print(f"Successfully imported {len(df)} merchant patterns from {args.input}")
    except Exception as e:
        print(f"Error importing merchant patterns: {e}")

def list_categories(processor, args):
    """List all unique categories from merchant patterns"""
    if len(processor.merchants) == 0:
        print("No merchant patterns found.")
        return
    
    # Get unique categories
    categories = processor.merchants['category'].unique()
    categories.sort()
    
    # Print categories
    print(f"\nFound {len(categories)} unique categories:")
    print("-" * 40)
    for category in categories:
        count = len(processor.merchants[processor.merchants['category'] == category])
        print(f"{category:<30} ({count} patterns)")

def main():
    parser = argparse.ArgumentParser(description='Manage merchant patterns for transaction categorization')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all merchant patterns')
    list_parser.add_argument('--category', type=str, help='Filter by category')
    
    # Add command
    add_parser = subparsers.add_parser('add', help='Add a new merchant pattern')
    add_parser.add_argument('--pattern', type=str, help='Merchant pattern to match')
    add_parser.add_argument('--category', type=str, help='Category to assign')
    add_parser.add_argument('--confidence', type=float, help='Confidence level (0-1)')
    
    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove a merchant pattern')
    remove_parser.add_argument('--pattern', type=str, help='Merchant pattern to remove')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export merchant patterns to a CSV file')
    export_parser.add_argument('--output', type=str, help='Output file path')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import merchant patterns from a CSV file')
    import_parser.add_argument('--input', type=str, help='Input file path')
    
    # Categories command
    categories_parser = subparsers.add_parser('categories', help='List all unique categories')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create processor
    processor = MerchantPostProcessor()
    
    # Run command
    if args.command == 'list':
        list_merchants(processor, args)
    elif args.command == 'add':
        add_merchant(processor, args)
    elif args.command == 'remove':
        remove_merchant(processor, args)
    elif args.command == 'export':
        export_merchants(processor, args)
    elif args.command == 'import':
        import_merchants(processor, args)
    elif args.command == 'categories':
        list_categories(processor, args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 