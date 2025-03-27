#!/usr/bin/env python
"""
Script to compare the results of transaction categorization with and without merchant post-processing
"""

import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from improved_categorizer import categorize_transactions

def compare_categorization(input_file, output_dir='data/output', model_file='improved_model.pkl'):
    """
    Compare categorization with and without merchant post-processing
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # File paths for results
    filename = os.path.basename(input_file)
    with_post_output = os.path.join(output_dir, f"with_post_{filename}")
    without_post_output = os.path.join(output_dir, f"without_post_{filename}")
    
    # Remove existing files to avoid conflicts
    for file_path in [with_post_output, without_post_output]:
        if os.path.exists(file_path):
            os.remove(file_path)
    
    # Run categorization without post-processing
    print("\nRunning categorization WITHOUT merchant post-processing...")
    categorize_transactions(
        input_dir=os.path.dirname(input_file),
        output_dir=output_dir,
        model_file=model_file,
        use_postprocessor=False
    )
    
    # Rename the output file
    standard_output = os.path.join(output_dir, f"improved_categorized_{filename}")
    if os.path.exists(standard_output):
        os.rename(standard_output, without_post_output)
    
    # Run categorization with post-processing
    print("\nRunning categorization WITH merchant post-processing...")
    categorize_transactions(
        input_dir=os.path.dirname(input_file),
        output_dir=output_dir,
        model_file=model_file,
        use_postprocessor=True
    )
    
    # Rename the output file
    if os.path.exists(standard_output):
        os.rename(standard_output, with_post_output)
    
    # Load results
    if not os.path.exists(with_post_output) or not os.path.exists(without_post_output):
        print("Error: One or more output files were not created.")
        return
    
    with_post = pd.read_csv(with_post_output)
    without_post = pd.read_csv(without_post_output)
    
    # Compare results
    compare_and_report(without_post, with_post)

def compare_and_report(without_post, with_post):
    """
    Compare and report the differences between the two sets of results
    """
    # Check if dataframes have the same number of rows
    if len(without_post) != len(with_post):
        print("Error: The two result sets have different numbers of transactions.")
        return
    
    # Count differences
    diff_count = 0
    diff_details = []
    
    for i in range(len(without_post)):
        if without_post.iloc[i]['Category'] != with_post.iloc[i]['Category']:
            diff_count += 1
            diff_details.append({
                'Description': with_post.iloc[i]['Description'],
                'Original Category': without_post.iloc[i]['Category'],
                'Original Confidence': without_post.iloc[i]['Confidence'],
                'New Category': with_post.iloc[i]['Category'],
                'New Confidence': with_post.iloc[i]['Confidence']
            })
    
    # Print summary
    print("\n----- Comparison Results -----")
    print(f"Total transactions: {len(with_post)}")
    print(f"Transactions with different categories: {diff_count} ({diff_count/len(with_post)*100:.1f}%)")
    
    # Confidence comparison
    without_conf_avg = without_post['Confidence'].mean()
    with_conf_avg = with_post['Confidence'].mean()
    
    without_low_conf = len(without_post[without_post['Confidence'] < 0.7])
    with_low_conf = len(with_post[with_post['Confidence'] < 0.7])
    
    print(f"\nConfidence metrics:")
    print(f"  Average confidence WITHOUT post-processing: {without_conf_avg:.2f}")
    print(f"  Average confidence WITH post-processing: {with_conf_avg:.2f}")
    print(f"  Change in average confidence: {with_conf_avg - without_conf_avg:+.2f}")
    
    print(f"\n  Low confidence transactions (<0.7) WITHOUT post-processing: {without_low_conf} ({without_low_conf/len(without_post)*100:.1f}%)")
    print(f"  Low confidence transactions (<0.7) WITH post-processing: {with_low_conf} ({with_low_conf/len(with_post)*100:.1f}%)")
    print(f"  Change in low confidence transactions: {with_low_conf - without_low_conf:+d} ({(with_low_conf - without_low_conf)/len(with_post)*100:+.1f}%)")
    
    # Category distribution comparison
    without_cats = without_post['Category'].value_counts()
    with_cats = with_post['Category'].value_counts()
    
    print("\nCategory distribution changes:")
    all_cats = sorted(list(set(without_cats.index) | set(with_cats.index)))
    
    for cat in all_cats:
        without_count = without_cats.get(cat, 0)
        with_count = with_cats.get(cat, 0)
        diff = with_count - without_count
        
        if diff != 0:
            print(f"  {cat}: {without_count} -> {with_count} ({diff:+d})")
    
    # Print transactions with changed categories
    if diff_details:
        print("\nTransactions with changed categories:")
        for i, detail in enumerate(diff_details):
            print(f"  {i+1}. {detail['Description']}")
            print(f"     {detail['Original Category']} ({detail['Original Confidence']:.2f}) -> {detail['New Category']} ({detail['New Confidence']:.2f})")
    
    # Create visualization
    try:
        create_comparison_charts(without_post, with_post, diff_details)
    except Exception as e:
        print(f"Error creating charts: {e}")

def create_comparison_charts(without_post, with_post, diff_details):
    """
    Create visualization of the comparison
    """
    # Only create charts if matplotlib is available
    if not plt:
        return
    
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Confidence distribution comparison
    ax1.hist([without_post['Confidence'], with_post['Confidence']], 
             bins=10, range=(0, 1), alpha=0.7, 
             label=['Without Post-Processing', 'With Post-Processing'])
    ax1.set_title('Confidence Distribution Comparison')
    ax1.set_xlabel('Confidence Score')
    ax1.set_ylabel('Number of Transactions')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Category distribution comparison
    without_cats = without_post['Category'].value_counts()
    with_cats = with_post['Category'].value_counts()
    
    # Get the union of all categories
    all_cats = sorted(list(set(without_cats.index) | set(with_cats.index)))
    
    # Prepare data for plotting
    without_vals = [without_cats.get(cat, 0) for cat in all_cats]
    with_vals = [with_cats.get(cat, 0) for cat in all_cats]
    
    # Plotting
    x = range(len(all_cats))
    width = 0.35
    
    ax2.bar([i - width/2 for i in x], without_vals, width, alpha=0.7, label='Without Post-Processing')
    ax2.bar([i + width/2 for i in x], with_vals, width, alpha=0.7, label='With Post-Processing')
    
    ax2.set_title('Category Distribution Comparison')
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Number of Transactions')
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_cats, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_results.png')
    print("\nVisualization saved to 'comparison_results.png'")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Compare transaction categorization with and without merchant post-processing')
    parser.add_argument('--input', type=str, required=True, help='Input transaction file to categorize')
    parser.add_argument('--output-dir', type=str, default='data/output', help='Directory for output files')
    parser.add_argument('--model-file', type=str, default='improved_model.pkl', help='Model file to use')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        return 1
    
    # Run comparison
    compare_categorization(args.input, args.output_dir, args.model_file)
    return 0

if __name__ == '__main__':
    import sys
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available. Charts will not be generated.")
        plt = None
    
    sys.exit(main()) 