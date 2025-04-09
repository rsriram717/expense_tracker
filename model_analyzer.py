from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

def load_results():
    """Load the most recent model comparison results"""
    # Find the most recent comparison file
    comparison_files = [f for f in os.listdir() if f.startswith('model_comparison_') and f.endswith('.csv')]
    if not comparison_files:
        print("No comparison results found")
        return None
    
    # Get the most recent file
    latest_file = max(comparison_files)
    print(f"Loading results from {latest_file}")
    
    # Load the data
    results = pd.read_csv(latest_file)
    print(f"Loaded {len(results)} comparison records")
    return results

def analyze_llama_performance(results):
    """Analyze why the Llama model might be performing poorly"""
    if results is None or len(results) == 0:
        return
        
    print("\n===== Llama Model Performance Analysis =====")
    
    # 1. Basic accuracy metrics
    llama_accuracy = results['llama_correct'].mean()
    print(f"Overall Llama Accuracy: {llama_accuracy:.2%}")
    
    # 2. Count of unique true categories vs Llama categories
    true_categories = set(results['true_category'])
    llama_categories = set(results['llama_prediction'])
    print(f"Unique True Categories: {len(true_categories)}")
    print(f"Unique Llama Categories: {len(llama_categories)}")
    
    # 3. Confusion matrix (simplified as counts)
    confusion = {}
    for _, row in results.iterrows():
        true_cat = row['true_category']
        pred_cat = row['llama_prediction']
        if true_cat not in confusion:
            confusion[true_cat] = Counter()
        confusion[true_cat][pred_cat] += 1
    
    # 4. Print most common confusions
    print("\nTop confusions (true -> predicted):")
    all_confusions = []
    for true_cat, pred_counts in confusion.items():
        for pred_cat, count in pred_counts.items():
            if true_cat != pred_cat:  # Only include actual confusion
                all_confusions.append((true_cat, pred_cat, count))
    
    # Sort by count (highest first)
    all_confusions.sort(key=lambda x: x[2], reverse=True)
    for true_cat, pred_cat, count in all_confusions[:15]:  # Show top 15
        print(f"  {true_cat} -> {pred_cat}: {count} times")
    
    # 5. Analyze if Llama is biased toward certain categories
    print("\nLlama category usage vs. actual distribution:")
    true_counts = results['true_category'].value_counts()
    llama_counts = results['llama_prediction'].value_counts()
    
    # Combine and sort by largest discrepancy
    cat_comparison = []
    all_cats = set(true_counts.index) | set(llama_counts.index)
    for cat in all_cats:
        true_count = true_counts.get(cat, 0)
        llama_count = llama_counts.get(cat, 0)
        difference = llama_count - true_count
        cat_comparison.append((cat, true_count, llama_count, difference))
    
    cat_comparison.sort(key=lambda x: abs(x[3]), reverse=True)
    for cat, true_count, llama_count, diff in cat_comparison:
        print(f"  {cat}: True={true_count}, Llama={llama_count}, Diff={diff:+d}")
    
    # 6. Check for potential category matching issues
    print("\nPotential category matching issues:")
    true_cats_lower = [cat.lower() for cat in true_categories]
    llama_cats_lower = [cat.lower() for cat in llama_categories]
    
    # Check for categories that might be synonyms or subsets
    for true_cat in sorted(true_categories):
        true_cat_lower = true_cat.lower()
        for llama_cat in sorted(llama_categories):
            llama_cat_lower = llama_cat.lower()
            
            # Skip exact matches
            if true_cat_lower == llama_cat_lower:
                continue
                
            # Check for substring matches or similarity
            if (true_cat_lower in llama_cat_lower or 
                llama_cat_lower in true_cat_lower or
                similar_terms(true_cat_lower, llama_cat_lower)):
                print(f"  Potential match confusion: '{true_cat}' vs '{llama_cat}'")
    
    # 7. Check for transaction text patterns that may be causing problems
    print("\nAnalyzing transaction descriptions for patterns:")
    
    # For correctly classified entries
    correct_descriptions = results[results['llama_correct']]['description'].tolist()
    incorrect_descriptions = results[~results['llama_correct']]['description'].tolist()
    
    # Look for patterns in length
    correct_lengths = [len(desc) for desc in correct_descriptions if isinstance(desc, str)]
    incorrect_lengths = [len(desc) for desc in incorrect_descriptions if isinstance(desc, str)]
    
    avg_correct_len = np.mean(correct_lengths) if correct_lengths else 0
    avg_incorrect_len = np.mean(incorrect_lengths) if incorrect_lengths else 0
    
    print(f"  Average length of correctly classified descriptions: {avg_correct_len:.1f} chars")
    print(f"  Average length of incorrectly classified descriptions: {avg_incorrect_len:.1f} chars")
    
    # 8. Check for common words/terms in correct vs incorrect predictions
    print("\nCommon words/patterns in correct predictions:")
    correct_words = extract_common_words(correct_descriptions, top=15)
    for word, count in correct_words:
        print(f"  '{word}': {count} occurrences")
    
    print("\nCommon words/patterns in incorrect predictions:")
    incorrect_words = extract_common_words(incorrect_descriptions, top=15)
    for word, count in incorrect_words:
        print(f"  '{word}': {count} occurrences")
    
    # 9. Hypotheses for why Llama is performing poorly
    print("\nHypotheses for Llama's underperformance:")
    
    # Category mismatch hypothesis
    if len(true_categories - llama_categories) > 0:
        print("  1. Category mismatch: Llama is not recognizing these categories:")
        for cat in sorted(true_categories - llama_categories):
            print(f"     - {cat}")
    
    # Bias hypothesis
    most_overused = sorted(cat_comparison, key=lambda x: x[3], reverse=True)[:3]
    if most_overused and most_overused[0][3] > 10:  # If difference > 10
        print("  2. Category bias: Llama is overusing these categories:")
        for cat, _, _, diff in most_overused:
            if diff > 0:
                print(f"     - {cat}: +{diff} instances")
    
    # Context hypothesis
    print("  3. Contextual understanding: Llama may not understand transaction context")
    print("     - The model might not recognize specific store names or transaction patterns")
    print("     - It might be missing domain-specific knowledge about financial transactions")
    
    # Training examples hypothesis
    print("  4. Prompt format: The batch processing may not give enough context per transaction")
    print("     - Single transaction processing might yield better results than batch processing")
    
    # 10. Recommendations
    print("\nRecommendations to improve Llama performance:")
    print("  1. Fine-tune Llama with category examples from your specific transaction domain")
    print("  2. Ensure category definitions between RF model and Llama prompt are identical")
    print("  3. Try processing single transactions with more context rather than batches")
    print("  4. Provide clearer examples in the system prompt for each category")
    print("  5. Add transaction amount and other metadata to help categorization")
    
    # Save the analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"llama_analysis_{timestamp}.txt", "w") as f:
        f.write("===== Llama Model Performance Analysis =====\n")
        f.write(f"Overall Llama Accuracy: {llama_accuracy:.2%}\n\n")
        f.write(f"Most common confusions:\n")
        for true_cat, pred_cat, count in all_confusions[:10]:
            f.write(f"  {true_cat} -> {pred_cat}: {count} times\n")
        f.write("\nRecommendations:\n")
        f.write("  1. Fine-tune with domain-specific examples\n")
        f.write("  2. Ensure category alignment between models\n")
        f.write("  3. Try single transaction processing\n")
        f.write("  4. Provide clearer examples in the prompt\n")
        f.write("  5. Include more transaction context\n")
    
    print(f"\nAnalysis saved to llama_analysis_{timestamp}.txt")

def similar_terms(term1, term2):
    """Check if two terms might be similar or related"""
    # Split into words
    words1 = set(term1.lower().split())
    words2 = set(term2.lower().split())
    
    # Check for common words
    common_words = words1.intersection(words2)
    if common_words:
        return True
    
    # Check for word pairs that might be related
    related_pairs = [
        ('food', 'restaurants'), 
        ('food', 'dining'),
        ('dining', 'restaurants'),
        ('travel', 'airline'),
        ('travel', 'lodging'),
        ('travel', 'hotel'),
        ('shop', 'shopping'),
        ('entertainment', 'movie'),
        ('entertainment', 'theater'),
        ('subscription', 'service'),
        ('subscription', 'monthly'),
        ('medical', 'health'),
        ('transport', 'uber'),
        ('transport', 'lyft'),
        ('groceries', 'food')
    ]
    
    for w1 in words1:
        for w2 in words2:
            if (w1, w2) in related_pairs or (w2, w1) in related_pairs:
                return True
    
    return False

def extract_common_words(descriptions, top=10):
    """Extract common words from a list of descriptions"""
    words_count = Counter()
    
    for desc in descriptions:
        if not isinstance(desc, str):
            continue
            
        # Convert to lowercase and split into words
        words = re.findall(r'\b[A-Za-z]{3,}\b', desc.lower())
        words_count.update(words)
    
    # Filter out common stop words
    stop_words = {'the', 'and', 'for', 'from', 'with', 'this', 'that', 'was', 'inc'}
    filtered_words = [(word, count) for word, count in words_count.items() if word not in stop_words]
    
    # Return top N words
    return sorted(filtered_words, key=lambda x: x[1], reverse=True)[:top]

if __name__ == "__main__":
    print("Starting Llama model performance analysis...")
    results = load_results()
    if results is not None:
        analyze_llama_performance(results) 