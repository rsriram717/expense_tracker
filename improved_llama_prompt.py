from dotenv import load_dotenv
import os
import pandas as pd
import time
from datetime import datetime
import random
from improved_categorizer import (
    test_llama_connection,
    PREDEFINED_CATEGORIES,
    client
)
from db_connector import get_engine
import openai

# Import prompts from the separate file
from llm_prompts import (
    CATEGORY_EXAMPLES,
    SYSTEM_PROMPT,
    CONNECTION_TEST_PROMPT,
    generate_improved_prompt,
    generate_categorization_prompt,
    generate_single_transaction_prompt,
    parse_batch_response
)

# Load environment variables from .env file
load_dotenv()

# The CATEGORY_EXAMPLES dictionary has been moved to llm_prompts.py

# The generate_improved_prompt function has been moved to llm_prompts.py

def test_improved_prompt():
    """Test the improved Llama prompt on sample transactions"""
    # First ensure connection works
    if not test_llama_connection():
        print("Failed to connect to Llama API. Exiting.")
        return
    
    print("\n===== TESTING IMPROVED PROMPT =====")
    
    # Load a sample of transactions from the most recent comparison file
    comparison_files = [f for f in os.listdir() if f.startswith('model_comparison_') and f.endswith('.csv')]
    if not comparison_files:
        print("No comparison results found. Generating synthetic data instead.")
        test_data = generate_synthetic_transactions(20)
    else:
        # Get the most recent file
        latest_file = max(comparison_files)
        print(f"Loading sample transactions from {latest_file}")
        
        # Load data and select transactions that were incorrectly categorized
        results = pd.read_csv(latest_file)
        
        # Find incorrectly categorized transactions
        incorrect = results[~results['llama_correct']]
        if len(incorrect) > 0:
            # Select a sample of 20 transactions or all if fewer
            sample_size = min(20, len(incorrect))
            sample = incorrect.sample(sample_size)
            
            # Create transaction tuples
            test_data = []
            for _, row in sample.iterrows():
                test_data.append((row['description'], None))
                
            print(f"Selected {len(test_data)} previously misclassified transactions for testing")
        else:
            print("No incorrectly classified transactions found. Generating synthetic data.")
            test_data = generate_synthetic_transactions(20)
    
    # Process the test data in a batch
    batch_size = 10
    start_time = time.time()
    all_predictions = []
    
    for i in range(0, len(test_data), batch_size):
        batch_end = min(i + batch_size, len(test_data))
        batch = test_data[i:batch_end]
        print(f"\nProcessing batch {i//batch_size + 1} ({len(batch)} transactions)")
        
        # Generate the improved prompt
        improved_prompt = generate_improved_prompt(batch)
        
        # Print a shortened version of the prompt for debugging
        prompt_preview = '\n'.join(improved_prompt.split('\n')[:10]) + '\n... [truncated]'
        print(f"Using improved prompt:\n{prompt_preview}")
        
        # Call the API
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-3.1-8b-instruct/fp-8",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": improved_prompt}
                ],
                max_tokens=500,
                temperature=0.0
            )
            
            raw_response = response.choices[0].message.content.strip()
            print(f"API response received")
            
            # Parse the response
            predictions = parse_batch_response(raw_response, len(batch))
            all_predictions.extend(predictions)
            
            # Display sample results
            print(f"\nSample results from batch:")
            for j in range(min(5, len(batch))):
                print(f"  Transaction: {batch[j][0]} → Category: {predictions[j]}")
                
            # Add a delay between batches
            if batch_end < len(test_data):
                print("Waiting 1 second between batches...")
                time.sleep(1)
                
        except Exception as e:
            print(f"Error with API call: {e}")
            continue
    
    total_time = time.time() - start_time
    print(f"\nProcessed {len(test_data)} transactions in {total_time:.2f} seconds")
    
    # If we have true categories, compare the results
    if 'sample' in locals() and all(pd.notna(sample['true_category'])):
        print("\n--- Comparing with true categories ---")
        correct_count = 0
        
        for i, (pred, (_, row)) in enumerate(zip(all_predictions, sample.iterrows())):
            true_cat = row['true_category']
            if pred == true_cat:
                correct_count += 1
                print(f"✓ CORRECT: '{row['description']}' → {pred}")
            else:
                print(f"✗ INCORRECT: '{row['description']}' → Predicted: {pred}, True: {true_cat}")
                
        accuracy = correct_count / len(all_predictions) if all_predictions else 0
        print(f"\nImproved prompt accuracy: {accuracy:.2%} ({correct_count}/{len(all_predictions)})")
        
        # Compare with previous results if available
        if 'llama_correct' in sample.columns:
            prev_correct = sample['llama_correct'].sum()
            prev_accuracy = prev_correct / len(sample)
            print(f"Previous Llama accuracy on same transactions: {prev_accuracy:.2%} ({prev_correct}/{len(sample)})")
            
            if accuracy > prev_accuracy:
                improvement = (accuracy - prev_accuracy) / prev_accuracy * 100
                print(f"Improvement: +{improvement:.1f}%")
            else:
                decline = (prev_accuracy - accuracy) / prev_accuracy * 100
                print(f"Decline: -{decline:.1f}%")
    
    return all_predictions

# The parse_batch_response function has been moved to llm_prompts.py

def generate_synthetic_transactions(count=20):
    """Generate synthetic transaction data for testing"""
    sample_transactions = [
        "UBER EATS",
        "NETFLIX MONTHLY SUBSCRIPTION",
        "TRADER JOE'S #123",
        "SHELL OIL 1234",
        "CVS PHARMACY",
        "AMERICAN AIRLINES",
        "MARRIOTT HOTEL CHICAGO",
        "STARBUCKS COFFEE",
        "AMAZON PRIME",
        "DOORDASH",
        "TARGET",
        "UBER RIDE",
        "COMCAST CABLE",
        "PLANET FITNESS",
        "APPLE STORE",
        "DELTA AIRLINES",
        "MCDONALD'S",
        "WHOLE FOODS",
        "CHASE MORTGAGE PAYMENT",
        "VENMO PAYMENT RECEIVED"
    ]
    
    # Generate random transactions
    transactions = []
    for _ in range(count):
        desc = random.choice(sample_transactions)
        transactions.append((desc, None))
    
    return transactions

if __name__ == "__main__":
    print("Testing improved Llama prompt...")
    test_improved_prompt() 