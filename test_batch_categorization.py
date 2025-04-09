from dotenv import load_dotenv
import os
import pandas as pd
import time
from improved_categorizer import get_llama_categories_batch, test_llama_connection
import random

# Load environment variables from .env file
load_dotenv()

def generate_test_transactions(count=50):
    """Generate sample transaction data for testing batch processing"""
    # Sample transaction descriptions
    descriptions = [
        "UBER EATS",
        "AMAZON PRIME",
        "NETFLIX SUBSCRIPTION",
        "WALMART GROCERIES",
        "TARGET PURCHASE",
        "SHELL GAS STATION",
        "CHEESECAKE FACTORY",
        "UNITED AIRLINES TICKET",
        "HOTEL RESERVATION",
        "STARBUCKS COFFEE",
        "HOME DEPOT PURCHASE",
        "PET SUPPLIES PLUS",
        "VERIZON WIRELESS BILL",
        "CVS PHARMACY",
        "AMC THEATERS",
        "SUBWAY SANDWICH",
        "LYFT RIDE",
        "SPOTIFY PREMIUM",
        "COSTCO WHOLESALE",
        "CHASE MORTGAGE PAYMENT"
    ]
    
    # Extended details for variety
    details = [
        "Monthly subscription",
        "Online purchase",
        "Food delivery",
        "Gas station fill-up",
        "Travel expense",
        "Household items",
        "Bill payment",
        "Entertainment",
        "Dining out",
        "No details available"
    ]
    
    # Generate random combinations
    transactions = []
    for _ in range(count):
        desc = random.choice(descriptions)
        detail = random.choice(details)
        transactions.append((desc, detail))
    
    return transactions

def test_batches():
    """Test the batch categorization with different batch sizes"""
    # First ensure connection works
    if not test_llama_connection():
        print("Failed to connect to Llama API. Exiting.")
        return
    
    print("\n===== TESTING BATCH CATEGORIZATION =====")
    
    # First test with 50 transactions in batches of 10
    print("\n--- Testing with 50 transactions in batches of 10 ---")
    test_data = generate_test_transactions(50)
    
    # Process in batches of 10
    batch_size = 10
    start_time = time.time()
    results = []
    
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(test_data) + batch_size - 1)//batch_size}")
        batch_results = get_llama_categories_batch(batch, batch_size=len(batch))
        
        # Show sample results
        print(f"Sample results from batch:")
        for j in range(min(3, len(batch))):
            print(f"  Transaction: {batch[j][0]} | {batch[j][1]} → Category: {batch_results[j]}")
        
        results.extend(batch_results)
        
        # Small delay between batches
        if i + batch_size < len(test_data):
            print("Waiting 1 second between batches...")
            time.sleep(1)
    
    total_time = time.time() - start_time
    print(f"\nProcessed 50 transactions in batches of 10 in {total_time:.2f} seconds")
    print(f"Average time per transaction: {total_time/50:.4f} seconds")
    print(f"Average time per batch: {total_time/(len(test_data)/batch_size):.2f} seconds")
    
    # Now we'll check if larger batches (50) would work
    print("\n--- Testing with a batch of 50 ---")
    print("If this works well, we won't proceed to the full 200 transaction test.")
    
    # Take the same 50 transactions but process as a single batch
    large_batch_size = 50
    large_batch = test_data[:large_batch_size]
    
    start_time = time.time()
    print(f"Processing single batch of {large_batch_size} transactions...")
    
    try:
        large_batch_results = get_llama_categories_batch(large_batch, batch_size=large_batch_size)
        large_batch_time = time.time() - start_time
        
        print(f"Successfully processed batch of {large_batch_size} in {large_batch_time:.2f} seconds")
        print(f"Average time per transaction: {large_batch_time/large_batch_size:.4f} seconds")
        
        # Check if all transactions were categorized
        if len(large_batch_results) == large_batch_size:
            print(f"All {large_batch_size} transactions were successfully categorized")
            
            # Show sample results
            print(f"Sample results from large batch:")
            for j in range(min(5, len(large_batch))):
                print(f"  Transaction: {large_batch[j][0]} | {large_batch[j][1]} → Category: {large_batch_results[j]}")
                
            print("\nLarge batch processing works well. No need to test with 200 transactions.")
        else:
            print(f"Warning: Expected {large_batch_size} results but got {len(large_batch_results)}")
    
    except Exception as e:
        print(f"Error processing large batch: {e}")
        print("Large batch processing failed. We'll continue with smaller batches.")

if __name__ == "__main__":
    test_batches() 