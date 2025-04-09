from dotenv import load_dotenv
import os
import time
from improved_categorizer import process_file_with_llama, test_llama_connection

# Load environment variables from .env file
load_dotenv()

def test_file_categorization():
    """Test processing a real transaction file with batch categorization"""
    # First ensure connection works
    if not test_llama_connection():
        print("Failed to connect to Llama API. Exiting.")
        return
    
    print("\n===== TESTING FILE BATCH CATEGORIZATION =====")
    
    # Path to input file
    input_file = "data/to_categorize/transactions_to_cat.csv"
    output_file = "data/output/test_batch_categorized_output.csv"
    
    # Process the file
    print(f"\nProcessing file: {input_file}")
    start_time = time.time()
    
    # Call the processing function (without a processor for simplicity)
    success = process_file_with_llama(input_file, output_file, processor=None)
    
    if success:
        total_time = time.time() - start_time
        print(f"\nTotal processing time: {total_time:.2f} seconds")
        print(f"Output saved to: {output_file}")
        print("\nBatch processing works successfully with size 50!")
    else:
        print("\nError processing file.")

if __name__ == "__main__":
    test_file_categorization() 