"""
LLM Service Module for Financial Transaction Categorization

This module handles all interactions with the LLM (Llama) API, including:
- Connection testing
- Prompt generation
- Response parsing
- Batch processing
"""

import os
import time
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
import openai
from openai import OpenAI, AsyncOpenAI

from config import (
    INFERENCE_API_KEY,
    INFERENCE_BASE_URL,
    LLAMA_MODEL_NAME,
    PREDEFINED_CATEGORIES,
    CATEGORY_EXAMPLES
)

# --- Initialize OpenAI Client ---
client = None
async_client = None

if INFERENCE_API_KEY:
    try:
        # Synchronous client (for potential other uses)
        client = OpenAI(
            api_key=INFERENCE_API_KEY,
            base_url=INFERENCE_BASE_URL,
        )
        print("OpenAI client initialized successfully.")

        # Asynchronous client (for concurrent calls)
        async_client = AsyncOpenAI(
            api_key=INFERENCE_API_KEY,
            base_url=INFERENCE_BASE_URL,
        )
        print("Async OpenAI client initialized successfully.")

    except Exception as e:
        print(f"Error initializing OpenAI clients: {e}")
else:
    print(f"Warning: Inference API key not set. LLM features will be unavailable.")

# --- Prompt Templates ---
SYSTEM_PROMPT = """You are a financial transaction categorizer that precisely matches transactions to predefined categories."""

CONNECTION_TEST_PROMPT = "What is 2+2? Respond with only the numerical answer."

def generate_categorization_prompt(
    transactions: List[Tuple[str, Optional[str]]],
    category_list: Optional[List[str]] = None
) -> str:
    """Generate a prompt for categorizing financial transactions
    
    Args:
        transactions: List of (description, extended_details) tuples
        category_list: Optional list of categories to use (defaults to PREDEFINED_CATEGORIES)
        
    Returns:
        str: Formatted prompt for the LLM
    """
    if category_list is None:
        category_list = PREDEFINED_CATEGORIES
    
    category_examples_str = ""
    
    # Add examples for each category to provide context
    for category in category_list:
        if category in CATEGORY_EXAMPLES:
            examples = CATEGORY_EXAMPLES[category]
            examples_str = ", ".join([f'"{ex}"' for ex in examples[:3]])  # Use first 3 examples
            category_examples_str += f"- {category}: Includes transactions like {examples_str}\n"
    
    # Format transaction string with numbers for easy reference
    transactions_str = ""
    for i, details in enumerate(transactions):
        description = details[0]
        extended_details = details[1] if len(details) > 1 else None
        
        context = str(description if description is not None else "")
        if extended_details and pd.notna(extended_details) and str(extended_details).strip() != "":
            context += f" | Additional Info: {extended_details}"
            
        transactions_str += f"Transaction {i+1}: \"{context}\"\n"
    
    # Create prompt with clear instructions and examples
    prompt = f"""You are a financial transaction categorizer. Your task is to analyze each transaction description and assign 
the most appropriate category from the following list:
{", ".join(category_list)}

Here's what each category typically includes:
{category_examples_str}

Category clarifications:
- "Food & Drink" is for restaurants, cafes, bars, food delivery services like Uber Eats, and all dining out
- "Groceries" is specifically for grocery store purchases of food items to prepare at home
- "Transportation" includes rideshare (Uber, Lyft), public transit, gas stations, and vehicle expenses
- "Home" includes furniture, home improvement, rent payments, and household supplies
- "Subscriptions" covers recurring digital services like Netflix, Spotify, etc.
- "Travel-Airline" is specifically for airline tickets and related fees
- "Travel-Lodging" is specifically for hotels, Airbnb, and accommodations
- "Dante" is the category for pet-related expenses (pet supplies, vet visits, etc.)

Please categorize each of the following {len(transactions)} transactions:
{transactions_str}

For each transaction, respond ONLY with the exact category name from the provided list.
Format your response as follows:
Transaction 1: [Category]
Transaction 2: [Category]
...and so on.

Use "Misc" only when a transaction truly doesn't fit into any other category.
Be precise in your categorization based on the transaction description. Focus on what was purchased rather than the payment method.
"""
    
    return prompt

def generate_single_transaction_prompt(description: str, extended_details: Optional[str] = None, category_list: Optional[List[str]] = None) -> str:
    """Generate a prompt for categorizing a single transaction"""
    return generate_categorization_prompt([(description, extended_details)], category_list)

def parse_batch_response(raw_response: str, expected_count: int) -> List[str]:
    """Parse the batch response from Llama
    
    Args:
        raw_response: The raw text response from the LLM
        expected_count: The expected number of categories in the response
        
    Returns:
        List[str]: The parsed categories, one per transaction
    """
    predictions = []
    lines = raw_response.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Look for "Transaction X: Category" pattern
        import re
        match = re.search(r"transaction\s+(\d+)\s*:\s*(.*)", line.lower())
        if match:
            category = match.group(2).strip().strip('"').strip("'").strip("[]")
            predictions.append(category.title())  # Capitalize category names for consistency
    
    # Ensure we have the expected number of predictions
    if len(predictions) < expected_count:
        # Fill in missing predictions
        predictions.extend(["Misc"] * (expected_count - len(predictions)))
    
    return predictions[:expected_count]  # Truncate if too many

def test_llama_connection(max_retries=2, retry_delay=2) -> bool:
    """Test connection to Llama API
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    if not client:
        print("OpenAI client not initialized. Check your API key.")
        return False
    
    for attempt in range(max_retries + 1):
        try:
            print(f"Testing connection to Llama API... (Attempt {attempt + 1}/{max_retries + 1})")
            start_time = time.time()
            prompt = CONNECTION_TEST_PROMPT
            response = client.chat.completions.create(
                model=LLAMA_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10, # Increased slightly just in case
                temperature=0.0
            )
            elapsed_time = time.time() - start_time

            answer = None
            finish_reason = None
            if response.choices:
                message = response.choices[0].message
                finish_reason = response.choices[0].finish_reason
                if message and message.content:
                    answer = message.content.strip()

            if answer == "4":
                print(f"Connection successful! Response time: {elapsed_time:.2f} seconds")
                return True
            else:
                # Provide more detailed failure info
                print(f"Connection test failed on attempt {attempt + 1}.")
                print(f"  Expected: '4'")
                print(f"  Received: '{answer}'")
                print(f"  Finish Reason: {finish_reason}")
                # Optionally print more response details if needed, e.g., response.model_dump_json(indent=2)
                if attempt < max_retries:
                    print(f"  Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("  Max retries reached.")
                    return False
        except Exception as e:
            print(f"Error testing connection to Llama API on attempt {attempt + 1}: {e}")
            if attempt < max_retries:
                print(f"  Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("  Max retries reached.")
                return False
    return False # Should not be reached, but included for completeness

def get_llama_category(description: str, extended_details: Optional[str] = None) -> str:
    """Uses Llama-3.1 via Inference.net to categorize a transaction.
    
    Args:
        description: The transaction description
        extended_details: Optional additional details about the transaction
    
    Returns:
        str: The predicted category
    """
    if not client:
        print("API client not initialized. Cannot categorize.")
        return "Misc"
    if pd.isna(description) or str(description).strip() == "":
        print("Warning: Empty description provided. Defaulting to Misc.")
        return "Misc"
    
    try:
        prompt = generate_single_transaction_prompt(description, extended_details, PREDEFINED_CATEGORIES)
        
        response = client.chat.completions.create(
            model=LLAMA_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=30,
            temperature=0.0,
        )
        raw_category = response.choices[0].message.content.strip().strip('"').strip()

        # Check if the response is empty FIRST
        if not raw_category:
            print(f"Warning: LLM returned an empty response for description: '{description}'. Defaulting to Misc.")
            return "Misc"

        # Now proceed with matching non-empty responses
        if raw_category in PREDEFINED_CATEGORIES:
            return raw_category
        else:
            # Case-insensitive check
            for cat in PREDEFINED_CATEGORIES:
                if cat.lower() == raw_category.lower():
                    print(f"Warning: LLM response '{raw_category}' matched '{cat}' case-insensitively.")
                    return cat
            # Substring check
            found_cats = [cat for cat in PREDEFINED_CATEGORIES if cat.lower() in raw_category.lower()]
            if len(found_cats) == 1:
                chosen_cat = found_cats[0]
                print(f"Warning: LLM response '{raw_category}' contained '{chosen_cat}'. Using matched category.")
                return chosen_cat
            # If still not matched (non-empty, but unrecognized/ambiguous)
            print(f"Warning: LLM response '{raw_category}' not in predefined list or ambiguous. Defaulting to Misc.")
            return "Misc"
    except openai.APIConnectionError as e:
        print(f"API Connection Error: {e}")
    except openai.RateLimitError as e:
        print(f"API Rate Limit Error: {e}. Sleeping for 5s...")
        time.sleep(5)
    except openai.APIStatusError as e:
        print(f"API Status Error: {e.status_code} - {e.response}")
    except Exception as e:
        print(f"Error calling LLM API: {e}")
    return "Misc"

def get_llama_categories_batch(
    transactions: List[Tuple[str, Optional[str]]],
    batch_size: int = 10,
    max_retries: int = 3
) -> List[str]:
    """Get categories for a batch of transactions from Llama
    
    Args:
        transactions: List of tuples containing (description, extended_details)
        batch_size: Number of transactions to process in each batch
        max_retries: Maximum number of retries for failed API calls
        
    Returns:
        List of category predictions in the same order as input transactions
    """
    if not client:
        print("API client not initialized. Check your API key.")
        return ["Misc"] * len(transactions)
    
    all_predictions = []
    
    # Create batches
    for i in range(0, len(transactions), batch_size):
        batch_end = min(i + batch_size, len(transactions))
        batch = transactions[i:batch_end]
        
        # Generate prompt
        prompt = generate_categorization_prompt(batch)
        
        # Track retries
        retry_count = 0
        success = False
        
        while not success and retry_count < max_retries:
            try:
                # Call the API
                response = client.chat.completions.create(
                    model=LLAMA_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.0
                )
                
                raw_response = response.choices[0].message.content.strip()
                
                # Parse response
                predictions = parse_batch_response(raw_response, len(batch))
                
                # Add to results
                all_predictions.extend(predictions)
                success = True
                
            except Exception as e:
                retry_count += 1
                print(f"Error with API call (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    delay = 2 ** retry_count  # Exponential backoff
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print("Max retries reached. Using 'Misc' for failed transactions.")
                    all_predictions.extend(["Misc"] * len(batch))
        
        # Add a delay between batches to avoid rate limiting
        if batch_end < len(transactions):
            time.sleep(1)
    
    return all_predictions

def process_file_with_llama(file_to_process: str, output_file: str, processor=None) -> bool:
    """Process a single file with Llama API, batching transactions for efficiency.
    
    Args:
        file_to_process: Path to the input CSV file
        output_file: Path to save the categorized results
        processor: Optional post-processor for categories
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\nProcessing file {file_to_process} with Llama API...")
    
    try:
        data = pd.read_csv(file_to_process)
        # Basic preprocessing
        data.columns = [c.lower() for c in data.columns]
        if 'category' not in data.columns:
            data['category'] = ''
        if 'confidence' not in data.columns:
            data['confidence'] = 0.0
            
        # Count how many transactions we'll process
        mask = data['category'].isna() | (data['category'] == '')
        uncategorized_count = mask.sum()
        already_categorized_count = len(data) - uncategorized_count
        
        if uncategorized_count == 0:
            print(f"  All {len(data)} transactions already have categories. Nothing to do.")
            return True
            
        print(f"  Found {uncategorized_count} transactions without categories (out of {len(data)} total)")
        if already_categorized_count > 0:
            print(f"  Preserving {already_categorized_count} existing categorizations")
            
        # Process in batches
        batch_size = 50
        start_time = time.time()
        
        # Get indices of transactions that need categorization
        indices_to_process = data[mask].index.tolist()
        
        for batch_start in range(0, len(indices_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(indices_to_process))
            current_indices = indices_to_process[batch_start:batch_end]
            current_batch_size = len(current_indices)
            
            print(f"  Processing batch {batch_start//batch_size + 1}/{(len(indices_to_process) + batch_size - 1)//batch_size} " 
                  f"({current_batch_size} transactions, indices {current_indices[0]}-{current_indices[-1]})...")
            
            # Prepare batch of (description, extended_details) tuples
            batch_details = []
            for idx in current_indices:
                description = data.loc[idx, 'description']
                # Check if extended_details column exists
                extended_details = data.loc[idx, 'extended_details'] if 'extended_details' in data.columns else None
                batch_details.append((description, extended_details))
            
            # Get categories for the batch
            batch_categories = get_llama_categories_batch(batch_details, batch_size=current_batch_size)
            
            # Update categories in the dataframe
            for i, idx in enumerate(current_indices):
                if i < len(batch_categories):
                    data.loc[idx, 'category'] = batch_categories[i]
                    data.loc[idx, 'confidence'] = 0.95  # High confidence for LLM-based categories
            
            # Add delay between batches to avoid rate limiting
            if batch_end < len(indices_to_process):
                time.sleep(1)
        
        # Apply post-processing if requested
        if processor:
            print("  Applying post-processing to improve categories...")
            for idx in indices_to_process:
                data.loc[idx, 'category'] = processor.process(
                    data.loc[idx, 'description'], 
                    data.loc[idx, 'category']
                )
        
        # Save categorized data
        data.to_csv(output_file, index=False)
        
        elapsed_time = time.time() - start_time
        transactions_per_second = uncategorized_count / elapsed_time
        print(f"  Completed processing {uncategorized_count} transactions in {elapsed_time:.2f} seconds")
        print(f"  Average speed: {transactions_per_second:.2f} transactions per second")
        print(f"  Output saved to {output_file}")
        
        return True
        
    except Exception as e:
        print(f"  Error processing file with Llama: {e}")
        return False 