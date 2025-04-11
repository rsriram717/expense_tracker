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
SYSTEM_PROMPT = """You are a financial transaction categorizer that precisely matches transactions to predefined categories. For each transaction, provide the category and a confidence score between 0.0 and 1.0 indicating how certain you are about the assigned category."""

CONNECTION_TEST_PROMPT = "What is 2+2? Respond with only the numerical answer."

def generate_categorization_prompt(
    transactions: List[Tuple[str, Optional[str]]],
    category_list: List[str] = PREDEFINED_CATEGORIES,
    category_examples_json: Optional[str] = None
) -> str:
    """Generate a prompt for Llama to categorize a batch of transactions, requesting confidence scores.

    Args:
        transactions: List of tuples containing (description, extended_details)
        category_list: List of predefined categories
        category_examples_json: Optional JSON string with category examples

    Returns:
        The generated prompt string
    """
    # --- Instructions Section --- 
    instruction_lines = [
        f"Categorize the following financial transactions using ONLY these categories: {', '.join(category_list)}.",
        "For each transaction, determine the single best category and provide a confidence score between 0.0 and 1.0.",
        "The confidence score should reflect your certainty in the chosen category."
    ]

    # --- Category Information Section --- 
    category_info_lines = []
    if category_examples_json:
        category_info_lines.append("\nUse these examples to understand the categories better:")
        category_info_lines.append(category_examples_json)
    else:
        category_info_lines.append("\nCategory Clarifications:")
        category_info_lines.extend([
            "- Food & Drink: Restaurants, bars, cafes, fast food.",
            "- Groceries: Supermarkets, grocery stores.",
            "- Transportation: Rideshares (Uber/Lyft), public transit, gas, parking.",
            "- Utilities: Electricity, gas, water, internet, phone bills.",
            "- Home: Rent/mortgage, furniture, repairs, maintenance.",
            "- Subscriptions: Streaming services (Netflix, Spotify), memberships, software.",
            "- Shopping: Retail stores, online shopping (non-grocery).",
            "- Entertainment: Movies, concerts, events, hobbies.",
            "- Travel-Airline: Airline tickets.",
            "- Travel-Lodging: Hotels, Airbnb.",
            "- Medical: Doctor visits, pharmacy, hospital bills.",
            "- Clothes: Clothing stores, apparel.",
            "- Dante: Specific recurring payment related to 'Dante'.",
            "- Misc: Use ONLY for transactions that do not fit any other category."
        ])

    # --- Transaction Listing Section --- 
    transaction_lines = ["\nTransactions to categorize:"]
    for i, (description, extended_details) in enumerate(transactions):
        details_str = f" (Details: {extended_details})" if extended_details and pd.notna(extended_details) else ""
        transaction_lines.append(f"Transaction {i+1}: Description: {description}{details_str}")

    # --- Strict Output Format Section --- 
    output_format_lines = [
        "\n*** VERY IMPORTANT: RESPONSE FORMAT ***",
        "Your response MUST contain ONLY the categorized transactions.",
        "Do NOT include any introductory text, explanations, summaries, or any text other than the transaction lines.",
        f"Output exactly {len(transactions)} lines, one for each transaction.",
        "Each line MUST follow this exact format:",
        "Transaction [Number]: [Category] (Confidence: [Score])",
        "Example Line: Transaction 1: Shopping (Confidence: 0.85)",
        "Use the exact words 'Transaction', 'Confidence', parentheses, and colon as shown.",
        "Ensure the [Score] is a number between 0.0 and 1.0."
    ]

    # --- Combine Sections --- 
    prompt = "\n".join(instruction_lines + category_info_lines + transaction_lines + output_format_lines)

    return prompt

# Helper function for single transaction prompt (if needed later)
def generate_single_transaction_prompt(
    description: str,
    extended_details: Optional[str] = None,
    category_list: List[str] = PREDEFINED_CATEGORIES
) -> str:
    """Generates a prompt for a single transaction, requesting confidence score."""
    # Reuses the batch prompt logic for consistency
    return generate_categorization_prompt([(description, extended_details)], category_list)

def parse_batch_response(raw_response: str, expected_count: int) -> List[Tuple[str, float]]:
    """Parse the batch response from Llama, extracting category and confidence score.

    Args:
        raw_response: The raw text response from the LLM
        expected_count: The expected number of categories in the response

    Returns:
        List[Tuple[str, float]]: List of (category, confidence) tuples
    """
    predictions: Dict[int, Tuple[str, float]] = {} # Store by transaction index
    lines = raw_response.split('\n')

    import re
    # Regex V3: Handles optional leading number+period, more flexible spacing, optional confidence.
    pattern = re.compile(
        r"^\"?\s*(?:\d+\.\s*)?"     # Optional start quote, optional leading number/period (e.g., "1.")
        r"Transaction\s+(\d+)\s*:?"  # Transaction Num (Group 1), optional colon
        r"\s*(.+?)\s*"              # Category (Group 2), non-greedy
        r"(?:\(\s*Confidence\s*:?\s*(\d\.?\d*)\s*\))?" # Optional Confidence group (Group 3)
        r"\s*\"?$",               # Optional end quote, end of line
        re.IGNORECASE
    )

    for line in lines:
        line = line.strip()
        # Skip common conversational lines or empty lines
        if not line or line.lower().startswith("here") or line.lower().startswith("please note") or line.lower().startswith("-"):
            continue

        match = pattern.search(line)
        if match:
            try:
                idx = int(match.group(1)) - 1
                category_str = match.group(2).strip().rstrip('(').strip()
                confidence_str = match.group(3)

                category = "Misc"
                confidence = 0.1

                if category_str:
                    parsed_category = category_str.title()
                    matched_predefined = False
                    for pc in PREDEFINED_CATEGORIES:
                        if pc.lower() == parsed_category.lower():
                            category = pc
                            matched_predefined = True
                            break
                    if not matched_predefined:
                        category = "Misc"
                        confidence = 0.2

                if confidence_str:
                    try:
                        parsed_confidence = float(confidence_str)
                        confidence = max(0.0, min(1.0, parsed_confidence))
                    except ValueError:
                        print(f"Warning: Could not parse confidence '{confidence_str}' in line: '{line}'. Using default 0.1.")
                        confidence = 0.1
                else:
                    print(f"Warning: Confidence score pattern not matched in line: '{line}'. Assigning default 0.1.")
                    confidence = 0.1

                if idx not in predictions:
                     predictions[idx] = (category, confidence)

            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse structure of matched line: '{line}'. Error: {e}")
                continue
        else:
            # Fallback for simpler "Transaction X: Category" format
            simple_match = re.search(r"^\"?\s*(?:\d+\.\s*)?Transaction\s+(\d+)\s*:?\s*(.+?)\s*\"?$", line, re.IGNORECASE)
            if simple_match:
                 try:
                      idx = int(simple_match.group(1)) - 1
                      category_str = simple_match.group(2).strip()
                      print(f"Warning: Line matched simplified format (no confidence): '{line}'. Assigning default 0.1 confidence.")
                      category = "Misc"
                      confidence = 0.1
                      if category_str:
                           parsed_category = category_str.title()
                           for pc in PREDEFINED_CATEGORIES:
                                if pc.lower() == parsed_category.lower():
                                     category = pc
                                     break
                           if category == "Misc":
                                confidence = 0.05

                      if idx not in predictions:
                           predictions[idx] = (category, confidence)
                 except (ValueError, IndexError) as e:
                      print(f"Warning: Could not parse structure of simplified matched line: '{line}'. Error: {e}")
            else:
                 # Only print warning if it doesn't look like a header/footer line
                 if not line.startswith(("Here", "Please", "-")):
                     print(f"Warning: Line did not match expected format: '{line}'")

    # Fill gaps
    final_results = []
    for i in range(expected_count):
        if i in predictions:
            final_results.append(predictions[i])
        else:
            print(f"Warning: Missing prediction entirely for transaction index {i+1}. Using ('Misc', 0.05).")
            final_results.append(("Misc", 0.05))

    return final_results

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

def get_llama_category(description: str, extended_details: Optional[str] = None) -> Tuple[str, float]:
    """Get category and confidence for a single transaction from Llama."""
    if not client:
        print("API client not initialized. Check your API key.")
        return "Misc", 0.0

    prompt = generate_single_transaction_prompt(description, extended_details)

    try:
        response = client.chat.completions.create(
            model=LLAMA_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100, # Enough for one line response
            temperature=0.0
        )
        raw_response = response.choices[0].message.content.strip()
        # Parse the single response (expected_count=1)
        parsed_results = parse_batch_response(raw_response, 1)
        return parsed_results[0] if parsed_results else ("Misc", 0.0)

    except Exception as e:
        print(f"Error getting single Llama category: {e}")
        return "Misc", 0.0

def get_llama_categories_batch(
    transactions: List[Tuple[str, Optional[str]]],
    batch_size: int = 50,
    max_retries: int = 3
) -> List[Tuple[str, float]]: # Return list of tuples
    """Get categories and confidence for a batch of transactions from Llama.

    Args:
        transactions: List of tuples containing (description, extended_details)
        batch_size: Number of transactions to process in each batch
        max_retries: Maximum number of retries for failed API calls

    Returns:
        List[Tuple[str, float]]: List of (category, confidence) tuples
    """
    if not client:
        print("API client not initialized. Check your API key.")
        return [("Misc", 0.0)] * len(transactions)

    all_predictions: List[Tuple[str, float]] = []

    # Process in batches as defined by the input batch_size argument
    # This loop manages the chunks sent to the API if len(transactions) > batch_size
    num_api_batches = (len(transactions) + batch_size - 1) // batch_size
    current_api_batch_num = 0

    for i in range(0, len(transactions), batch_size):
        batch_end = min(i + batch_size, len(transactions))
        batch = transactions[i:batch_end]
        current_batch_size = len(batch)
        current_api_batch_num += 1

        print(f"  Sending API batch {current_api_batch_num}/{num_api_batches} ({current_batch_size} transactions)... ")

        # Generate prompt with instructions for confidence scores
        prompt = generate_categorization_prompt(batch)

        retry_count = 0
        success = False
        batch_result: List[Tuple[str, float]] = []

        while not success and retry_count < max_retries:
            try:
                start_api_call_time = time.time()
                response = client.chat.completions.create(
                    model=LLAMA_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    # Adjust max_tokens based on expected output length
                    # ~25 chars per line * current_batch_size + buffer
                    max_tokens=max(500, 30 * current_batch_size),
                    temperature=0.0
                )
                api_call_duration = time.time() - start_api_call_time
                print(f"    API call duration: {api_call_duration:.2f}s")

                raw_response = response.choices[0].message.content.strip()

                # Parse response to get (category, confidence) tuples
                predictions = parse_batch_response(raw_response, current_batch_size)
                batch_result = predictions
                success = True

            except Exception as e:
                retry_count += 1
                print(f"    Error during API call (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    delay = 2 ** retry_count
                    print(f"    Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"    Max retries reached for API batch {current_api_batch_num}. Using ('Misc', 0.0).")
                    batch_result = [("Misc", 0.0)] * current_batch_size
                    success = True # Exit retry loop, failure recorded

        all_predictions.extend(batch_result)

        # Add a delay ONLY if successful and more batches remaining
        if success and batch_result[0][1] > 0.0 and batch_end < len(transactions):
            time.sleep(1) # Configurable delay

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
        # Ensure confidence column exists and is float
        if 'confidence' not in data.columns:
            data['confidence'] = 0.0
        else:
            # Convert existing confidence to float, coercing errors to NaN then filling with 0.0
            data['confidence'] = pd.to_numeric(data['confidence'], errors='coerce').fillna(0.0)
            
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
                  f"({current_batch_size} transactions)...")
            
            # Prepare batch of (description, extended_details) tuples
            batch_details = []
            for idx in current_indices:
                description = data.loc[idx, 'description']
                extended_details = data.loc[idx, 'extended_details'] if 'extended_details' in data.columns else None
                batch_details.append((description, extended_details))
            
            # Get categories and confidences for the batch
            batch_results = get_llama_categories_batch(batch_details, batch_size=current_batch_size) # Returns List[Tuple[str, float]]
            
            # Update categories and confidences in the dataframe
            for i, idx in enumerate(current_indices):
                if i < len(batch_results):
                    category, confidence = batch_results[i]
                    data.loc[idx, 'category'] = category
                    data.loc[idx, 'confidence'] = confidence # Use confidence from LLM result
                    data.loc[idx, 'source'] = 'llm' # Add source tracking
            
            # Add delay between batches to avoid rate limiting (handled within get_llama_categories_batch)
            # if batch_end < len(indices_to_process):
            #     time.sleep(1)
        
        # Apply post-processing if requested (might modify category, keep confidence)
        if processor:
            print("  Applying post-processing to improve categories...")
            processed_indices_count = 0
            for idx in indices_to_process:
                 # Check if the row has a category assigned by the LLM before processing
                 if data.loc[idx, 'source'] == 'llm':
                     original_category = data.loc[idx, 'category']
                     processed_category = processor.process(
                         data.loc[idx, 'description'],
                         original_category
                     )
                     if processed_category != original_category:
                          data.loc[idx, 'category'] = processed_category
                          # Optional: Adjust confidence if post-processor changes category?
                          # data.loc[idx, 'confidence'] *= 0.9 # Example: Reduce confidence slightly
                          processed_indices_count += 1
            if processed_indices_count > 0:
                 print(f"  Post-processor modified {processed_indices_count} categories.")
        
        # Save categorized data
        # Ensure confidence column is float before saving
        data['confidence'] = data['confidence'].astype(float)
        data.to_csv(output_file, index=False)
        
        elapsed_time = time.time() - start_time
        # Avoid division by zero if elapsed_time is very small or zero
        transactions_per_second = uncategorized_count / elapsed_time if elapsed_time > 0 else float('inf')
        print(f"  Completed processing {uncategorized_count} transactions in {elapsed_time:.2f} seconds")
        print(f"  Average speed: {transactions_per_second:.2f} transactions per second")
        print(f"  Output saved to {output_file}")
        
        return True
        
    except Exception as e:
        print(f"  Error processing file with Llama: {e}")
        # Optionally re-raise the exception if needed for debugging
        # raise e
        return False 