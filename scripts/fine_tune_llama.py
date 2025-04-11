#!/usr/bin/env python
"""
Script to prepare data and initiate Llama fine-tuning job via OpenAI API.
"""

import pandas as pd
import json
import os
import time
from openai import OpenAI, BadRequestError

# Assuming regenerate_holdout defines create_holdout_split and load_full_dataset
from regenerate_holdout import load_full_dataset, create_holdout_split, HOLDOUT_SIZE, RANDOM_STATE
from config import INFERENCE_API_KEY, INFERENCE_BASE_URL

# --- Configuration ---
TRAINING_FILE_PATH = "data/llama_finetune_train_data.jsonl"
BASE_MODEL = "llama-3.1-8b-instruct" # Or the specific base model you want to fine-tune
# --- End Configuration ---

def format_data_for_finetuning(df: pd.DataFrame) -> str:
    """Formats the training data into JSONL format for OpenAI fine-tuning."""
    if not all(col in df.columns for col in ['Description', 'Category']):
        raise ValueError("DataFrame must contain 'Description' and 'Category' columns.")

    # Use a list comprehension for cleaner iteration and data creation
    lines_to_write = []
    for _, row in df.iterrows():
        description = str(row['Description'])
        category = str(row['Category'])

        # Basic cleaning to avoid potential issues
        description = description.strip().replace('\n', ' ').replace('\r', ' ')
        category = category.strip()

        messages = [
            {"role": "user", "content": description}, # Use description directly as user content
            {"role": "assistant", "content": category}
        ]

        # Ensure the line is valid JSON
        try:
            line = json.dumps({"messages": messages})
            lines_to_write.append(line)
        except Exception as e:
            print(f"Warning: Skipping row due to JSON conversion error: {e} - Row: {row.to_dict()}")

    # Join all valid lines with newline characters
    return "\n".join(lines_to_write) + "\n" # Ensure trailing newline

def save_training_file(data: str, filepath: str):
    """Saves the formatted data to a JSONL file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Explicitly use UTF-8 encoding
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(data)
        print(f"Training data successfully saved to {filepath}")
    except Exception as e:
        print(f"Error saving training file: {e}")
        raise

def upload_file_to_openai(client: OpenAI, filepath: str) -> str | None:
    """Uploads the training file to OpenAI.
    Returns the file ID if successful, otherwise None.
    """
    try:
        print(f"Uploading training file {filepath} to OpenAI...")
        with open(filepath, "rb") as f:
            response = client.files.create(file=f, purpose="fine-tune")
        file_id = response.id
        print(f"File uploaded successfully. File ID: {file_id}")
        return file_id
    except Exception as e:
        print(f"Error uploading file to OpenAI: {e}")
        return None

def create_finetuning_job(client: OpenAI, file_id: str, base_model: str) -> str | None:
    """Creates a fine-tuning job on OpenAI.
    Returns the job ID if successful, otherwise None.
    """
    try:
        print(f"Creating fine-tuning job for file {file_id} using base model {base_model}...")
        # Note: You might need to adjust hyperparameters depending on the API version/options
        response = client.fine_tuning.jobs.create(
            training_file=file_id,
            model=base_model,
            # hyperparameters={ # Optional: Adjust as needed
            #     "n_epochs": 3,
            # }
        )
        job_id = response.id
        print(f"Fine-tuning job created successfully. Job ID: {job_id}")
        print("Monitor the job status via the OpenAI dashboard or API.")
        print("Once completed, the fine-tuned model ID will be available in the job details.")
        return job_id
    except BadRequestError as e:
         print(f"Error creating fine-tuning job (Bad Request): {e}")
         print("This often indicates an issue with the training file format or content.")
         print("Please check the file and OpenAI's documentation.")
         return None
    except Exception as e:
        print(f"Error creating fine-tuning job: {e}")
        return None

def main():
    print("--- Llama Fine-tuning Preparation ---")

    # 1. Initialize OpenAI Client
    if not INFERENCE_API_KEY:
        print("Error: OpenAI API key (INFERENCE_API_KEY) not found in config.")
        return
    try:
        client = OpenAI(api_key=INFERENCE_API_KEY, base_url=INFERENCE_BASE_URL)
        print("OpenAI client initialized.")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return

    # 2. Load and split the full dataset to get the training portion
    print("Loading and splitting dataset...")
    full_data = load_full_dataset()
    if full_data is None:
        print("Failed to load data. Aborting.")
        return

    try:
        # Use the same split parameters as holdout generation
        train_data, _ = create_holdout_split(full_data, holdout_size=HOLDOUT_SIZE, random_state=RANDOM_STATE)
        print(f"Using {len(train_data)} transactions for fine-tuning training data.")
    except Exception as e:
        print(f"Error splitting data: {e}")
        return

    if train_data.empty:
        print("Training data is empty. Aborting.")
        return

    # 3. Format data
    print("Formatting data for fine-tuning...")
    try:
        formatted_data = format_data_for_finetuning(train_data)
    except Exception as e:
        print(f"Error formatting data: {e}")
        return

    # 4. Save formatted data to file
    try:
        save_training_file(formatted_data, TRAINING_FILE_PATH)
    except Exception:
        print("Failed to save training file. Aborting.")
        return

    # 5. Upload file to OpenAI
    file_id = upload_file_to_openai(client, TRAINING_FILE_PATH)
    if not file_id:
        print("Failed to upload file. Aborting.")
        return

    # 6. Create fine-tuning job
    job_id = create_finetuning_job(client, file_id, BASE_MODEL)
    if not job_id:
        print("Failed to create fine-tuning job.")
        # Consider adding cleanup logic here if needed (e.g., delete uploaded file)
    else:
        print("--- Fine-tuning job initiated successfully --- ")

if __name__ == "__main__":
    main()
# Ensure final newline 