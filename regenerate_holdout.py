#!/usr/bin/env python
"""
Script to regenerate the holdout dataset (models/holdout/holdout_data.csv)
using the full dataset from the database.
"""

import pandas as pd
from models.evaluation.evaluator import ModelEvaluator
from db_connector import get_engine # Assuming db_connector is accessible

# --- Configuration ---
HOLDOUT_SIZE = 0.2 # Fraction of data to use for holdout (e.g., 0.2 for 20%)
RANDOM_STATE = 42  # Ensure consistency with previous splits if desired
# --- End Configuration ---

def load_full_dataset() -> pd.DataFrame | None:
    """Loads the full, cleaned dataset for creating the holdout set."""
    print("Loading full dataset from database...")
    try:
        engine = get_engine()
        # Adjust SQL query if needed to select relevant columns or filter data
        data = pd.read_sql('SELECT * FROM transactions', engine)
        print(f"Loaded {len(data)} records from the database.")

        if data.empty:
            print("No data found in the database.")
            return None

        # --- Optional: Apply Deduplication (similar to model_training.py) ---
        identifying_columns = ['transaction_date', 'description', 'amount', 'source_file']
        if all(col in data.columns for col in identifying_columns + ['timestamp']):
             print(f"Deduplicating based on: {identifying_columns} keeping the latest entry.")
             data['timestamp'] = pd.to_datetime(data['timestamp'])
             data = data.sort_values(by=identifying_columns + ['timestamp'], ascending=[True]*len(identifying_columns) + [False])
             initial_count = len(data)
             data = data.drop_duplicates(subset=identifying_columns, keep='first')
             deduplicated_count = len(data)
             print(f"Deduplication complete. Kept {deduplicated_count} unique transactions (removed {initial_count - deduplicated_count}).")
        else:
             print("Warning: Skipping deduplication due to missing columns (timestamp or identifying columns).")
        # --- End Deduplication ---

        # --- Data Validation ---
        required_cols = ['Description', 'Amount', 'Category'] # Match columns expected by ModelEvaluator
        # Rename if necessary from DB columns
        rename_map = {'description': 'Description', 'amount': 'Amount', 'category': 'Category'}
        data = data.rename(columns=rename_map)

        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            print(f"Error: Loaded data is missing required columns for holdout creation: {missing_cols}")
            print(f"Available columns after rename: {list(data.columns)}")
            return None

        # Ensure Category is suitable for stratification
        if data['Category'].nunique() < 2:
             print("Error: Not enough unique categories (< 2) for stratified split.")
             return None
        category_counts = data['Category'].value_counts()
        if (category_counts < 2).any():
             print("Warning: Some categories have fewer than 2 samples. Stratification might behave unexpectedly or fail.")
             print(category_counts[category_counts < 2])
        # --- End Data Validation ---


        print(f"Using {len(data)} valid, unique records for holdout set creation.")
        return data

    except Exception as e:
        print(f"Error loading data from database: {e}")
        return None

def main():
    print("Regenerating Holdout Set...")
    
    # 1. Load full dataset
    full_data = load_full_dataset()
    if full_data is None:
        print("Failed to load data. Aborting holdout set regeneration.")
        return

    # 2. Initialize evaluator (uses default holdout path)
    evaluator = ModelEvaluator()
    holdout_path = evaluator.holdout_path
    print(f"Target holdout file: {holdout_path}")

    # 3. Create and save the new holdout set
    try:
        print(f"Creating holdout set with size {HOLDOUT_SIZE:.1%}...")
        # Temporarily adjust evaluator's internal random state if needed, though it uses 42 in split
        evaluator.create_holdout_set(full_data, holdout_size=HOLDOUT_SIZE)
        print(f"Successfully regenerated holdout set and saved to {holdout_path}")
        # Verify size
        try:
            new_holdout = pd.read_csv(holdout_path)
            print(f"New holdout set contains {len(new_holdout)} transactions.")
        except Exception as e:
            print(f"Could not verify the size of the newly created holdout file: {e}")

    except ValueError as e:
         print(f"Error during train_test_split (likely due to stratification issues): {e}")
         print("Holdout set regeneration failed.")
    except Exception as e:
        print(f"An error occurred during holdout set creation: {e}")
        print("Holdout set regeneration failed.")

if __name__ == "__main__":
    main()
# Ensure final newline 