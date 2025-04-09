#!/usr/bin/env python
"""
Script to compare the performance of Random Forest and Llama models
on a holdout set of transactions.
"""

import os
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
from tabulate import tabulate

from transaction_categorizer import TransactionCategorizer
from config import PREDEFINED_CATEGORIES
from llm_service import test_llama_connection, get_llama_category
from model_training import load_model
# Import necessary classes for joblib/pickle to find them when loading the model
# even if they are not directly used in this script.
from model_training import FeaturePreparation, TextFeatures, NumericFeatures
# Also import custom functions used within the saved pipeline
from model_training import preprocess_text


def evaluate_random_forest(holdout_data: pd.DataFrame) -> pd.DataFrame | None:
    """
    Evaluate the Random Forest model on the holdout set.

    Args:
        holdout_data: DataFrame containing the holdout transactions

    Returns:
        DataFrame with RF predictions and metrics, or None if evaluation fails.
    """
    print("\nEvaluating Random Forest model...")

    try:
        # Initialize categorizer (will load RF model using joblib)
        categorizer = TransactionCategorizer(rf_confidence_threshold=0.0)

        if not categorizer.rf_model:
            raise ValueError("No RF model loaded after TransactionCategorizer initialization")

        # Prepare results dataframe
        results = holdout_data.copy()
        results['rf_category'] = None
        results['rf_confidence'] = 0.0

        # Process each transaction
        start_time = time.time()
        for idx, row in results.iterrows():
            # Get prediction using the categorizer's internal method
            rf_pred, rf_conf = categorizer._get_rf_prediction(row['description'], row['amount'])

            # Store prediction
            results.loc[idx, 'rf_category'] = rf_pred
            results.loc[idx, 'rf_confidence'] = rf_conf

        elapsed_time = time.time() - start_time

        # Calculate metrics
        results['rf_correct'] = results['rf_category'] == results['true_category']
        rf_accuracy = results['rf_correct'].mean() if not results.empty else 0.0

        print(f"Random Forest accuracy: {rf_accuracy:.2%}")
        print(f"Processing time: {elapsed_time:.2f} seconds")
        if not results.empty:
             print(f"Average time per transaction: {elapsed_time/len(results):.4f} seconds")
        else:
             print("Average time per transaction: N/A (no results)")

        return results

    except Exception as e:
        print(f"Warning: Error evaluating Random Forest model: {repr(e)}")
        print("Cannot evaluate Random Forest model.")
        return None # Return None if the model fails to load or run


def evaluate_llama(holdout_data: pd.DataFrame) -> pd.DataFrame | None:
    """
    Evaluate the Llama model on the holdout set.

    Args:
        holdout_data: DataFrame containing the holdout transactions

    Returns:
        DataFrame with Llama predictions and metrics, or None if evaluation fails.
    """
    print("\nEvaluating Llama model...")

    # Check if Llama is available
    llama_available = test_llama_connection()
    if not llama_available:
        print("Llama API not available. Skipping Llama evaluation.")
        return None

    # Prepare results dataframe
    results = holdout_data.copy()
    results['llama_category'] = None

    # Process each transaction
    start_time = time.time()
    total_transactions = len(results)
    for idx, row in results.iterrows():
        try:
            # Get prediction
            llama_prediction = get_llama_category(row['description'], row.get('extended_details'))

            # Store prediction
            results.loc[idx, 'llama_category'] = llama_prediction

            # Print progress for every 10 transactions
            if (idx + 1) % 10 == 0 or (idx + 1) == total_transactions:
                print(f"  Processed {idx + 1}/{total_transactions} transactions...")
        except Exception as e:
            print(f"  Error processing transaction {idx+1} with Llama: {repr(e)}")
            results.loc[idx, 'llama_category'] = "Error" # Mark as error

    elapsed_time = time.time() - start_time

    # Calculate metrics (excluding rows with errors)
    valid_results = results[results['llama_category'] != "Error"].copy() # Use .copy() to avoid SettingWithCopyWarning
    if not valid_results.empty:
        valid_results['llama_correct'] = valid_results['llama_category'] == valid_results['true_category']
        llama_accuracy = valid_results['llama_correct'].mean()
        print(f"Llama accuracy (excluding errors): {llama_accuracy:.2%}")
    else:
        llama_accuracy = 0.0
        print("Llama accuracy: N/A (no valid results)")

    print(f"Processing time: {elapsed_time:.2f} seconds")
    if total_transactions > 0:
        print(f"Average time per transaction: {elapsed_time/total_transactions:.4f} seconds")

    # Merge correctness back into the main results df
    if not valid_results.empty:
         results = results.merge(valid_results[['llama_correct']], left_index=True, right_index=True, how='left')
         results['llama_correct'] = results['llama_correct'].fillna(False) # Errors are incorrect
    else:
         results['llama_correct'] = False


    return results


def compare_models(results: pd.DataFrame):
    """
    Compare the performance of the models based on the merged results.

    Args:
        results: DataFrame containing true categories and predictions from both models.
    """
    print("\n=== MODEL COMPARISON ===")

    has_rf = 'rf_category' in results.columns
    has_llama = 'llama_category' in results.columns

    if not has_rf and not has_llama:
        print("No model results available for comparison.")
        return

    # --- Accuracy ---
    if has_rf:
        rf_accuracy = results['rf_correct'].mean() if 'rf_correct' in results.columns and not results.empty else 0.0
        print(f"Random Forest accuracy: {rf_accuracy:.2%}")
    if has_llama:
        llama_accuracy = results['llama_correct'].mean() if 'llama_correct' in results.columns and not results.empty else 0.0
        print(f"Llama accuracy: {llama_accuracy:.2%}")

    # --- Agreement ---
    if has_rf and has_llama:
        # Handle potential N/A or Error values before comparison
        rf_valid = results['rf_category'].notna() & (results['rf_category'] != 'N/A')
        llama_valid = results['llama_category'].notna() & (results['llama_category'] != 'N/A') & (results['llama_category'] != 'Error')
        both_valid_mask = rf_valid & llama_valid

        if both_valid_mask.any():
            agree_mask = results.loc[both_valid_mask, 'rf_category'] == results.loc[both_valid_mask, 'llama_category']
            disagree_mask = ~agree_mask

            agreement_rate = agree_mask.mean()
            print(f"Model agreement rate (where both predicted): {agreement_rate:.2%}")

            # Accuracy when models agree/disagree (only on commonly valid predictions)
            agree_indices = agree_mask[agree_mask].index
            disagree_indices = disagree_mask[disagree_mask].index

            accuracy_when_agree = results.loc[agree_indices, 'rf_correct'].mean() if not agree_indices.empty else np.nan
            print(f"Accuracy when models agree: {accuracy_when_agree:.2%}")

            rf_correct_disagree = results.loc[disagree_indices, 'rf_correct'].sum()
            llama_correct_disagree = results.loc[disagree_indices, 'llama_correct'].sum()
            print(f"When models disagree: RF wins {rf_correct_disagree} times, Llama wins {llama_correct_disagree} times")

            # --- Example Disagreements ---
            disagreements = results.loc[disagree_indices].copy()
            if not disagreements.empty:
                print("\n=== EXAMPLE DISAGREEMENTS (where both predicted) ===")
                disagreements['RF'] = np.where(disagreements['rf_correct'], '✓', '✗')
                disagreements['Llama'] = np.where(disagreements['llama_correct'], '✓', '✗')

                # Select relevant columns for display
                display_cols = ['description', 'true_category', 'rf_category', 'rf_confidence', 'llama_category', 'RF', 'Llama']
                # Ensure all columns exist before selecting
                display_cols = [col for col in display_cols if col in disagreements.columns]

                print(tabulate(disagreements.head(5)[display_cols], headers='keys', tablefmt='simple', showindex=False, floatfmt=".2f"))
            else:
                 print("No disagreements found where both models made a valid prediction.")
        else:
            print("No instances where both models made a valid prediction.")


    # --- Save Detailed Comparison ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_transactions = len(results)
    filename = f"model_comparison_holdout_{num_transactions}tx_{timestamp}.csv"
    try:
        results.to_csv(filename, index=False)
        print(f"\nComplete comparison saved to {filename}")
    except Exception as e:
        print(f"\nError saving comparison results to {filename}: {repr(e)}")


def main():
    print("Model Comparison Test")
    print("=====================")

    # Load the actual holdout data
    holdout_path = "models/holdout/holdout_data.csv"
    print(f"Loading full holdout set from {holdout_path}...")
    try:
        holdout_data = pd.read_csv(holdout_path)
        # Rename columns to match expected format
        holdout_data = holdout_data.rename(columns={
            "Description": "description",
            "Amount": "amount",
            "Extended Details": "extended_details",
            "Category": "true_category"
        })
        # Ensure required columns exist
        required_cols = ['description', 'amount', 'true_category']
        if not all(col in holdout_data.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in holdout_data.columns]
            print(f"Error: Holdout data is missing required columns: {missing_cols}")
            return
        # Add 'extended_details' if it's missing
        if 'extended_details' not in holdout_data.columns:
            holdout_data['extended_details'] = None

        # Use the full holdout set
        holdout_data = holdout_data.reset_index(drop=True) # Ensure clean index
        print(f"Using {len(holdout_data)} transactions from holdout set.")

    except FileNotFoundError:
        print(f"Error: Holdout file not found at {holdout_path}")
        return
    except Exception as e:
        print(f"Error loading or processing holdout data: {repr(e)}")
        return

    # Evaluate Random Forest
    rf_results = evaluate_random_forest(holdout_data)

    # Evaluate Llama
    llama_results = evaluate_llama(holdout_data)

    # Merge results
    # Start with a copy of the holdout data to ensure all original rows are present
    results = holdout_data.copy()

    if rf_results is not None:
        # Merge RF results based on index
        results = results.merge(
            rf_results[['rf_category', 'rf_confidence', 'rf_correct']],
            left_index=True, right_index=True, how='left'
        )
    else:
        results['rf_category'] = "N/A"
        results['rf_confidence'] = 0.0
        results['rf_correct'] = False

    if llama_results is not None:
        # Merge Llama results based on index
        results = results.merge(
            llama_results[['llama_category', 'llama_correct']],
            left_index=True, right_index=True, how='left'
        )
    else:
        results['llama_category'] = "N/A"
        results['llama_correct'] = False

    # Ensure columns exist even if merge didn't happen or results were None
    for col, default in [('rf_category', "N/A"), ('rf_confidence', 0.0), ('rf_correct', False),
                         ('llama_category', "N/A"), ('llama_correct', False)]:
        if col not in results.columns:
            results[col] = default

    if rf_results is None and llama_results is None:
         print("Neither model could be evaluated.")
         # No comparison possible, but we might still want to see the empty results structure
         # return # Optionally exit early

    # Compare models
    compare_models(results)


if __name__ == "__main__":
    main()
# Ensure a final newline character