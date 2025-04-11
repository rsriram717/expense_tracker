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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Import the base model and the new wrappers
from models.base_model import BaseTransactionCategorizer
from models.rf_categorizer import RandomForestCategorizer
from models.llama_categorizer import LlamaCategorizer

# Import config and other utils needed
from config import PREDEFINED_CATEGORIES
from model_training import FeaturePreparation, TextFeatures, NumericFeatures, preprocess_text, store_evaluation_results


def evaluate_model(model: BaseTransactionCategorizer, holdout_data: pd.DataFrame) -> Tuple[pd.DataFrame | None, Dict | None]:
    """
    Evaluate any model conforming to BaseTransactionCategorizer on the holdout set.

    Args:
        model: An instance of BaseTransactionCategorizer (e.g., RandomForestCategorizer, LlamaCategorizer)
        holdout_data: DataFrame containing the holdout transactions with 'description', 'amount', 'true_category'

    Returns:
        Tuple:
            - DataFrame with predictions and confidence, or None on error.
            - Dictionary containing evaluation metrics, or None on error.
    """
    model_info = model.get_model_info()
    model_name = model_info.get('model_version', 'unknown_model')
    print(f"\nEvaluating model: {model_name} ({model_info.get('model_type', '')})...")

    # Prepare results dataframe
    results = holdout_data.copy()
    pred_col = f"pred_category_{model_name}"
    conf_col = f"pred_confidence_{model_name}"
    correct_col = f"correct_{model_name}"
    results[pred_col] = None
    results[conf_col] = 0.0

    try:
        start_time = time.time()
        # Use the model's predict method which returns (predictions, confidence_scores)
        predictions, confidence_scores = model.predict(results)
        elapsed_time = time.time() - start_time

        # Store results
        results[pred_col] = predictions
        results[conf_col] = confidence_scores
        results[correct_col] = results[pred_col] == results['true_category']

        # Calculate metrics
        accuracy = results[correct_col].mean()
        avg_confidence = results[conf_col].mean()
        std_confidence = results[conf_col].std()
        
        # Weighted precision, recall, f1
        precision, recall, f1, _ = precision_recall_fscore_support(
            results['true_category'],
            results[pred_col],
            average='weighted',
            zero_division=0
        )

        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Avg Confidence: {avg_confidence:.4f} (StdDev: {std_confidence:.4f})")
        print(f"  Precision (weighted): {precision:.4f}")
        print(f"  Recall (weighted): {recall:.4f}")
        print(f"  F1 Score (weighted): {f1:.4f}")
        print(f"  Processing time: {elapsed_time:.2f} seconds")
        if not results.empty:
            print(f"  Average time per transaction: {elapsed_time/len(results):.4f} seconds")
        else:
            print("  Average time per transaction: N/A (no results)")

        metrics = {
            'model_name': model_name,
            'model_type': model_info.get('model_type', ''),
            'evaluation_timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'avg_confidence': avg_confidence,
            'std_confidence': std_confidence,
            'processing_time_seconds': elapsed_time,
            'transactions_processed': len(results)
        }

        # Select only relevant columns for the output dataframe slice
        output_df_cols = [pred_col, conf_col, correct_col]
        output_df = results[output_df_cols]

        return output_df, metrics

    except Exception as e:
        print(f"Error evaluating model {model_name}: {repr(e)}")
        import traceback
        traceback.print_exc() # Print stack trace for debugging
        return None, None

def compare_evaluation_results(results_df: pd.DataFrame, model_metrics: List[Dict]):
    """
    Compare the performance of models based on the merged evaluation results.

    Args:
        results_df: DataFrame containing true categories and predictions/confidences for all models.
        model_metrics: List of metric dictionaries returned by evaluate_model.
    """
    print("\n=== MODEL COMPARISON ===")

    if not model_metrics:
        print("No model evaluation metrics available for comparison.")
        return

    # --- Accuracy & Metrics Summary ---
    metric_summary = []
    model_names = []
    for metrics in model_metrics:
        model_names.append(metrics['model_name'])
        metric_summary.append({
            'Model': metrics['model_name'],
            'Type': metrics['model_type'],
            'Accuracy': f"{metrics.get('accuracy', 0):.2%}",
            'F1 (Wt)': f"{metrics.get('f1_weighted', 0):.4f}",
            'Avg Conf': f"{metrics.get('avg_confidence', 0):.4f}",
            'Time (s)': f"{metrics.get('processing_time_seconds', 0):.2f}"
        })
    
    if metric_summary:
        print(tabulate(metric_summary, headers='keys', tablefmt='simple'))
    else:
        print("No valid model metrics to display.")
        return # Cannot proceed with comparison if no metrics

    # --- Agreement (only if exactly 2 models were compared) ---
    if len(model_names) == 2:
        model1_name = model_names[0]
        model2_name = model_names[1]

        pred_col1 = f"pred_category_{model1_name}"
        pred_col2 = f"pred_category_{model2_name}"
        conf_col1 = f"pred_confidence_{model1_name}"
        conf_col2 = f"pred_confidence_{model2_name}"
        correct_col1 = f"correct_{model1_name}"
        correct_col2 = f"correct_{model2_name}"

        # Check if prediction columns exist
        if pred_col1 not in results_df.columns or pred_col2 not in results_df.columns:
            print("\nComparison requires predictions from both models. Skipping agreement analysis.")
        else:
            # Agreement assumes predictions are not None/NaN
            valid_mask1 = results_df[pred_col1].notna()
            valid_mask2 = results_df[pred_col2].notna()
            both_valid_mask = valid_mask1 & valid_mask2

            if both_valid_mask.any():
                agree_mask = results_df.loc[both_valid_mask, pred_col1] == results_df.loc[both_valid_mask, pred_col2]
                disagree_mask = ~agree_mask

                agreement_rate = agree_mask.mean()
                print(f"\nModel agreement rate ({model1_name} vs {model2_name}, where both predicted): {agreement_rate:.2%}")

                agree_indices = agree_mask[agree_mask].index
                disagree_indices = disagree_mask[disagree_mask].index

                # Use model 1's correctness when they agree
                accuracy_when_agree = results_df.loc[agree_indices, correct_col1].mean() if not agree_indices.empty else np.nan
                if pd.isna(accuracy_when_agree):
                    print("Accuracy when models agree: N/A")
                else:
                    print(f"Accuracy when models agree: {accuracy_when_agree:.2%}")

                model1_correct_disagree = results_df.loc[disagree_indices, correct_col1].sum()
                model2_correct_disagree = results_df.loc[disagree_indices, correct_col2].sum()
                print(f"When models disagree: {model1_name} wins {model1_correct_disagree} times, {model2_name} wins {model2_correct_disagree} times")

                # --- Example Disagreements ---
                disagreements = results_df.loc[disagree_indices].copy()
                if not disagreements.empty:
                    print("\n=== EXAMPLE DISAGREEMENTS (where both predicted) ===")
                    disagreements[f'{model1_name}_OK'] = np.where(disagreements[correct_col1], '✓', '✗')
                    disagreements[f'{model2_name}_OK'] = np.where(disagreements[correct_col2], '✓', '✗')

                    display_cols = [
                        'description', 'true_category',
                        pred_col1, conf_col1, f'{model1_name}_OK',
                        pred_col2, conf_col2, f'{model2_name}_OK'
                    ]
                    # Ensure all columns exist before selecting
                    display_cols = [col for col in display_cols if col in disagreements.columns]

                    # Adjust headers for clarity
                    headers = [
                        'Description', 'True',
                        f'{model1_name} Pred', f'{model1_name} Conf', 'OK?',
                        f'{model2_name} Pred', f'{model2_name} Conf', 'OK?'
                    ]
                    headers = [h for i, h in enumerate(headers) if display_cols[i] in disagreements.columns] # Match headers to existing cols


                    print(tabulate(disagreements.head(5)[display_cols], headers=headers, tablefmt='simple', showindex=False, floatfmt=".2f"))
                else:
                    print("No disagreements found where both models made a valid prediction.")
            else:
                print("No instances where both models made a valid prediction.")
    elif len(model_names) > 2:
         print("\nAgreement analysis only performed for exactly two models.")

    # --- Save Detailed Comparison ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_transactions = len(results_df)
    model_names_str = "_".join(model_names) if model_names else "models"
    filename = f"model_comparison_{model_names_str}_{num_transactions}tx_{timestamp}.csv"
    try:
        results_df.to_csv(filename, index=False)
        print(f"\nComplete comparison results saved to {filename}")
    except Exception as e:
        print(f"\nError saving comparison results to {filename}: {repr(e)}")

def main():
    print("Unified Model Evaluation Test")
    print("=============================")

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

        holdout_data = holdout_data.reset_index(drop=True) # Ensure clean index
        holdout_size = len(holdout_data)
        print(f"Using {holdout_size} transactions from holdout set.")

    except FileNotFoundError:
        print(f"Error: Holdout file not found at {holdout_path}")
        return
    except Exception as e:
        print(f"Error loading or processing holdout data: {repr(e)}")
        return

    # --- Initialize Models --- #
    models_to_evaluate: List[BaseTransactionCategorizer] = []

    # Try initializing RandomForest
    # try:
    #     rf_model = RandomForestCategorizer()
    #     models_to_evaluate.append(rf_model)
    # except Exception as e:
    #     print(f"Could not initialize RandomForestCategorizer: {repr(e)}")

    # Try initializing Llama
    try:
        llama_model = LlamaCategorizer() # Uses defaults (batch 50, version llama_v1)
        if llama_model.is_available:
            models_to_evaluate.append(llama_model)
        else:
            print("LlamaCategorizer initialized but API is not available. Skipping Llama evaluation.")
    except Exception as e:
        print(f"Could not initialize LlamaCategorizer: {repr(e)}")

    if not models_to_evaluate:
        print("No models could be initialized for evaluation.")
        return

    # --- Evaluate Models --- #
    all_results_dfs = [holdout_data.copy()] # Start with original data
    all_metrics = []

    for model in models_to_evaluate:
        results_df_slice, metrics = evaluate_model(model, holdout_data)
        if results_df_slice is not None and metrics is not None:
            all_results_dfs.append(results_df_slice)
            all_metrics.append(metrics)
            # --- Store results in Database ---
            try:
                 model_info = model.get_model_info()
                 store_evaluation_results(metrics, model_info, holdout_size)
            except Exception as db_e:
                 print(f"Error storing evaluation results for {metrics['model_name']} in DB: {repr(db_e)}")

    # --- Merge Results --- #
    if len(all_results_dfs) > 1:
        # Merge all result slices back onto the original holdout data using index
        final_results = all_results_dfs[0]
        for df_slice in all_results_dfs[1:]:
            final_results = final_results.merge(df_slice, left_index=True, right_index=True, how='left')

        # --- Compare Models --- #
        compare_evaluation_results(final_results, all_metrics)
    elif all_metrics: # Only one model was evaluated successfully
         print("\nOnly one model evaluated successfully:")
         print(tabulate([all_metrics[0]], headers="keys", tablefmt="simple"))
         # Save the single model results
         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
         num_transactions = len(holdout_data)
         model_name = all_metrics[0]['model_name']
         filename = f"model_evaluation_{model_name}_{num_transactions}tx_{timestamp}.csv"
         try:
              # Merge the single result back for saving
              final_results = all_results_dfs[0].merge(all_results_dfs[1], left_index=True, right_index=True, how='left')
              final_results.to_csv(filename, index=False)
              print(f"\nEvaluation results saved to {filename}")
         except Exception as e:
              print(f"\nError saving evaluation results to {filename}: {repr(e)}")
    else:
        print("No models were evaluated successfully.")

if __name__ == "__main__":
    main()
# Ensure a final newline character