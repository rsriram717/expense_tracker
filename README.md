# Financial Transaction Categorizer

A machine learning system for automatically categorizing financial transactions with post-processing rules for common merchants.

## Overview

This system uses a combination of machine learning and rule-based approaches to categorize financial transactions. It works by:

1. Training a machine learning model on labeled transaction data
2. Using the model to predict categories for new transactions
3. Applying post-processing rules based on merchant patterns to correct common transactions

## Features

- Random Forest classification with text and amount features
- Merchant pattern matching for reliable categorization of known vendors
- Confidence scores for each prediction
- Detailed statistics and reporting
- Support for incremental training with new data

## Directory Structure

```
.
├── data/
│   ├── categorized/        # Training data with labeled transactions
│   ├── to_categorize/      # New transactions to be categorized
│   ├── output/             # Categorized transaction results
│   └── merchants/          # Merchant pattern rules
├── improved_categorizer.py # Core ML categorization engine
├── merchant_postprocessor.py # Post-processing rules engine
├── categorize.py           # Command-line interface
├── manage_merchants.py     # Merchant pattern management tool
└── improved_model.pkl      # Trained model file
```

## Usage

### Categorizing Transactions

To categorize new transactions:

```bash
python categorize.py
```

This will:
1. Look for CSV files in `data/to_categorize/`
2. Apply the ML model from `improved_model.pkl`
3. Apply merchant post-processing rules from `data/merchants/merchant_categories.csv`
4. Save categorized results to `data/output/`

### Training the Model

To retrain the model with new data:

```bash
python categorize.py --train
```

Training data should be placed in `data/categorized/` with columns:
- `Description`: Transaction description
- `Amount`: Transaction amount
- `Category`: The correct category (for training data)

### Command-line Options

```
usage: categorize.py [-h] [--train] [--input-dir INPUT_DIR] [--output-dir OUTPUT_DIR] [--model-file MODEL_FILE] [--skip-post]

Categorize financial transactions using ML and merchant rules

options:
  -h, --help            show this help message and exit
  --train               Retrain the model before categorizing
  --input-dir INPUT_DIR
                        Directory containing transaction CSV files to categorize
  --output-dir OUTPUT_DIR
                        Directory where categorized files will be saved
  --model-file MODEL_FILE
                        Path to the model file
  --skip-post           Skip merchant post-processing
```

## Managing Merchant Rules

The merchant rules are stored in `data/merchants/merchant_categories.csv` with columns:
- `merchant_pattern`: Text pattern to match in the transaction description
- `category`: Category to assign when pattern matches
- `confidence`: Confidence level (0.0-1.0) for the rule

### Using the Merchant Management Tool

You can manage merchant patterns using the `manage_merchants.py` tool:

```bash
# List all merchant patterns
python manage_merchants.py list

# List patterns for a specific category
python manage_merchants.py list --category "Food & Drink"

# Add a new merchant pattern
python manage_merchants.py add --pattern "CHIPOTLE" --category "Food & Drink" --confidence 0.95

# Remove a merchant pattern
python manage_merchants.py remove --pattern "CHIPOTLE"

# List all categories
python manage_merchants.py categories

# Export merchant patterns to a CSV file
python manage_merchants.py export --output "backup_merchants.csv"

# Import merchant patterns from a CSV file
python manage_merchants.py import --input "backup_merchants.csv"
```

### Using the MerchantPostProcessor Class

You can also use the `MerchantPostProcessor` class directly in your Python code:

```python
from merchant_postprocessor import MerchantPostProcessor

processor = MerchantPostProcessor()

# Add a new merchant pattern
processor.add_merchant_pattern("CHIPOTLE", "Food & Drink", 0.95)

# Remove a merchant pattern
processor.remove_merchant_pattern("CHIPOTLE")

# Process transactions with merchant rules
updated_df = processor.process_transactions(transactions_df)
```

## Improving the System

To improve categorization accuracy:

1. Add more labeled training data to `data/categorized/`
2. Add specific merchant patterns for frequently encountered transactions
3. Review low-confidence predictions and correct them
4. Run `python categorize.py --train` to retrain with new data 