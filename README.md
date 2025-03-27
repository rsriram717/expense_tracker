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
├── update_merchants.py     # Tool to update merchants from training data
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
usage: categorize.py [-h] [--train] [--update-merchants] [--input-dir INPUT_DIR]
                    [--output-dir OUTPUT_DIR] [--model-file MODEL_FILE] [--skip-post]
                    [--categorize-only] [--update-merchants-only]

Categorize financial transactions using ML and merchant rules

options:
  -h, --help            show this help message and exit
  --train               Retrain the model before categorizing
  --update-merchants    Prompt to update merchant patterns during training (implies --train)
  --input-dir INPUT_DIR
                        Directory containing transaction CSV files to categorize
  --output-dir OUTPUT_DIR
                        Directory where categorized files will be saved
  --model-file MODEL_FILE
                        Path to the model file
  --skip-post           Skip merchant post-processing
  --categorize-only     Only categorize, without training (even if --train or 
                        --update-merchants are specified)
  --update-merchants-only
                        Only update merchant patterns without training or categorizing
```

## Managing Merchant Rules

The merchant rules are stored in `data/merchants/merchant_categories.csv` with columns:
- `merchant_pattern`: Text pattern to match in the transaction description
- `category`: Category to assign when pattern matches
- `confidence`: Confidence level (0.0-1.0) for the rule

### Updating Merchant Patterns from Training Data

You can update merchant patterns based on patterns found in your training data:

```bash
# Update as part of model training
python categorize.py --train --update-merchants

# Only update merchant patterns without training or categorizing
python categorize.py --update-merchants-only

# Run the merchant updater directly
python update_merchants.py
```

The merchant updater will:
1. Analyze your categorized transactions
2. Identify common merchant patterns
3. Suggest new patterns or updates to existing ones
4. Let you interactively review and approve each suggestion

It will only suggest patterns with high confidence (consistent categorization) and will never automatically update merchant patterns without your approval.

### Command-line Options for update_merchants.py

```
usage: update_merchants.py [-h] [--data-dir DATA_DIR] [--merchant-file MERCHANT_FILE]
                          [--confidence CONFIDENCE] [--min-occurrences MIN_OCCURRENCES]
                          [--batch-size BATCH_SIZE] [--non-interactive] [--output OUTPUT]

Update merchant categories based on model predictions

options:
  -h, --help            show this help message and exit
  --data-dir DATA_DIR   Directory containing categorized transaction data
  --merchant-file MERCHANT_FILE
                        Merchant categories file
  --confidence CONFIDENCE
                        Minimum confidence threshold for suggestions (0.0-1.0)
  --min-occurrences MIN_OCCURRENCES
                        Minimum number of occurrences for a merchant pattern
  --batch-size BATCH_SIZE
                        Number of suggestions to review at one time
  --non-interactive     Generate a report file without interactive review
  --output OUTPUT       Output file for non-interactive mode
```

### Using the Merchant Management Tool

You can also manually manage merchant patterns using the `manage_merchants.py` tool:

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
4. Run `python categorize.py --train --update-merchants` to retrain with new data and update merchant patterns 