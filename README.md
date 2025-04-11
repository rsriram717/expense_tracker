# Financial Transaction Categorizer

A tool for automatically categorizing financial transactions using machine learning (RandomForest) and large language models (Llama).

## Features

- Automatic categorization of financial transactions from CSV files
- Multiple categorization methods:
  - Machine Learning (Random Forest model)
  - Large Language Model (Llama 3.1)
  - Hybrid approach that combines ML and LLM
  - Merchant pattern matching for known merchants
- Batch processing for efficient categorization
- Model training and evaluation
- SQLite database for storing categorized transactions
- Command-line interface

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file based on `.env.sample` and add your Inference.net API key

## Usage

### Categorize Transactions

Place CSV files with transactions in the `data/to_categorize` directory, then run:

```
python categorize.py
```

The categorized files will be saved in the `data/output` directory.

### Command Line Options

```
python categorize.py --help
```

Options:
- `--train`: Train a new model before categorizing
- `--input-dir PATH`: Custom input directory
- `--output-dir PATH`: Custom output directory
- `--skip-post`: Skip merchant post-processing
- `--force-local`: Force using local model even if LLM is available

### Train a New Model

To train a new model from the database:

```
python -m model_training
```

## Project Structure

```
.
├── config.py                  # Configuration settings
├── transaction_categorizer.py # Main transaction categorization logic
├── llm_service.py             # LLM service for transaction categorization
├── model_training.py          # ML model training functionality
├── merchant_postprocessor.py  # Post-processing for merchant matching
├── db_connector.py            # Database connection and schema
├── categorize.py              # Command-line interface
├── data/
│   ├── to_categorize/         # Input directory for transactions
│   ├── output/                # Output directory for categorized transactions
│   ├── merchants/             # Merchant patterns for post-processing
│   ├── model_comparisons/     # Model comparison results
│   └── transactions.db        # SQLite database for storing transactions
├── models/                    # Trained models
├── scripts/                   # Utility scripts
│   ├── dashboard.py           # UI dashboard for transaction management
│   ├── review_transactions.py # Tool for reviewing transactions
│   ├── manage_merchants.py    # Tool for managing merchant patterns
│   ├── fine_tune_llama.py     # Script for fine-tuning LLM
│   └── ...
├── analysis/                  # Analysis files and reports
│   ├── model_comparison_analysis.md # Analysis of model comparisons
│   ├── compare_models.py      # Script for comparing models
│   └── ...
└── tests/                     # Unit tests
```

## How It Works

The tool uses a hybrid approach to categorize transactions:

1. First, it tries to match the merchant to a known pattern in the merchant cache
2. If not found, it uses a Random Forest model to predict the category
3. If the RF model has low confidence, it uses an LLM (Llama) for prediction
4. Finally, post-processing is applied to improve accuracy

## Development

### Running Tests

```
python -m unittest discover tests
```

## License

MIT 