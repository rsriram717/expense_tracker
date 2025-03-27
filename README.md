# Transaction Categorization Tool

A tool for categorizing credit card transactions using machine learning based on previously categorized transactions.

## Features

- **Automatic Categorization**: Uses machine learning to categorize new transactions based on existing categorized data
- **Interactive Review**: Review and adjust predicted categories
- **Category Management**: Add, modify, and manage transaction categories
- **Model Retraining**: Update the model with feedback to improve future predictions
- **Spending Reports**: Generate reports and visualizations of your spending by category
- **Multiple File Support**: Work with multiple CSV files in organized directories

## Requirements

- Python 3.6+
- Required packages (install with `pip install -r requirements.txt`):
  - pandas
  - numpy
  - scikit-learn
  - tabulate
  - matplotlib (optional, for visualization)
  - seaborn (optional, for visualization)

## Installation

1. Clone or download this repository
2. Install required packages:

```bash
pip install -r requirements.txt
```

## Directory Structure

The tool uses the following directory structure:

```
├── data/
│   ├── categorized/   # Place your pre-categorized CSV files here
│   ├── to_categorize/ # Place new transactions to categorize here
│   └── output/        # Categorized transactions will be saved here
├── categorize_transactions.py
├── category_manager.py
├── transaction_tool.py
├── requirements.txt
└── README.md
```

## Usage

### 1. Train the Model

Train the model using your existing categorized transactions:

```bash
python transaction_tool.py train --data data/categorized
```

The tool will train a model on all CSV files in the categorized directory.

### 2. Categorize New Transactions

Categorize new transactions using the trained model:

```bash
python transaction_tool.py categorize --transactions data/to_categorize --output data/output
```

This will process all CSV files in the `to_categorize` directory and save the results to the `output` directory.

To categorize a single file:

```bash
python transaction_tool.py categorize --transactions data/to_categorize/your_file.csv --output data/output
```

### 3. Review and Modify Categories

Review and modify the predicted categories:

```bash
python transaction_tool.py review --transactions data/to_categorize --output data/output
```

This opens an interactive interface where you can:
- Select which file(s) to review
- View transactions and their predicted categories
- Edit categories for individual transactions
- Add new categories
- Navigate through batches of transactions

To review a specific file:

```bash
python transaction_tool.py review --transactions data/to_categorize --output data/output --file your_file.csv
```

### 4. Retrain with Feedback

Retrain the model with your feedback to improve future predictions:

```bash
python transaction_tool.py retrain
```

### 5. Generate Spending Reports

Generate spending reports and visualizations:

```bash
python transaction_tool.py report --data data/output --output spending_report.csv
```

This will create a report from all categorized files in the output directory.

To generate a report for a specific file:

```bash
python transaction_tool.py report --data data/output --file categorized_your_file.csv
```

## Data Format

### Input Data

The tool works with CSV files containing credit card transactions. The required columns are:

- **Date**: Transaction date
- **Description**: Transaction description
- **Amount**: Transaction amount
- **Appears On Your Statement As**: How the transaction appears on your statement (optional but helpful)

For categorized data, an additional column is required:
- **Category**: The transaction category

### Output Data

The output files will contain all the original columns plus:

- **Category**: The predicted or manually adjusted category
- **Confidence**: Confidence score for the prediction (0-1)

## Machine Learning Model

The tool uses a Random Forest classifier with TF-IDF features to predict transaction categories. This model:

1. Extracts text features from transaction descriptions using TF-IDF vectorization
2. Trains a Random Forest classifier on these features to predict categories
3. Calculates confidence scores for each prediction

You can customize the model by modifying the `TransactionCategorizer` class in `categorize_transactions.py`.

## Category Management

Categories are extracted from your existing categorized data. You can:

- Add new categories through the interactive review interface
- Edit assigned categories for transactions
- Provide feedback that improves future categorization

## Files

- `transaction_tool.py`: Main command-line interface
- `categorize_transactions.py`: Core categorization functionality
- `category_manager.py`: Interactive category management
- `category_feedback.csv`: Stored feedback for retraining (created automatically)
- `transaction_model.pkl`: Trained model file (created automatically)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 