# Expense Tracker

A tool for categorizing and analyzing financial transactions.

## Project Structure

- **categorize_transactions.py**: Main script for transaction categorization
- **category_manager.py**: Utilities for managing transaction categories
- **transaction_tool.py**: CLI tool for working with transactions

## Data Directory Structure

The project expects the following data directories (not included in repo):

- `data/to_categorize/`: Place your transaction CSV files here for categorization
- `data/categorized/`: Stores categorized transaction data
- `data/output/`: Contains processed output files

## Setup

1. Clone the repository
```
git clone https://github.com/rsriram717/expense_tracker.git
cd expense_tracker
```

2. Create a virtual environment and install dependencies
```
pip install -r requirements.txt
```

3. Create the data directories
```
mkdir -p data/to_categorize data/categorized data/output
```

4. Add your transaction files to the `data/to_categorize/` directory

## Usage

Run the transaction tool:
```
python transaction_tool.py
```

## Note

All data files are excluded from version control. You'll need to add your own data files to the appropriate directories. 