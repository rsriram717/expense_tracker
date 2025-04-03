# Transaction Categorizer

## Overview

Transaction Categorizer is a Streamlit-based web application that helps users categorize financial transactions. The app uses machine learning to automatically assign categories to transactions based on their descriptions and other metadata, while also allowing users to manually review and edit these categorizations.

## Directory Structure

```
finance/
├── data/
│   ├── categorized/        # Training data with known categories
│   ├── to_categorize/      # Files that need categorization
│   └── output/             # Results of categorization
├── review_transactions.py  # Main Streamlit application
└── improved_categorizer.py # ML model for categorization
```

## Data Flow

1. **Input**: Transaction data in CSV format placed in `data/to_categorize/`
2. **Training**: Model trains on data in `data/categorized/`
3. **Processing**: Model categorizes transactions in the input files
4. **Output**: Categorized data saved to `data/output/`
5. **Review**: Users review and edit categorizations in the UI
6. **Feedback**: Manually verified categorizations can be added to training data

## Core Functionality

### Data Loading and Preparation

- The app loads transaction files from the `data/to_categorize/` directory
- Column names are normalized to a consistent format
- "MOBILE PAYMENT - THANK YOU" transactions are filtered out
- Missing columns (category, confidence) are added if they don't exist

### Categorization

- The categorization process uses the `improved_categorizer.py` module
- Two main functions are exposed:
  - `train_model()`: Trains on data in the `data/categorized/` directory
  - `categorize_transactions()`: Applies the model to files in `data/to_categorize/`
- Results are saved with a prefix of `improved_categorized_` in the `data/output/` directory
- Categories include: Food & Drink, Transportation, Entertainment, Groceries, Shopping, Travel-Airline, Travel-Lodging, Travel-Other, Clothes, Subscriptions, Home, Pets, Beauty, Professional Services, Medical, and Misc

### User Interface

The application has two main tabs:

#### Review Transactions Tab
- File selector for choosing which transaction file to review
- Data editor for viewing and editing categorizations
- Two action buttons:
  - **Save Changes**: Saves edits to the output directory
  - **Submit to Training Data**: Copies categorized data to the training directory

#### Analytics Tab
- Visual analytics of categorized transactions
- Charts include:
  - Category distribution (pie chart)
  - Monthly spending by category (stacked bar chart)
  - Model confidence distribution (histogram)

### Session State Management

The app uses Streamlit's session state to manage:
- Currently selected output file (`output_file`)
- Whether categorization has been run (`categorization_run`)
- Whether to reload categorized data (`reload_categorized`)

## Workflow

1. **Initial Load**: The app loads uncategorized transaction data
2. **Run Categorization**: User clicks the "Run Categorization" button to apply the ML model
3. **Review**: After categorization, results are loaded into the UI for review
4. **Edit**: User can manually adjust categories as needed
5. **Save**: Changes can be saved to the output directory
6. **Submit**: Finalized categorizations can be submitted to the training data for model improvement

## Key Design Decisions

### Separation of Input and Output
- The app maintains clear separation between input files and categorized output
- Each time the app loads, it starts with uncategorized data by default
- Categorized data is only loaded immediately after running categorization

### Consistent Data Format
- Column names are normalized to lowercase for consistent internal processing
- Date columns are converted to datetime format for proper sorting and analytics
- Standard categories are enforced through a dropdown in the UI

### User Control
- The app gives users final control over categorization
- ML categorization serves as a starting point, not final authority
- Manual edits can be saved and used to improve future model performance

## Technical Implementation Notes

### Dependencies
- Streamlit: UI framework
- Pandas: Data manipulation
- Plotly Express: Data visualization
- Pathlib: File path management

### Data Editor
- Uses Streamlit's `st.data_editor` for interactive editing
- Custom column configurations for different data types
- Required category selection enforced through UI

### Analytics
- Data visualization powered by Plotly Express
- Aggregated views of spending patterns
- Model confidence metrics to evaluate categorization quality

## Future Enhancements

Potential areas for improvement:
- Multiple category support (primary/secondary categories)
- Filter and search capabilities within the UI
- Export functionality for categorized data
- Historical trend analysis
- User-defined categories
- Bulk recategorization tools

## Getting Started

### Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install streamlit pandas plotly
   ```

### Running the App

1. Navigate to the project directory
2. Launch the app:
   ```
   streamlit run review_transactions.py
   ```
3. The app will open in your default web browser

### Adding Your Transactions

1. Place your transaction CSV files in the `data/to_categorize/` directory
2. Files should have columns for date, description, amount, and optionally extended details
3. Launch the app and select your file from the dropdown 