# Model Evaluation Framework

This document describes the model evaluation framework used in the Financial Transaction Categorizer project. The framework provides a standardized way to evaluate and compare different categorization models (Random Forest and Llama) using a consistent holdout set.

## Overview

The evaluation framework consists of several components:
- Holdout set creation and management
- Model evaluation and scoring
- Results storage and tracking
- Model comparison utilities

## Holdout Set

### Creation
- The holdout set is created from the full dataset
- Default size: 20% of the full dataset
- Stratified by category to maintain distribution
- Stored in `models/holdout/holdout_data.csv`

### Regeneration
The holdout set can be regenerated using `regenerate_holdout.py`:
```bash
python regenerate_holdout.py
```

## Model Evaluation

### Metrics
The framework tracks several key metrics:
- Overall accuracy
- Per-category metrics:
  - Precision
  - Recall
  - F1 score
- Confidence scores:
  - Mean confidence
  - Standard deviation
  - Distribution (very low, low, medium, high)
- Processing time per transaction
- Confusion matrix

### Evaluation Process
1. Load holdout set
2. Make predictions for each transaction
3. Calculate metrics
4. Store results in multiple formats:
   - Detailed CSV files
   - JSON history files
   - Database tables

## Model Comparison

### Comparison Script
The `compare_models_test.py` script provides a comprehensive comparison between models:
```bash
python compare_models_test.py
```

### Comparison Features
- Side-by-side accuracy comparison
- Processing time analysis
- Identification of model disagreements
- Example transactions where models disagree
- Error handling and reporting

## Results Storage

### File-based Storage
- CSV files: Detailed predictions and comparisons
- JSON files: Evaluation metrics and history
- Location: `data/model_history/`

### Database Storage
Two main tables track model performance over time:
1. `model_versions`:
   - Model version information
   - Training metadata
   - Dataset sizes

2. `model_scores`:
   - Evaluation metrics
   - Timestamps
   - Model version references

## Usage Guidelines

### Running Evaluations
1. Ensure holdout set exists
2. Run model comparison script
3. Review results in:
   - Console output
   - Generated CSV files
   - Database tables

### Best Practices
- Keep holdout set consistent between evaluations
- Document any changes to the evaluation process
- Review model disagreements for insights
- Monitor processing times for performance impact

## Troubleshooting

### Common Issues
1. Missing holdout set
   - Solution: Run `regenerate_holdout.py`

2. Database connection issues
   - Check database configuration
   - Verify table existence

3. Model loading failures
   - Verify model file paths
   - Check model version compatibility

### Error Handling
- Scripts include comprehensive error handling
- Failed evaluations are logged
- Partial results are saved when possible

## Future Improvements

Potential enhancements to consider:
- Automated model selection based on metrics
- Visualization of evaluation results
- Integration with CI/CD pipelines
- Enhanced error analysis tools
- Performance optimization for large datasets 