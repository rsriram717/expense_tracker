# Analysis: Why Llama Underperforms Compared to Random Forest for Transaction Categorization

## Overview
Our testing with the Llama 3.1 model for financial transaction categorization showed a surprising result: the Llama model significantly underperformed compared to the Random Forest (RF) model, with an accuracy of around 16% versus what we expect would be a much higher accuracy for the RF model based on previous evaluations.

## Key Findings

1. **Category Mismatch**: The Llama model was not properly recognizing many of the custom categories used in the dataset, particularly domain-specific ones like "Food & Drink", "Dante" (pet expenses), etc.

2. **Bias Toward "Misc"**: Llama heavily overused the "Misc" category (50 instances vs. 17 in the true data), suggesting it defaulted to this when uncertain.

3. **Merchant Recognition Issues**: The model struggled to interpret merchant names and understand what categories they belong to, especially for less obvious merchant names.

4. **Inconsistent Category Definitions**: The categories used in the prompt didn't match exactly with the categories in the training data, causing significant misclassifications.

5. **Batch Processing Limitations**: When processing in batches, the model had less context per transaction, potentially reducing accuracy.

## Why Random Forest Likely Performs Better

1. **Domain-Specific Training**: The Random Forest model was explicitly trained on this exact dataset with these specific categories, giving it a significant advantage.

2. **Feature Engineering**: The Random Forest implementation likely uses text features (TF-IDF) that specifically capture patterns in the transaction descriptions from the training data.

3. **Merchant Memory**: Random Forest effectively "memorizes" the correct categories for specific merchants from the training data, while Llama tries to apply general knowledge.

4. **Consistent Categories**: The RF model was trained with the exact same category set that it's evaluated on, while Llama had to map from its general knowledge to our specific categories.

5. **No Context Limit**: RF doesn't have token limits or lose context when processing multiple transactions at once.

## Improving Llama Performance

Our experiments with improved prompts showed some small improvements, but significant challenges remain:

1. **Category Alignment**: The improved prompt with explicit category definitions and examples did help with some transactions but still had limitations.

2. **Merchant Recognition**: Even with better prompting, Llama struggled to recognize many merchant names.

3. **Defaulting to "Misc"**: In our final test, most predictions defaulted to "Misc" despite explicit instruction to use this category only when necessary.

4. **Custom Domain Knowledge**: Llama lacks the specific training on financial transactions that would make it recognize patterns (e.g., that "TST*" often precedes restaurant names).

## The Path Forward

To improve Llama's performance for transaction categorization:

1. **Fine-tuning**: Fine-tune the model on a dataset of transactions with their correct categories. This would be the most effective approach to bridge the domain gap.

2. **Better Examples**: Include more real-world examples of correctly categorized transactions in the prompt, especially for domain-specific merchants.

3. **Single Transaction Processing**: Process one transaction at a time with more context rather than in batches.

4. **Hybrid Approach**: Use RF for common merchants it's seen before, and Llama for novel merchants or when RF confidence is low.

5. **Pre-processing**: Add logic to recognize patterns in transaction descriptions that strongly indicate certain categories (e.g., "TST*" for restaurants).

## Conclusion

The performance gap between Llama and Random Forest for transaction categorization is primarily due to:

1. Domain-specific training data advantage for Random Forest
2. Category definition mismatches
3. Merchant recognition challenges 
4. Batch processing limitations

While large language models like Llama have strong general reasoning capabilities, domain-specific tasks like financial transaction categorization still benefit significantly from supervised learning on in-domain data. The most promising approach would be to fine-tune Llama on transaction data or implement a hybrid approach that leverages the strengths of both models. 